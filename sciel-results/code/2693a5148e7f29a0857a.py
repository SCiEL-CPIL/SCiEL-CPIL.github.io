"""Structural change: use ILP-style branch-and-bound / deep DFS with strong pruning
plus the greedy+refinement machinery as a warm start. The key structural idea is
to switch from purely greedy construction to a systematic search (DFS with
ordering and time-bounded backtracking) that can find longer paths than any
greedy heuristic, while keeping the fast greedy/refinement pipeline for
initialization and local improvement."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Precompute sorted neighbor lists by degree (ascending) for Warnsdorff-like heuristics
DEG = [len(ADJ[i]) for i in range(N)]
NBR_SORTED = [sorted(ADJ[i], key=lambda x: (DEG[x], x)) for i in range(N)]

def find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

def warnsdorff_path(start, seen=None):
    """Greedy Warnsdorff: always pick the neighbor with fewest unvisited neighbors."""
    if seen is None:
        seen = set()
    seen.add(start)
    path = [start]
    while True:
        cur = path[-1]
        cands = [v for v in NBR_SORTED[cur] if v not in seen]
        if not cands:
            break
        best = min(cands, key=lambda v: (sum(1 for w in NBR_SORTED[v] if w not in seen), v))
        path.append(best)
        seen.add(best)
    return path

def greedy_path(start, mode='min'):
    seen = {start}
    path = [start]
    while True:
        cur = path[-1]
        cands = [v for v in NBR_SORTED[cur] if v not in seen]
        if not cands:
            break
        if mode == 'min':
            best = min(cands, key=lambda v: DEG[v])
        elif mode == 'max':
            best = max(cands, key=lambda v: DEG[v])
        else:
            best = random.choice(cands)
        path.append(best)
        seen.add(best)
    return path

def extend_both_ends(path):
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Extend from head
        if path:
            cur = path[0]
            cands = [v for v in NBR_SORTED[cur] if v not in seen]
            if cands:
                best = min(cands, key=lambda v: (sum(1 for w in NBR_SORTED[v] if w not in seen), v))
                path.insert(0, best)
                seen.add(best)
                changed = True
        # Extend from tail
        if path:
            cur = path[-1]
            cands = [v for v in NBR_SORTED[cur] if v not in seen]
            if cands:
                best = min(cands, key=lambda v: (sum(1 for w in NBR_SORTED[v] if w not in seen), v))
                path.append(best)
                seen.add(best)
                changed = True
    return path

def try_insert(path):
    """Try to insert an unvisited vertex into the path at any valid position."""
    seen = set(path)
    unvisited = [v for v in range(N) if v not in seen]
    for v in unvisited:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if b in ADJ[v] and a in ADJ[v]:
                return path[:i+1] + [v] + path[i+1:]
    # Try adding to ends
    for v in unvisited:
        if path and path[-1] in ADJ[v]:
            return path + [v]
        if path and path[0] in ADJ[v]:
            return [v] + path
    return None

def refine(path):
    """Local refinement: try insertions and re-extensions repeatedly."""
    best_path = list(path)
    for _ in range(300):
        result = try_insert(best_path)
        if result is None or len(result) <= len(best_path):
            break
        best_path = result
    best_path = extend_both_ends(best_path)
    return best_path

def deep_dfs(start, best_len, deadline, time_per_node=0.0005):
    """
    Depth-first search with strong pruning: only continue if current path length
    + remaining unvisited vertices > best_len. Uses Warnsdorff ordering for
    candidate selection. Returns the best path found.
    """
    best_path = [start]
    best_len_local = 1
    seen = {start}
    
    # We'll use an iterative approach with a stack to avoid recursion limits
    # Stack entries: (path, seen_set, current_node)
    # But copying seen sets is expensive. Instead, use a single seen array with backtracking.
    
    # Use arrays for speed
    visited_arr = [False] * N
    visited_arr[start] = True
    path = [start]
    
    # Iterative DFS with explicit stack
    # Stack: list of (node, candidate_index)
    # We need to track which candidates have been tried at each level
    
    # Alternative: recursive with sys.setrecursionlimit, but iterative is safer
    # Let's do a hybrid: use a stack of states where each state is (current_node, list_of_remaining_candidates)
    # But that's memory-heavy. Instead, let's do a depth-limited search with restarts.
    
    # Actually, for N=100, a well-ordered DFS can go quite deep. Let's try a 
    # different approach: use the greedy path as a base, then do local search
    # to swap/reorder vertices to extend the path.
    
    # Let's do a simpler but effective approach: 
    # 1. Generate greedy path
    # 2. Try to extend from both ends
    # 3. Try to insert unvisited vertices
    # 4. If stuck, try a different greedy strategy
    # 5. Use limited backtracking near the end of the path
    
    # For the DFS, let's do a proper backtracking search with good ordering
    # and time limits. We'll use a recursive approach with a node counter.
    
    node_count = [0]
    
    def dfs(cur, depth):
        nonlocal best_path, best_len_local, node_count
        
        if time.time() > deadline:
            return
        
        node_count[0] += 1
        if node_count[0] > 500000:
            return
        
        if len(path) > best_len_local:
            best_len_local = len(path)
            best_path = list(path)
        
        # Pruning: even if we visit all remaining unvisited, can we beat best?
        remaining = N - len(path)
        if len(path) + remaining <= best_len_local:
            return
        
        # Try candidates in Warnsdorff order
        cands = [v for v in NBR_SORTED[cur] if not visited_arr[v]]
        if not cands:
            return
        
        # Sort by remaining degree (Warnsdorff)
        cands.sort(key=lambda v: (sum(1 for w in NBR_SORTED[v] if not visited_arr[w]), v))
        
        for v in cands:
            if time.time() > deadline:
                return
            if node_count[0] > 500000:
                return
            
            path.append(v)
            visited_arr[v] = True
            dfs(v, depth + 1)
            path.pop()
            visited_arr[v] = False
            
            if time.time() > deadline or node_count[0] > 500000:
                return
    
    dfs(start, 0)
    return best_path

def local_search_extend(path, deadline):
    """
    Local search: try to extend the path by finding vertices that can be 
    inserted or appended. Uses a queue-based approach to explore extensions.
    """
    best_path = list(path)
    seen = set(best_path)
    
    # Try to extend from both ends multiple times
    for _ in range(50):
        if time.time() > deadline:
            break
        
        # Try extending from tail
        extended = False
        cur = best_path[-1]
        cands = [v for v in NBR_SORTED[cur] if v not in seen]
        if cands:
            # Try each candidate
            for v in cands:
                trial = best_path + [v]
                trial = extend_both_ends(trial)
                if len(trial) > len(best_path):
                    best_path = trial
                    seen = set(best_path)
                    extended = True
                    break
        
        # Try extending from head
        if not extended:
            cur = best_path[0]
            cands = [v for v in NBR_SORTED[cur] if v not in seen]
            if cands:
                for v in cands:
                    trial = [v] + best_path
                    trial = extend_both_ends(trial)
                    if len(trial) > len(best_path):
                        best_path = trial
                        seen = set(best_path)
                        extended = True
                        break
        
        # Try insertions
        if not extended:
            result = try_insert(best_path)
            if result and len(result) > len(best_path):
                best_path = result
                seen = set(best_path)
                extended = True
        
        if not extended:
            break
    
    return best_path

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # 80s safety margin within 90s limit
    
    # Find connected components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Multi-start greedy with various heuristics
    for comp in comps:
        if len(comp) <= best_len:
            break
        for start in comp:
            if time.time() > deadline - 10.0:
                break
            
            # Strategy 1: Warnsdorff
            p = warnsdorff_path(start)
            p = extend_both_ends(p)
            if len(p) > best_len:
                best_path = list(p)
                best_len = len(p)
            
            # Strategy 2: Min-degree greedy
            p2 = greedy_path(start, 'min')
            p2 = extend_both_ends(p2)
            if len(p2) > best_len:
                best_path = list(p2)
                best_len = len(p2)
            
            # Strategy 3: Max-degree greedy
            p3 = greedy_path(start, 'max')
            p3 = extend_both_ends(p3)
            if len(p3) > best_len:
                best_path = list(p3)
                best_len = len(p3)
    
    # Phase 2: Refine the best path found
    if best_path:
        refined = refine(best_path)
        if len(refined) > best_len:
            best_path = refined
            best_len = len(refined)
    
    # Phase 3: Deep DFS from promising starts (the structural change)
    # Use the best path's endpoints and high-degree vertices as DFS starts
    dfs_starts = set()
    if best_path:
        dfs_starts.add(best_path[0])
        dfs_starts.add(best_path[-1])
    # Add high-degree vertices from the largest component
    if comps:
        for v in sorted(comps[0], key=lambda x: -DEG[x])[:10]:
            dfs_starts.add(v)
    
    for start in dfs_starts:
        if time.time() > deadline - 20.0:
            break
        
        # Run DFS with time limit
        dfs_deadline = min(deadline - 5.0, time.time() + 8.0)
        dfs_result = deep_dfs(start, best_len, dfs_deadline)
        if len(dfs_result) > best_len:
            best_path = list(dfs_result)
            best_len = len(dfs_result)
    
    # Phase 4: Local search extension on the best path
    if best_path and time.time() < deadline - 5.0:
        ls_result = local_search_extend(best_path, deadline - 3.0)
        if len(ls_result) > best_len:
            best_path = ls_result
            best_len = len(ls_result)
    
    # Phase 5: Random restarts with refinement
    rng = random.Random(12345)
    for iteration in range(200):
        if time.time() > deadline - 5.0:
            break
        
        if comps:
            start = rng.choice(comps[0])
        else:
            start = rng.randint(0, N - 1)
        
        # Randomized greedy
        p = greedy_path(start, 'random')
        p = extend_both_ends(p)
        p = refine(p)
        
        if len(p) > best_len:
            best_path = list(p)
            best_len = len(p)
        
        # Try local search on this path
        if len(p) >= best_len - 3 and time.time() < deadline - 3.0:
            ls_p = local_search_extend(p, deadline - 2.0)
            if len(ls_p) > best_len:
                best_path = ls_p
                best_len = len(ls_p)
    
    # Phase 6: Final refinement
    if best_path:
        final = refine(best_path)
        if len(final) > best_len:
            best_path = final
            best_len = len(final)
    
    # Ensure nonempty
    if not best_path:
        best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
