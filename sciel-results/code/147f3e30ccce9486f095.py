"""Long-path search in the pinned G(100, 1.8/100) graph."""
import time
import random

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

# Build adjacency list
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)

# Convert to lists for faster iteration
adj_list = [list(s) for s in adj]

def warnsdorff_extend(path, seen, deadline):
    """Greedy Warnsdorff extension: always pick the neighbor with fewest unvisited neighbors."""
    while time.time() < deadline:
        cur = path[-1]
        candidates = [v for v in adj_list[cur] if v not in seen]
        if not candidates:
            break
        # Warnsdorff rule: pick candidate with fewest onward moves
        candidates.sort(key=lambda v: (len([w for w in adj_list[v] if w not in seen]), v))
        path.append(candidates[0])
        seen.add(candidates[0])
    return path

def iddfs_search(start, deadline):
    """
    Iterative deepening DFS with Warnsdorff ordering and backtracking.
    Tries to find a path longer than current best.
    """
    best_path = []
    
    # First, get a quick greedy path from this start
    greedy_path = [start]
    greedy_seen = {start}
    greedy_path = warnsdorff_extend(greedy_path, greedy_seen, deadline)
    if len(greedy_path) > len(best_path):
        best_path = list(greedy_path)
    
    # Now try to improve with limited backtracking
    # We'll do a DFS that explores alternatives at each step, but with a depth limit
    # and Warnsdorff ordering to keep it fast
    
    # Try increasing depth limits
    max_depth = len(greedy_path) + 5  # try to go a bit beyond greedy
    
    for target_len in range(len(greedy_path) + 1, min(max_depth, N + 1)):
        if time.time() > deadline:
            break
        
        found = _dfs_extend([start], {start}, target_len, deadline)
        if found is not None:
            if len(found) > len(best_path):
                best_path = found
            break
    
    # Also try a more exhaustive search with limited backtracking
    # Use a randomized approach: try different orderings
    for trial in range(3):
        if time.time() > deadline:
            break
        path = [start]
        seen = {start}
        _randomized_dfs(path, seen, deadline, max_tries=5000)
        if len(path) > len(best_path):
            best_path = list(path)
    
    return best_path

def _dfs_extend(path, seen, target_len, deadline):
    """
    DFS trying to extend path to target_len using Warnsdorff ordering.
    Returns the path if successful, None otherwise.
    """
    if len(path) >= target_len:
        return path
    if time.time() > deadline:
        return None
    
    cur = path[-1]
    candidates = [v for v in adj_list[cur] if v not in seen]
    if not candidates:
        return None
    
    # Sort by Warnsdorff rule
    candidates.sort(key=lambda v: (len([w for w in adj_list[v] if w not in seen]), v))
    
    for v in candidates:
        path.append(v)
        seen.add(v)
        result = _dfs_extend(path, seen, target_len, deadline)
        if result is not None:
            return result
        path.pop()
        seen.discard(v)
        if time.time() > deadline:
            return None
    
    return None

def _randomized_dfs(path, seen, deadline, max_tries):
    """
    Randomized DFS: at each step, try several candidates randomly.
    Modifies path and seen in place.
    """
    tries = 0
    while tries < max_tries and time.time() < deadline:
        cur = path[-1]
        candidates = [v for v in adj_list[cur] if v not in seen]
        if not candidates:
            break
        # Pick a random candidate, but bias toward Warnsdorff
        candidates.sort(key=lambda v: (len([w for w in adj_list[v] if w not in seen]), v))
        # Pick from top few or random
        if len(candidates) <= 3:
            v = candidates[0]
        else:
            v = random.choice(candidates[:max(2, len(candidates)//3)])
        path.append(v)
        seen.add(v)
        tries += 1
    # Continue extending greedily from here
    warnsdorff_extend(path, seen, deadline)

def local_refinement(path, deadline):
    """
    Try to improve the path by local moves: 
    1. Try extending from both ends
    2. Try swapping/rotating to find better extensions
    """
    best = list(path)
    
    # Try extending from the end
    while time.time() < deadline:
        cur = best[-1]
        candidates = [v for v in adj_list[cur] if v not in set(best)]
        if not candidates:
            break
        candidates.sort(key=lambda v: (len([w for w in adj_list[v] if w not in set(best)]), v))
        best.append(candidates[0])
    
    # Try extending from the beginning
    while time.time() < deadline:
        cur = best[0]
        candidates = [v for v in adj_list[cur] if v not in set(best)]
        if not candidates:
            break
        candidates.sort(key=lambda v: (len([w for w in adj_list[v] if w not in set(best)]), v))
        best.insert(0, candidates[0])
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # First, get a quick baseline with greedy from all starts
    best_path = []
    for s in range(N):
        if time.time() > deadline:
            break
        path = [s]
        seen = {s}
        path = warnsdorff_extend(path, seen, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Now do IDDFS from promising starts (those that gave longer greedy paths)
    # Sort starts by greedy path length
    greedy_lengths = []
    for s in range(N):
        if time.time() > deadline:
            break
        path = [s]
        seen = {s}
        path = warnsdorff_extend(path, seen, deadline)
        greedy_lengths.append((len(path), s))
    
    greedy_lengths.sort(reverse=True)
    
    # Try IDDFS from the top starts
    for _, s in greedy_lengths[:20]:
        if time.time() > deadline:
            break
        path = iddfs_search(s, deadline)
        if len(path) > len(best_path):
            best_path = path
    
    # Local refinement
    best_path = local_refinement(best_path, deadline)
    
    # One more round of refinement with different strategies
    if time.time() < deadline:
        # Try to extend from both ends more aggressively
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Extend end
            cur = best_path[-1]
            cands = [v for v in adj_list[cur] if v not in set(best_path)]
            if cands:
                cands.sort(key=lambda v: (len([w for w in adj_list[v] if w not in set(best_path)]), v))
                best_path.append(cands[0])
                improved = True
            # Extend start
            cur = best_path[0]
            cands = [v for v in adj_list[cur] if v not in set(best_path)]
            if cands:
                cands.sort(key=lambda v: (len([w for w in adj_list[v] if w not in set(best_path)]), v))
                best_path.insert(0, cands[0])
                improved = True
    
    # Final check: ensure validity
    # Verify all consecutive pairs are edges and all vertices distinct
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in adj[best_path[i]]:
            # This shouldn't happen, but truncate if it does
            best_path = best_path[:i+1]
            break
    
    return best_path
