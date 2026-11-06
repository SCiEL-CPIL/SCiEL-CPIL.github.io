"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Pinned graph data from seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency structure once
def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return adj

ADJ = build_adj()

# Precompute degrees
DEG = [len(adj) for adj in ADJ]

def path_is_valid(path):
    """Verify all consecutive pairs are edges and vertices are distinct."""
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if path[i+1] not in ADJ[path[i]]:
            return False
    return True

def greedy_grow(start, adj, visited=None, strategy='least_remaining'):
    """Grow a path greedily from start vertex."""
    path = [start]
    seen = {start}
    while True:
        cur = path[-1]
        candidates = adj[cur] - seen
        if not candidates:
            break
        if strategy == 'least_remaining':
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        elif strategy == 'most_remaining':
            v = max(candidates, key=lambda w: (len(adj[w] - seen), -w))
        elif strategy == 'random':
            v = random.choice(list(candidates))
        else:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def try_extend_both_ends(path, adj):
    """Try to extend a path from both ends."""
    path = list(path)
    seen = set(path)
    changed = True
    while changed:
        changed = False
        # Try extending from start
        cur = path[0]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Try extending from end
        cur = path[-1]
        candidates = adj[cur] - seen
        if candidates:
            v = min(candidates, key=lambda w: (len(adj[w] - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def try_insert_vertex(path, adj, vertex):
    """Try to insert a vertex into the path at a valid position."""
    if vertex in path:
        return None
    # Check if vertex can be inserted between any two consecutive vertices
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            new_path = path[:i+1] + [vertex] + path[i+1:]
            return new_path
    # Check if vertex can be added at either end
    if vertex in adj[path[0]]:
        return [vertex] + path
    if vertex in adj[path[-1]]:
        return path + [vertex]
    return None

def local_search_improve(path, adj, max_iter=100):
    """Try to improve a path by inserting unvisited vertices."""
    if len(path) < 2:
        return path
    best_path = list(path)
    best_len = len(best_path)
    seen = set(best_path)
    
    for _ in range(max_iter):
        improved = False
        # Try inserting each unvisited vertex
        for v in range(N):
            if v in seen:
                continue
            new_path = try_insert_vertex(best_path, adj, v)
            if new_path is not None:
                best_path = new_path
                best_len = len(best_path)
                seen.add(v)
                improved = True
                break
        if not improved:
            break
    
    # After insertion, try extending both ends
    best_path = try_extend_both_ends(best_path, adj)
    return best_path

def beam_search(start, adj, beam_width=50, max_depth=None):
    """Beam search for longest path starting from start."""
    if max_depth is None:
        max_depth = N
    # Each beam entry: (path, seen_set)
    beam = [([start], {start})]
    best_path = [start]
    
    for depth in range(max_depth):
        if not beam:
            break
        new_beam = []
        seen_entries = set()
        for path, seen in beam:
            cur = path[-1]
            candidates = sorted(adj[cur] - seen, key=lambda w: (len(adj[w] - seen), w))
            for v in candidates:
                key = (path[-1], v)
                if key in seen_entries:
                    continue
                new_path = path + [v]
                new_seen = seen | {v}
                new_beam.append((new_path, new_seen))
                seen_entries.add(key)
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        if not new_beam:
            break
        # Keep top beam_width by path length (all same length at this depth, so just keep first)
        new_beam.sort(key=lambda x: (len(x[0]), -min(len(adj[v] - x[1]) for v in [x[0][-1]])))
        beam = new_beam[:beam_width]
    
    return best_path

def random_walk_with_backtracking(start, adj, num_tries=100):
    """Multiple random walks with occasional backtracking."""
    best_path = []
    for t in range(num_tries):
        rng = random.Random(start * 1000 + t)
        path = [start]
        seen = {start}
        for _ in range(N):
            cur = path[-1]
            candidates = list(adj[cur] - seen)
            if not candidates:
                break
            # Weight by remaining degree
            weights = [len(adj[v] - seen) + 1 for v in candidates]
            v = rng.choices(candidates, weights=weights)[0]
            path.append(v)
            seen.add(v)
        # Try to extend both ends
        path = try_extend_both_ends(path, adj)
        if len(path) > len(best_path):
            best_path = list(path)
    return best_path

def dfs_longest_path(start, adj, time_limit=5.0):
    """DFS to find longest path starting from start with time limit."""
    start_time = time.time()
    best_path = [start]
    best_len = 1
    
    def dfs(path, seen):
        nonlocal best_path, best_len
        if time.time() - start_time > time_limit:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        cur = path[-1]
        for v in adj[cur]:
            if v not in seen:
                new_seen = seen | {v}
                dfs(path + [v], new_seen)
    
    dfs([start], {start})
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    best_path = []
    best_len = 0
    
    # Phase 1: Greedy from all starts with multiple strategies
    for start in range(N):
        if time.time() > deadline:
            break
        for strategy in ['least_remaining', 'most_remaining']:
            path = greedy_grow(start, ADJ, strategy=strategy)
            # Extend both ends
            path = try_extend_both_ends(path, ADJ)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
    
    # Phase 2: Local search improvement with vertex insertion
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, ADJ, max_iter=200)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Phase 3: Random walks from promising starts
    if time.time() < deadline:
        # Prioritize starts that led to longer paths
        for start in range(N):
            if time.time() > deadline:
                break
            path = random_walk_with_backtracking(start, ADJ, num_tries=10)
            path = try_extend_both_ends(path, ADJ)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            # Apply local search to improved path
            if len(path) > best_len - 5 and time.time() < deadline:
                improved = local_search_improve(path, ADJ, max_iter=50)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = improved
    
    # Phase 4: Beam search from best starts
    if time.time() < deadline:
        # Try beam search from a few promising starts
        promising_starts = [best_path[0], best_path[-1], 0, 3, 8, 9, 19, 24]
        promising_starts = list(set(promising_starts))
        for start in promising_starts:
            if time.time() > deadline:
                break
            path = beam_search(start, ADJ, beam_width=50, max_depth=N)
            path = try_extend_both_ends(path, ADJ)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            # Apply local search to beam search result
            if time.time() < deadline:
                improved = local_search_improve(path, ADJ, max_iter=50)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = improved
    
    # Phase 5: DFS from a few promising starts with time limit
    if time.time() < deadline:
        dfs_starts = [best_path[0], best_path[-1], 0, 3, 8, 9, 19, 24]
        dfs_starts = list(set(dfs_starts))
        for start in dfs_starts:
            if time.time() > deadline:
                break
            remaining_time = deadline - time.time()
            if remaining_time < 1.0:
                break
            path = dfs_longest_path(start, ADJ, time_limit=min(remaining_time * 0.8, 5.0))
            path = try_extend_both_ends(path, ADJ)
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            # Apply local search to DFS result
            if time.time() < deadline:
                improved = local_search_improve(path, ADJ, max_iter=50)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_path = improved
    
    # Phase 6: Final aggressive local search on best
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, ADJ, max_iter=500)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved
    
    # Ensure we have a valid nonempty path
    if not best_path:
        # Fallback: single vertex
        best_path = [0]
    
    # Final validation
    if not path_is_valid(best_path):
        # Try to fix or fall back
        # Find a valid edge
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]
    
    return best_path
# EVOLVE-BLOCK-END
