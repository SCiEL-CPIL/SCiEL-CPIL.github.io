"""Longest simple path in the pinned G(100, 1.8/100) graph."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']
DEADLINE = 85.0  # safety margin before 90s kill

# Build adjacency as bitmasks for fast membership tests
adj_mask = [0] * N
adj_list = [[] for _ in range(N)]
for u, v in EDGES:
    adj_mask[u] |= (1 << v)
    adj_mask[v] |= (1 << u)
    adj_list[u].append(v)
    adj_list[v].append(u)

# Precompute degree-sorted neighbor lists for each vertex
for i in range(N):
    adj_list[i].sort(key=lambda x: len(adj_list[x]))

# Also keep set-based adjacency for local search
adj_set = [set(adj_list[i]) for i in range(N)]


def time_remaining():
    return DEADLINE - time.time()


def greedy_path(start, adj_mask, adj_list, rng=None):
    """Greedy path extension from start using a heuristic."""
    path = [start]
    seen_mask = 1 << start
    while True:
        last = path[-1]
        # Find unvisited neighbors
        avail = adj_mask[last] & ~seen_mask
        if avail == 0:
            break
        # Choose neighbor with fewest unvisited neighbors (Warnsdorff-like)
        best_v = -1
        best_score = (float('inf'), float('inf'))
        while avail:
            v = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            unvisited_deg = (adj_mask[v] & ~seen_mask).bit_count()
            if rng:
                score = (unvisited_deg, rng.random())
            else:
                score = (unvisited_deg, v)
            if score < best_score:
                best_score = score
                best_v = v
        if best_v < 0:
            break
        path.append(best_v)
        seen_mask |= (1 << best_v)
    return path


def try_extend_endpoints(path, adj_mask, adj_list, rng):
    """Try to extend the path from either endpoint."""
    path = list(path)
    seen_mask = 0
    for v in path:
        seen_mask |= (1 << v)
    improved = True
    while improved:
        improved = False
        # Try extending from the end
        last = path[-1]
        avail = adj_mask[last] & ~seen_mask
        if avail:
            best_v = -1
            best_score = (float('inf'), float('inf'))
            while avail:
                v = (avail & -avail).bit_length() - 1
                avail &= avail - 1
                unvisited_deg = (adj_mask[v] & ~seen_mask).bit_count()
                score = (unvisited_deg, v)
                if score < best_score:
                    best_score = score
                    best_v = v
            if best_v >= 0:
                path.append(best_v)
                seen_mask |= (1 << best_v)
                improved = True
        # Try extending from the front
        first = path[0]
        avail = adj_mask[first] & ~seen_mask
        if avail:
            best_v = -1
            best_score = (float('inf'), float('inf'))
            while avail:
                v = (avail & -avail).bit_length() - 1
                avail &= avail - 1
                unvisited_deg = (adj_mask[v] & ~seen_mask).bit_count()
                score = (unvisited_deg, v)
                if score < best_score:
                    best_score = score
                    best_v = v
            if best_v >= 0:
                path.insert(0, best_v)
                seen_mask |= (1 << best_v)
                improved = True
    return path


def segment_reversal(path, adj_mask, adj_list, rng):
    """Try reversing a segment to expose new endpoints and extend."""
    n = len(path)
    if n < 3:
        return path
    best_path = list(path)
    best_len = n
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            if i > 0:
                if not (adj_mask[path[i-1]] >> path[j]) & 1:
                    continue
            if j < n - 1:
                if not (adj_mask[path[j+1]] >> path[i]) & 1:
                    continue
            
            new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
            extended = try_extend_endpoints(new_path, adj_mask, adj_list, rng)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = extended
    
    return best_path


def local_search(path, adj_mask, adj_list, rng, max_iters=200):
    """Local search: try segment reversals and endpoint extensions."""
    path = list(path)
    best = list(path)
    best_len = len(path)
    
    for it in range(max_iters):
        if time_remaining() < 2.0:
            break
        candidate = segment_reversal(path, adj_mask, adj_list, rng)
        if len(candidate) > best_len:
            best = candidate
            best_len = len(candidate)
            path = candidate
        else:
            if len(path) >= 4 and rng.random() < 0.5:
                i = rng.randint(0, len(path) - 3)
                j = rng.randint(i + 1, len(path) - 1)
                if i > 0 and not (adj_mask[path[i-1]] >> path[j]) & 1:
                    continue
                if j < len(path) - 1 and not (adj_mask[path[j+1]] >> path[i]) & 1:
                    continue
                new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
                extended = try_extend_endpoints(new_path, adj_mask, adj_list, rng)
                if len(extended) > best_len:
                    best = extended
                    best_len = len(extended)
                    path = extended
                else:
                    path = new_path
            else:
                break
    
    return best


def dfs_longest_path(adj_mask, adj_list, timeout=5.0, start_vertices=None):
    """DFS with backtracking using bitmask for visited set."""
    start_time = time.time()
    best_path = []
    best_len = 0
    
    if start_vertices is None:
        start_vertices = list(range(N))
    
    def dfs(path, seen_mask):
        nonlocal best_path, best_len
        if time.time() - start_time > timeout:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        last = path[-1]
        avail = adj_mask[last] & ~seen_mask
        if avail == 0:
            return
        
        # Sort candidates by degree (pre-sorted, but filter unvisited)
        candidates = []
        while avail:
            v = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            candidates.append(v)
        # candidates already in degree order from adj_list pre-sort? No, need to re-sort
        candidates.sort(key=lambda x: len(adj_list[x]))
        
        for v in candidates:
            if time.time() - start_time > timeout:
                return
            path.append(v)
            dfs(path, seen_mask | (1 << v))
            path.pop()
    
    for start in start_vertices:
        if time.time() - start_time > timeout:
            break
        path = [start]
        dfs(path, 1 << start)
    
    return best_path


def bitmask_dfs_targeted(adj_mask, adj_list, timeout=5.0, target_vertices=None):
    """Targeted DFS focusing on vertices likely to be in long paths."""
    start_time = time.time()
    best_path = []
    best_len = 0
    
    if target_vertices is None:
        # Use vertices with highest degree as starting points
        target_vertices = sorted(range(N), key=lambda x: len(adj_list[x]), reverse=True)
    
    def dfs(path, seen_mask):
        nonlocal best_path, best_len
        if time.time() - start_time > timeout:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        last = path[-1]
        avail = adj_mask[last] & ~seen_mask
        if avail == 0:
            return
        
        candidates = []
        while avail:
            v = (avail & -avail).bit_length() - 1
            avail &= avail - 1
            candidates.append(v)
        candidates.sort(key=lambda x: len(adj_list[x]))
        
        for v in candidates:
            if time.time() - start_time > timeout:
                return
            path.append(v)
            dfs(path, seen_mask | (1 << v))
            path.pop()
    
    for start in target_vertices:
        if time.time() - start_time > timeout:
            break
        path = [start]
        dfs(path, 1 << start)
    
    return best_path


def random_walk_path(adj_mask, adj_list, rng, steps=1000):
    """Generate a path via random walk, then extend greedily."""
    start = rng.randint(0, N - 1)
    path = [start]
    seen_mask = 1 << start
    
    for _ in range(steps):
        last = path[-1]
        avail = adj_mask[last] & ~seen_mask
        if avail == 0:
            break
        # Pick random unvisited neighbor
        bits = avail
        v = (bits & -bits).bit_length() - 1
        # Random selection
        count = bits.bit_count()
        idx = rng.randint(0, count - 1)
        temp = bits
        for _ in range(idx):
            temp &= temp - 1
        v = (temp & -temp).bit_length() - 1
        path.append(v)
        seen_mask |= (1 << v)
    
    path = try_extend_endpoints(path, adj_mask, adj_list, rng)
    return path


def two_opt_improve(path, adj_mask, adj_list, rng, max_iters=100):
    """Try 2-opt style moves: remove and reinsert vertices."""
    path = list(path)
    best = list(path)
    best_len = len(path)
    seen_mask = 0
    for v in path:
        seen_mask |= (1 << v)
    
    for it in range(max_iters):
        if time_remaining() < 2.0:
            break
        improved = False
        
        # Try removing a vertex and reinserting elsewhere
        if len(path) > 3:
            for pos in range(1, len(path) - 1):
                v = path[pos]
                prev_v = path[pos - 1]
                next_v = path[pos + 1]
                # Check if prev_v and next_v are connected
                if (adj_mask[prev_v] >> next_v) & 1:
                    # Remove v
                    new_path = path[:pos] + path[pos+1:]
                    new_seen = seen_mask & ~(1 << v)
                    # Try to reinsert v somewhere
                    for insert_pos in range(len(new_path) + 1):
                        if insert_pos == 0:
                            if (adj_mask[v] >> new_path[0]) & 1:
                                candidate = [v] + new_path
                        elif insert_pos == len(new_path):
                            if (adj_mask[v] >> new_path[-1]) & 1:
                                candidate = new_path + [v]
                        else:
                            if ((adj_mask[v] >> new_path[insert_pos-1]) & 1) and ((adj_mask[v] >> new_path[insert_pos]) & 1):
                                candidate = new_path[:insert_pos] + [v] + new_path[insert_pos:]
                            else:
                                continue
                        # Extend
                        extended = try_extend_endpoints(candidate, adj_mask, adj_list, rng)
                        if len(extended) > best_len:
                            best = extended
                            best_len = len(extended)
                            path = extended
                            seen_mask = 0
                            for x in path:
                                seen_mask |= (1 << x)
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            break
    
    return best


def construct():
    start_time = time.time()
    DEADLINE_LOCAL = start_time + DEADLINE
    
    def time_ok():
        return time.time() < DEADLINE_LOCAL - 2.0
    
    rng = random.Random(42)
    
    # Phase 1: Greedy from all starts
    best_path = []
    best_len = 0
    for s in range(N):
        if not time_ok():
            break
        p = greedy_path(s, adj_mask, adj_list, None)
        p = try_extend_endpoints(p, adj_mask, adj_list, rng)
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
    
    # Phase 2: Local search on best path
    if time_ok():
        best_path = local_search(best_path, adj_mask, adj_list, rng, max_iters=500)
        best_len = len(best_path)
    
    # Phase 3: Targeted DFS from high-degree vertices
    if time_ok():
        remaining = time_remaining()
        dfs_timeout = min(5.0, max(1.0, remaining * 0.2))
        dfs_path = bitmask_dfs_targeted(adj_mask, adj_list, timeout=dfs_timeout)
        if len(dfs_path) > best_len:
            best_len = len(dfs_path)
            best_path = dfs_path
            if time_ok():
                best_path = local_search(best_path, adj_mask, adj_list, rng, max_iters=300)
                best_len = len(best_path)
    
    # Phase 4: 2-opt improvement
    if time_ok():
        best_path = two_opt_improve(best_path, adj_mask, adj_list, rng, max_iters=200)
        best_len = len(best_path)
    
    # Phase 5: Multiple random restarts with local search
    if time_ok():
        for _ in range(30):
            if not time_ok():
                break
            p = random_walk_path(adj_mask, adj_list, rng, steps=rng.randint(50, 200))
            p = local_search(p, adj_mask, adj_list, rng, max_iters=150)
            if len(p) > best_len:
                best_len = len(p)
                best_path = p
    
    # Phase 6: More local search and 2-opt on the best
    if time_ok():
        remaining = time_remaining()
        if remaining > 3.0:
            best_path = local_search(best_path, adj_mask, adj_list, rng, max_iters=int(remaining * 40))
            best_len = len(best_path)
            if time_ok():
                best_path = two_opt_improve(best_path, adj_mask, adj_list, rng, max_iters=int(remaining * 20))
                best_len = len(best_path)
    
    # Phase 7: Final DFS with more time if available
    if time_ok():
        remaining = time_remaining()
        if remaining > 5.0:
            dfs_path = dfs_longest_path(adj_mask, adj_list, timeout=remaining * 0.4)
            if len(dfs_path) > best_len:
                best_len = len(dfs_path)
                best_path = dfs_path
                if time_ok():
                    best_path = local_search(best_path, adj_mask, adj_list, rng, max_iters=200)
                    best_len = len(best_path)
                    best_path = two_opt_improve(best_path, adj_mask, adj_list, rng, max_iters=100)
                    best_len = len(best_path)
    
    # Final: ensure we have a valid path
    if not best_path:
        best_path = [0]
    
    # Validate
    for i in range(len(best_path) - 1):
        if not (adj_mask[best_path[i]] >> best_path[i+1]) & 1:
            best_path = best_path[:i+1]
            break
    
    return best_path
# EVOLVE-BLOCK-END
