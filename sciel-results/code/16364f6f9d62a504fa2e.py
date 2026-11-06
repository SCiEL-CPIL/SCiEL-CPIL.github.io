"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

# Pinned graph data from seed
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
EDGES = P['edges']

# Build adjacency as bitmask for O(1) operations
ADJ_MASK = [0] * N
for u, v in EDGES:
    ADJ_MASK[u] |= (1 << v)
    ADJ_MASK[v] |= (1 << u)

DEG = [ADJ_MASK[i].bit_count() for i in range(N)]

def path_is_valid(path):
    if not path:
        return False
    if len(path) != len(set(path)):
        return False
    for i in range(len(path) - 1):
        if not (ADJ_MASK[path[i]] >> path[i+1]) & 1:
            return False
    return True

def greedy_grow(start, strategy='least_remaining'):
    path = [start]
    seen_mask = 1 << start
    while True:
        cur = path[-1]
        candidates_mask = ADJ_MASK[cur] & ~seen_mask
        if candidates_mask == 0:
            break
        if strategy == 'least_remaining':
            best_v = -1
            best_key = (10**9, 10**9)
            cm = candidates_mask
            while cm:
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                rem = (ADJ_MASK[v] & ~seen_mask).bit_count()
                key = (rem, v)
                if key < best_key:
                    best_key = key
                    best_v = v
            v = best_v
        elif strategy == 'most_remaining':
            best_v = -1
            best_key = (-1, 10**9)
            cm = candidates_mask
            while cm:
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                rem = (ADJ_MASK[v] & ~seen_mask).bit_count()
                key = (rem, -v)
                if key > best_key:
                    best_key = key
                    best_v = v
            v = best_v
        else:
            cm = candidates_mask
            idx = random.getrandbits(32) % max(1, cm.bit_count())
            for _ in range(idx + 1):
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
        path.append(v)
        seen_mask |= (1 << v)
    return path

def try_extend_both_ends(path):
    path = list(path)
    seen_mask = 0
    for v in path:
        seen_mask |= (1 << v)
    changed = True
    while changed:
        changed = False
        cur = path[0]
        cands = ADJ_MASK[cur] & ~seen_mask
        if cands:
            best_v = -1
            best_key = (10**9, 10**9)
            cm = cands
            while cm:
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                rem = (ADJ_MASK[v] & ~seen_mask).bit_count()
                key = (rem, v)
                if key < best_key:
                    best_key = key
                    best_v = v
            path.insert(0, best_v)
            seen_mask |= (1 << best_v)
            changed = True
        cur = path[-1]
        cands = ADJ_MASK[cur] & ~seen_mask
        if cands:
            best_v = -1
            best_key = (10**9, 10**9)
            cm = cands
            while cm:
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                rem = (ADJ_MASK[v] & ~seen_mask).bit_count()
                key = (rem, v)
                if key < best_key:
                    best_key = key
                    best_v = v
            path.append(best_v)
            seen_mask |= (1 << best_v)
            changed = True
    return path

def try_insert_vertex(path, v):
    if (1 << v) & 0:
        return None
    seen_mask = 0
    for x in path:
        seen_mask |= (1 << x)
    if (1 << v) & seen_mask:
        return None
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if (ADJ_MASK[a] >> v) & 1 and (ADJ_MASK[b] >> v) & 1:
            return path[:i+1] + [v] + path[i+1:]
    if (ADJ_MASK[path[0]] >> v) & 1:
        return [v] + path
    if (ADJ_MASK[path[-1]] >> v) & 1:
        return path + [v]
    return None

def insert_all_possible(path, max_rounds=100):
    best_path = list(path)
    seen_mask = 0
    for v in best_path:
        seen_mask |= (1 << v)
    for _ in range(max_rounds):
        improved = False
        for v in range(N):
            if (1 << v) & seen_mask:
                continue
            new_path = try_insert_vertex(best_path, v)
            if new_path is not None:
                best_path = new_path
                seen_mask |= (1 << v)
                improved = True
                break
        if not improved:
            break
    return try_extend_both_ends(best_path)

def local_search_improve(path, max_iter=200):
    if len(path) < 3:
        return path
    best_path = list(path)
    best_len = len(best_path)
    seen_mask = 0
    for v in best_path:
        seen_mask |= (1 << v)

    for _ in range(max_iter):
        improved = False
        # Try inserting each unvisited vertex
        for v in range(N):
            if (1 << v) & seen_mask:
                continue
            new_path = try_insert_vertex(best_path, v)
            if new_path is not None:
                best_path = new_path
                best_len = len(best_path)
                seen_mask |= (1 << v)
                improved = True
                break
        if improved:
            continue
        # 2-opt: try replacing internal vertex with alternative connecting both neighbors
        for i in range(1, len(best_path) - 1):
            prev_v = best_path[i-1]
            next_v = best_path[i+1]
            common = ADJ_MASK[prev_v] & ADJ_MASK[next_v] & ~seen_mask
            if common:
                cm = common
                while cm:
                    cand = (cm & (-cm)).bit_length() - 1
                    cm &= cm - 1
                    new_path = list(best_path)
                    new_path[i] = cand
                    nm = 0
                    for x in new_path:
                        nm |= (1 << x)
                    extended = try_extend_both_ends(new_path)
                    if len(extended) > best_len:
                        best_path = extended
                        best_len = len(best_path)
                        seen_mask = 0
                        for x in best_path:
                            seen_mask |= (1 << x)
                        improved = True
                        break
            if improved:
                break
        if improved:
            continue
        # 3-opt: try removing two consecutive internal vertices and finding a replacement pair
        for i in range(1, len(best_path) - 2):
            a = best_path[i-1]
            b = best_path[i+2]
            # Find vertices x,y such that a-x-y-b is a valid subpath and x,y not in path
            # Simplified: try swapping segments
            pass
        if not improved:
            break

    return try_extend_both_ends(best_path)

def dfs_longest_path(start, time_limit=2.0):
    """Warnsdorff-guided DFS with backtracking."""
    start_time = time.time()
    best_path = [start]
    best_len = 1

    def dfs(path, seen_mask):
        nonlocal best_path, best_len
        if time.time() - start_time > time_limit:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        cur = path[-1]
        cands = ADJ_MASK[cur] & ~seen_mask
        if cands == 0:
            return
        # Order by Warnsdorff: prefer vertices with fewest unvisited neighbors
        nbrs = []
        cm = cands
        while cm:
            v = (cm & (-cm)).bit_length() - 1
            cm &= cm - 1
            rem = (ADJ_MASK[v] & ~seen_mask).bit_count()
            nbrs.append((rem, v))
        nbrs.sort()
        for _, v in nbrs:
            path.append(v)
            dfs(path, seen_mask | (1 << v))
            path.pop()
            if time.time() - start_time > time_limit:
                return

    dfs([start], 1 << start)
    return best_path

def beam_search(start, beam_width=30, max_depth=None):
    if max_depth is None:
        max_depth = N
    beam = [([start], 1 << start)]
    best_path = [start]
    for depth in range(max_depth):
        if not beam:
            break
        new_beam = []
        for path, seen_mask in beam:
            cur = path[-1]
            cands = ADJ_MASK[cur] & ~seen_mask
            if cands == 0:
                continue
            cm = cands
            while cm:
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                new_path = path + [v]
                new_seen = seen_mask | (1 << v)
                new_beam.append((new_path, new_seen))
                if len(new_path) > len(best_path):
                    best_path = list(new_path)
        if not new_beam:
            break
        new_beam.sort(key=lambda x: (len(x[0]), -(ADJ_MASK[x[0][-1]] & ~x[1]).bit_count()))
        beam = new_beam[:beam_width]
    return best_path

def random_walk(start, num_tries=10):
    best_path = []
    for t in range(num_tries):
        rng = random.Random(start * 1000 + t)
        path = [start]
        seen_mask = 1 << start
        for _ in range(N):
            cur = path[-1]
            cands = ADJ_MASK[cur] & ~seen_mask
            if cands == 0:
                break
            cm = cands
            idx = rng.getrandbits(32) % max(1, cm.bit_count())
            for _ in range(idx + 1):
                v = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
            path.append(v)
            seen_mask |= (1 << v)
        path = try_extend_both_ends(path)
        if len(path) > len(best_path):
            best_path = list(path)
    return best_path

def splice_paths(path1, path2):
    """Try to combine two paths by finding a valid splice point."""
    if not path1 or not path2:
        return max(path1, path2, key=len) if path1 or path2 else []
    seen1 = set(path1)
    seen2 = set(path2)
    common = seen1 & seen2
    if not common:
        # Try to connect end of one to start of other
        if (ADJ_MASK[path1[-1]] >> path2[0]) & 1:
            return path1 + path2
        if (ADJ_MASK[path2[-1]] >> path1[0]) & 1:
            return path2 + path1
        if (ADJ_MASK[path1[-1]] >> path2[-1]) & 1:
            return path1 + path2[::-1]
        if (ADJ_MASK[path2[-1]] >> path1[0]) & 1:
            return path2 + path1[::-1]
    return max(path1, path2, key=len)

def construct():
    start_time = time.time()
    deadline = start_time + 85

    best_path = []
    best_len = 0

    def consider(path):
        nonlocal best_path, best_len
        if path and len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    # Phase 1: Greedy from all starts with multiple strategies
    for start in range(N):
        if time.time() > deadline:
            break
        for strategy in ['least_remaining', 'most_remaining']:
            path = greedy_grow(start, strategy=strategy)
            path = try_extend_both_ends(path)
            path = insert_all_possible(path, max_rounds=50)
            consider(path)

    # Phase 2: Local search on best
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, max_iter=200)
        consider(improved)

    # Phase 3: Random walks from all starts
    if time.time() < deadline:
        for start in range(N):
            if time.time() > deadline:
                break
            path = random_walk(start, num_tries=5)
            path = try_extend_both_ends(path)
            path = insert_all_possible(path, max_rounds=50)
            consider(path)

    # Phase 4: DFS from promising starts with more time
    if time.time() < deadline:
        # Prioritize low-degree vertices (natural path endpoints) and best path endpoints
        promising = sorted(range(N), key=lambda v: (DEG[v], v))[:15]
        promising = list(dict.fromkeys(promising + [best_path[0], best_path[-1]]))
        for start in promising:
            if time.time() > deadline - 15:
                break
            path = dfs_longest_path(start, time_limit=3.0)
            path = try_extend_both_ends(path)
            path = insert_all_possible(path, max_rounds=100)
            consider(path)

    # Phase 5: Beam search with wider beams
    if time.time() < deadline:
        promising = list(dict.fromkeys([best_path[0], best_path[-1]] + sorted(range(N), key=lambda v: DEG[v])[:10]))
        for start in promising:
            if time.time() > deadline - 5:
                break
            path = beam_search(start, beam_width=60, max_depth=N)
            path = try_extend_both_ends(path)
            path = insert_all_possible(path, max_rounds=100)
            consider(path)

    # Phase 6: Final local search
    if time.time() < deadline and best_len > 0:
        improved = local_search_improve(best_path, max_iter=300)
        consider(improved)

    # Phase 7: Additional restarts with time remaining - focus on DFS from low-degree starts
    remaining = deadline - time.time()
    if remaining > 5:
        low_deg_starts = sorted(range(N), key=lambda v: DEG[v])[:20]
        for i in range(min(40, int(remaining * 8))):
            if time.time() > deadline - 3:
                break
            if i < len(low_deg_starts):
                start = low_deg_starts[i % len(low_deg_starts)]
            else:
                start = random.randint(0, N-1)
            # Alternate between greedy and short DFS
            if i % 3 == 0:
                path = dfs_longest_path(start, time_limit=1.5)
            elif i % 3 == 1:
                path = greedy_grow(start, strategy='least_remaining')
            else:
                path = greedy_grow(start, strategy='most_remaining')
            path = try_extend_both_ends(path)
            path = insert_all_possible(path, max_rounds=50)
            consider(path)

    if not best_path:
        best_path = [0]

    if not path_is_valid(best_path):
        for u, v in EDGES:
            best_path = [u, v]
            break
        else:
            best_path = [0]

    return best_path
# EVOLVE-BLOCK-END
