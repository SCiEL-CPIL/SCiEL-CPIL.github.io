import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_N = P['n']
_ADJ = [0] * _N
_ADJ_LIST = [[] for _ in range(_N)]
for _e in P['edges']:
    _ADJ[_e[0]] |= 1 << _e[1]
    _ADJ[_e[1]] |= 1 << _e[0]
    _ADJ_LIST[_e[0]].append(_e[1])
    _ADJ_LIST[_e[1]].append(_e[0])

def _bb_clique(cand, size, best, best_set):
    """Branch-and-bound: return (best_size, best_set_mask) found in this subtree."""
    while cand:
        # upper bound: current size + number of candidates
        if size + bin(cand).count('1') <= best:
            return best, best_set
        # pick a candidate vertex (highest-degree among candidates for pruning)
        v = max([i for i in range(_N) if (cand >> i) & 1],
                key=lambda x: bin(_ADJ[x] & cand).count('1'))
        # include v
        cand2 = cand & _ADJ[v]
        if size + 1 > best and cand2 == 0:
            # leaf, v alone extends
            if size + 1 > best:
                best = size + 1
                best_set = best_set | (1 << v)
        elif size + 1 > best:
            b2, s2 = _bb_clique(cand2, size + 1, best, best_set)
            best, best_set = b2, s2
        # exclude v
        cand &= ~(1 << v)
    return best, best_set

def _max_clique():
    """Iterative deepening max clique via branch-and-bound."""
    # seed with a greedy clique for a lower bound
    order = sorted(range(_N), key=lambda x: bin(_ADJ[x]).count('1'), reverse=True)
    seed = []
    rem = (1 << _N) - 1
    for v in order:
        if (rem >> v) & 1:
            seed.append(v)
            rem &= _ADJ[v]
    lower = len(seed)
    best_set = 0
    for v in seed:
        best_set |= 1 << v

    # try to find a clique of size lower+1, lower+2, ...
    target = lower + 1
    while target <= _N:
        # check feasibility with a quick BB that stops when size>=target
        found = _find_clique_of_size(target, (1 << _N) - 1)
        if found is None:
            break
        best_set = found
        lower = target
        target += 1
    return lower, best_set

def _find_clique_of_size(target, cand):
    """Return a mask of a clique of size >= target, or None. Uses BB with early stop."""
    result = [None]

    def dfs(c, size, cur):
        if result[0] is not None:
            return
        if size >= target:
            result[0] = cur
            return
        if size + bin(c).count('1') < target:
            return
        while c:
            # pick vertex with max degree in c
            v = max([i for i in range(_N) if (c >> i) & 1],
                    key=lambda x: bin(_ADJ[x] & c).count('1'))
            c2 = c & _ADJ[v]
            dfs(c2, size + 1, cur | (1 << v))
            if result[0] is not None:
                return
            c &= ~(1 << v)

    dfs(cand, 0, 0)
    return result[0]

def _bfs_path(src, dst, blocked):
    """BFS shortest path from src to dst avoiding blocked interior vertices."""
    if src == dst:
        return [src]
    if (1 << dst) & _ADJ[src]:
        return [src, dst]
    
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for nb in _ADJ_LIST[u]:
            if nb in parent:
                continue
            if nb != dst and nb in blocked:
                continue
            parent[nb] = u
            if nb == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append(nb)
    return None

def _generate_candidates(src, dst, blocked):
    """Generate a list of candidate paths (shortest + alternatives) from src to dst avoiding blocked."""
    candidates = []
    p_min = _bfs_path(src, dst, blocked)
    if p_min is None:
        return []
    
    candidates.append(p_min)
    seen_paths = {tuple(p_min)}
    
    # Generate alternatives by blocking each interior vertex of the shortest path
    for v in p_min[1:-1]:
        new_blocked = blocked | {v}
        p_alt = _bfs_path(src, dst, new_blocked)
        if p_alt is not None:
            key = tuple(p_alt)
            if key not in seen_paths:
                candidates.append(p_alt)
                seen_paths.add(key)
                
    # Sort by length (shortest first)
    candidates.sort(key=len)
    return candidates

def _route_paths(branches):
    """Find internally-disjoint paths for every unordered pair of branches."""
    pairs = [(i, j) for i in range(len(branches)) for j in range(i + 1, len(branches))]
    
    # Sort pairs by shortest path length to prioritize constrained pairs
    def bfs_dist(a, b):
        if a == b: return 0
        if (1 << b) & _ADJ[a]: return 1
        seen = {a}
        q = deque([(a, 0)])
        while q:
            u, d = q.popleft()
            for nb in _ADJ_LIST[u]:
                if nb not in seen:
                    if nb == b:
                        return d + 1
                    seen.add(nb)
                    q.append((nb, d + 1))
        return 10**9

    pairs.sort(key=lambda p: bfs_dist(branches[p[0]], branches[p[1]]))

    def solve(idx, used_interiors, paths):
        if idx == len(pairs):
            return paths
        
        # Pruning: Check if all remaining pairs are still connected
        # This is expensive but N is small.
        for k in range(idx, len(pairs)):
            i, j = pairs[k]
            a, b = branches[i], branches[j]
            if _bfs_path(a, b, used_interiors) is None:
                return None

        i, j = pairs[idx]
        a, b = branches[i], branches[j]
        
        candidates = _generate_candidates(a, b, used_interiors)
        
        for path in candidates:
            interiors = set(path[1:-1])
            if interiors & used_interiors:
                continue
            new_used = used_interiors | interiors
            new_paths = paths + [path]
            res = solve(idx + 1, new_used, new_paths)
            if res is not None:
                return res
        return None

    # Try with a fixed order first
    res = solve(0, set(), [])
    if res is not None:
        return res
        
    # Randomized restarts if the deterministic order failed
    for attempt in range(50):
        random.shuffle(pairs)
        res = solve(0, set(), [])
        if res is not None:
            return res
            
    return None

def construct():
    from itertools import combinations
    size, mask = _max_clique()
    branches = [i for i in range(_N) if (mask >> i) & 1]
    if len(branches) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    paths = _route_paths(branches)
    if paths is not None:
        out_paths = []
        for p in paths:
            out_paths.append([int(x) for x in p])
        return {'branches': [int(b) for b in branches], 'paths': out_paths}

    # Fallback: reduce to a smaller clique that we can route
    for k in range(len(branches) - 1, 1, -1):
        for combo in combinations(range(len(branches)), k):
            sub = [branches[i] for i in combo]
            ps = _route_paths(sub)
            if ps is not None:
                out_paths = [[int(x) for x in p] for p in ps]
                return {'branches': [int(b) for b in sub], 'paths': out_paths}
        if k <= 3:
            break

    # Ultimate fallback: a single edge
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
