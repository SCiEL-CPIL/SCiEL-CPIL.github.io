"""Rewritten program: branch-and-bound + local search for minimal triangle-free diameter-4 augmentation of P24."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

_N = P['n']
_BASE = [tuple(e) for e in P['edges']]
_BASE_SET = set(_BASE)
_BASE_SET |= {(b, a) for (a, b) in _BASE}
_TARGET = P['diameter']

# Precompute all candidate edges (non-base, undirected, u < v)
_CANDIDATES = []
for u in range(_N):
    for v in range(u + 1, _N):
        if (u, v) not in _BASE_SET:
            _CANDIDATES.append((u, v))
_CANDIDATES_SET = set(_CANDIDATES)

def _build_adj(added):
    """Build adjacency list from base + added edges. Returns list of sets."""
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_triangle_free_incre(added, candidate):
    """Check if adding 'candidate' to 'added' keeps triangle-free.
    candidate is (u, v). Returns True if valid."""
    u, v = candidate
    # Build adjacency for base + added
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        adj[a].add(b)
        adj[b].add(a)
    # Check if u and v share a neighbor
    if adj[u] & adj[v]:
        return False
    return True

def _diameter_fast(added):
    """Compute diameter of base+added. Returns large number if disconnected."""
    adj = _build_adj(added)
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du
                    dq.append(v)
        if -1 in dist:
            return 10**9
        m = max(dist)
        if m > best:
            best = m
    return best

def _valid_solution(added):
    """Check triangle-free, connected, diameter <= target."""
    # Triangle-free check
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    for (a, b) in added:
        if adj[a] & adj[b]:
            return False
        adj[a].add(b)
        adj[b].add(a)
    # Diameter check
    best = 0
    for s in range(_N):
        dist = [-1] * _N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u] + 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du
                    dq.append(v)
        if -1 in dist:
            return False
        m = max(dist)
        if m > best:
            best = m
    return best <= _TARGET

def _try_remove(added, seed, deadline):
    """Local search: try to remove edges while maintaining validity."""
    rng = random.Random(seed)
    cur = list(added)
    improved = True
    while improved:
        improved = False
        # Shuffle order to avoid systematic bias
        order = list(range(len(cur)))
        rng.shuffle(order)
        for i in order:
            if time.time() > deadline:
                break
            trial = cur[:i] + cur[i + 1:]
            if _valid_solution(trial):
                cur = trial
                improved = True
                break
    return cur

def _try_swap(added, seed, deadline):
    """Try to swap one edge for another to enable further removals."""
    rng = random.Random(seed)
    cur = list(added)
    for _ in range(20):  # limited swaps
        if time.time() > deadline:
            break
        if len(cur) < 2:
            break
        i = rng.randrange(len(cur))
        old_edge = cur[i]
        # Find a candidate edge to swap in
        swap_in = None
        for cand in _CANDIDATES:
            if cand in cur:
                continue
            if cand == old_edge:
                continue
            trial = [e for j, e in enumerate(cur) if j != i]
            trial.append(cand)
            if _valid_solution(trial):
                swap_in = cand
                break
        if swap_in is not None:
            cur[i] = swap_in
            # Try to remove after swap
            cur = _try_remove(cur, seed + 1000 + _, deadline)
    return cur

def _greedy_set_cover(deadline):
    """Greedy set-cover approach: select edges that cover the most 'long' pairs.
    A pair (u,v) is 'long' if dist(u,v) > target in current graph."""
    added = []
    added_set = set()
    adj = [set() for _ in range(_N)]
    for (a, b) in _BASE:
        adj[a].add(b)
        adj[b].add(a)
    
    def compute_all_dists(adj):
        dists = [[0] * _N for _ in range(_N)]
        for s in range(_N):
            dist = [-1] * _N
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                du = dist[u] + 1
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du
                        dq.append(v)
            dists[s] = dist
        return dists
    
    dists = compute_all_dists(adj)
    
    while True:
        if time.time() > deadline:
            return added
        # Find all pairs with dist > target
        long_pairs = set()
        for u in range(_N):
            for v in range(u + 1, _N):
                if dists[u][v] > _TARGET:
                    long_pairs.add((u, v))
        if not long_pairs:
            break
        
        # For each candidate, count how many long_pairs it would fix
        best_cand = None
        best_count = -1
        for cand in _CANDIDATES:
            if cand in added_set:
                continue
            u, v = cand
            # Check triangle-free
            if adj[u] & adj[v]:
                continue
            # Count how many long pairs would be fixed
            # A long pair (a,b) is fixed if there's a path a-...-u-v-...-b or a-...-v-u-...-b
            # with length <= target
            count = 0
            for (a, b) in long_pairs:
                # Check if adding (u,v) creates a path of length <= target between a and b
                # Path via u,v: dist(a,u) + 1 + dist(v,b) or dist(a,v) + 1 + dist(u,b)
                if dists[a][u] != -1 and dists[v][b] != -1:
                    if dists[a][u] + 1 + dists[v][b] <= _TARGET:
                        count += 1
                if dists[a][v] != -1 and dists[u][b] != -1:
                    if dists[a][v] + 1 + dists[u][b] <= _TARGET:
                        count += 1
            if count > best_count:
                best_count = count
                best_cand = cand
        
        if best_cand is None or best_count == 0:
            # No candidate fixes any long pair; break
            break
        
        # Add best candidate
        added.append(best_cand)
        added_set.add(best_cand)
        adj[best_cand[0]].add(best_cand[1])
        adj[best_cand[1]].add(best_cand[0])
        # Recompute distances
        dists = compute_all_dists(adj)
    
    return added

def _initial_solution(deadline):
    """Get an initial valid solution using greedy set cover + local search."""
    for seed in range(10):
        if time.time() > deadline:
            break
        g = _greedy_set_cover(deadline)
        if g is not None and _valid_solution(g):
            reduced = _try_remove(g, seed, deadline)
            if _valid_solution(reduced):
                return reduced
    # Fallback: use a known structure
    # Try a symmetric pattern
    candidates = [
        ((0, 12), (11, 23), (1, 11), (12, 22), (2, 12), (11, 21), (12, 20), (3, 12)),
        ((0, 11), (12, 23), (1, 12), (11, 22), (2, 11), (12, 21), (3, 12), (11, 20)),
        ((0, 10), (11, 23), (1, 11), (12, 22), (2, 12), (11, 21), (12, 20), (3, 11)),
        ((0, 12), (11, 22), (1, 11), (12, 23), (2, 12), (11, 21), (12, 20), (3, 11)),
    ]
    for cand_set in candidates:
        if time.time() > deadline:
            break
        if _valid_solution(cand_set):
            reduced = _try_remove(cand_set, 42, deadline)
            if _valid_solution(reduced):
                return reduced
    # Last resort: dense solution
    added = []
    for u in range(_N):
        for v in range(u + 1, _N):
            if (u, v) not in _BASE_SET:
                if (u % 2 == 0 and v % 2 == 1) or (u % 2 == 1 and v % 2 == 0):
                    if (u + v) % 3 != 0:  # some filtering
                        added.append((u, v))
    if _valid_solution(added):
        return _try_remove(added, 999, deadline)
    return None

def construct():
    deadline = time.time() + 55.0
    best = None
    best_len = 10**9
    
    # Phase 1: Get an initial solution
    init = _initial_solution(deadline)
    if init is not None:
        best = init
        best_len = len(init)
    
    # Phase 2: Local search with multiple restarts
    rng = random.Random(12345)
    for seed in range(200):
        if time.time() > deadline:
            break
        if best is None:
            continue
        
        # Start from best and do local search
        trial = list(best)
        
        # Try removal
        trial = _try_remove(trial, seed, deadline)
        if _valid_solution(trial) and len(trial) < best_len:
            best = trial
            best_len = len(trial)
            if best_len <= 5:
                break
        
        # Try swaps
        if best_len > 5:
            trial = list(best)
            trial = _try_swap(trial, seed, deadline)
            if _valid_solution(trial) and len(trial) < best_len:
                best = trial
                best_len = len(trial)
                if best_len <= 5:
                    break
    
    # Phase 3: If still not optimal, try random restarts with different initial solutions
    if best_len > 5:
        for seed in range(50):
            if time.time() > deadline:
                break
            g = _greedy_set_cover(deadline)
            if g is not None:
                # Add some random perturbations
                rng = random.Random(seed * 7)
                perturbed = list(g)
                for _ in range(3):
                    if not perturbed:
                        break
                    i = rng.randrange(len(perturbed))
                    if rng.random() < 0.5:
                        perturbed.pop(i)
                    else:
                        # Add a random candidate
                        if _CANDIDATES:
                            cand = rng.choice(_CANDIDATES)
                            if cand not in perturbed:
                                perturbed.append(cand)
                if _valid_solution(perturbed):
                    reduced = _try_remove(perturbed, seed * 7 + 1, deadline)
                    if _valid_solution(reduced) and len(reduced) < best_len:
                        best = reduced
                        best_len = len(reduced)
                        if best_len <= 5:
                            break
    
    if best is None:
        # Fallback: return a valid dense solution
        added = []
        for u in range(_N):
            for v in range(u + 1, _N):
                if (u, v) not in _BASE_SET:
                    if (u % 2 != v % 2):
                        added.append((u, v))
        if not _valid_solution(added):
            # Even more dense
            added = [(u, v) for u in range(_N) for v in range(u + 1, _N) if (u, v) not in _BASE_SET]
        return [[int(u), int(v)] for (u, v) in added]
    
    return [[int(u), int(v)] for (u, v) in best]
# EVOLVE-BLOCK-END
