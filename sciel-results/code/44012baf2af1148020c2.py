"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2

def _edge_idx(u, v):
    if u > v:
        u, v = v, u
    return 15 * u - u * (u - 1) // 2 + (v - u - 1)

def _idx_to_edge(idx):
    u = 0
    while True:
        start = 15 * u - u * (u - 1) // 2
        count = 15 - u
        if idx < start + count:
            v = u + 1 + (idx - start)
            return u, v
        u += 1

def _build_adj_bitmask(edge_list):
    adj = [0] * _N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

_Q3_NEIGHBORS = [[] for _ in range(8)]
for i, j in _Q3_EDGES:
    _Q3_NEIGHBORS[i].append(j)
    _Q3_NEIGHBORS[j].append(i)

def _has_q3(adj):
    """Fast Q3 detection with backtracking."""
    assigned = [-1] * 8
    used_mask = 0
    order = [0, 1, 2, 4, 3, 5, 6, 7]
    
    def backtrack(pos):
        nonlocal used_mask
        if pos == 8:
            return True
        qv = order[pos]
        # Get mask of already-assigned neighbors
        neighbor_mask = 0
        for np in range(pos):
            if order[np] in _Q3_NEIGHBORS[qv]:
                neighbor_mask |= (1 << assigned[order[np]])
        
        # Candidate graph vertices: not used, and adjacent to all assigned neighbors
        candidates = ~used_mask & ((1 << _N) - 1)
        for gv in range(_N):
            if used_mask & (1 << gv):
                continue
            # Check adjacency to all assigned Q3-neighbors of qv
            ok = True
            for nqv in _Q3_NEIGHBORS[qv]:
                if assigned[nqv] >= 0:
                    if not (adj[gv] & (1 << assigned[nqv])):
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used_mask |= (1 << gv)
                if backtrack(pos + 1):
                    return True
                used_mask &= ~(1 << gv)
                assigned[qv] = -1
        return False
    
    return backtrack(0)

def _find_q3_witness(adj):
    """Find Q3 witness or return None."""
    assigned = [-1] * 8
    used_mask = 0
    order = [0, 1, 2, 4, 3, 5, 6, 7]
    
    def backtrack(pos):
        nonlocal used_mask
        if pos == 8:
            return True
        qv = order[pos]
        for gv in range(_N):
            if used_mask & (1 << gv):
                continue
            ok = True
            for nqv in _Q3_NEIGHBORS[qv]:
                if assigned[nqv] >= 0:
                    if not (adj[gv] & (1 << assigned[nqv])):
                        ok = False
                        break
            if ok:
                assigned[qv] = gv
                used_mask |= (1 << gv)
                if backtrack(pos + 1):
                    return True
                used_mask &= ~(1 << gv)
                assigned[qv] = -1
        return False
    
    if backtrack(0):
        edges = [(assigned[i], assigned[j]) for i, j in _Q3_EDGES]
        return edges
    return None

def _initial_k3_13_k13():
    """K3,13 + K13: all edges within {3..15} and all cross edges."""
    edges = []
    # Cross edges
    for a in range(3):
        for b in range(3, 16):
            edges.append((a, b))
    # Within B
    for i in range(3, 16):
        for j in range(i + 1, 16):
            edges.append((i, j))
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try K3,13 + K13 first
    edges = _initial_k3_13_k13()
    adj = _build_adj_bitmask(edges)
    
    if not _has_q3(adj):
        # It's Q3-free! Try to add edges within A = {0,1,2}
        for i in range(3):
            for j in range(i + 1, 3):
                if time.time() > deadline - 10:
                    break
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if not _has_q3(adj):
                    edges.append((i, j))
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
        
        # Try to add more edges if possible
        # All edges are already present in K16, so we're at max 120
        # Check if we can get all 120
        if len(edges) < _E:
            missing = [(u, v) for u in range(_N) for v in range(u + 1, _N)
                      if (u, v) not in set(edges)]
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _has_q3(adj):
                    edges.append((u, v))
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        
        return edges
    
    # K3,13 + K13 is NOT Q3-free, fall back to greedy
    # Start with K3,13 (39 edges, definitely Q3-free since one part has only 3 vertices)
    edges = [(a, b) for a in range(3) for b in range(3, 16)]
    adj = _build_adj_bitmask(edges)
    
    # Greedily add edges within B
    b_edges = [(i, j) for i in range(3, 16) for j in range(i + 1, 16)]
    random.seed(42)
    random.shuffle(b_edges)
    
    for u, v in b_edges:
        if time.time() > deadline - 10:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _has_q3(adj):
            edges.append((u, v))
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    # Try to add edges within A
    for i in range(3):
        for j in range(i + 1, 3):
            if time.time() > deadline - 5:
                break
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            if not _has_q3(adj):
                edges.append((i, j))
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
    
    # Local search with witness swaps
    edge_set = set(edges)
    missing = [(u, v) for u in range(_N) for v in range(u + 1, _N)
               if (u, v) not in edge_set]
    
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        progress = False
        for mu, mv in missing:
            if time.time() > deadline - 3:
                break
            if (mu, mv) in edge_set:
                continue
            adj[mu] |= (1 << mv)
            adj[mv] |= (1 << mu)
            witness = _find_q3_witness(adj)
            if witness is None:
                edge_set.add((mu, mv))
                progress = True
            else:
                for ru, rv in witness:
                    if (ru, rv) == (mu, mv):
                        continue
                    key = (min(ru, rv), max(ru, rv))
                    if key not in edge_set:
                        continue
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    if not _has_q3(adj):
                        edge_set.discard(key)
                        edge_set.add((mu, mv))
                        progress = True
                        break
                    else:
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                else:
                    adj[mu] &= ~(1 << mv)
                    adj[mv] &= ~(1 << mu)
        
        if not progress:
            break
    
    return list(edge_set)
# EVOLVE-BLOCK-END
