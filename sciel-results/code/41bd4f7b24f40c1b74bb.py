# EVOLVE-BLOCK-START
import time
import random
import sys

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 10  # seconds to spare before the 150s kill

# Precompute Q3 structure
_Q3_EDGES = []
for i in range(8):
    for j in range(i + 1, 8):
        if (i ^ j) in (1, 2, 4):
            _Q3_EDGES.append((i, j))

_Q3_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for bit in (1, 2, 4):
        j = i ^ bit
        if j > i:
            _Q3_NEIGHBORS[i].append(j)
            _Q3_NEIGHBORS[j].append(i)

_Q3_QV_NEIGHBOR_MASK = [0] * 8
for i in range(8):
    mask = 0
    for j in _Q3_NEIGHBORS[i]:
        mask |= (1 << j)
    _Q3_QV_NEIGHBOR_MASK[i] = mask

_Q3_PRIOR_NEIGHBORS = [[] for _ in range(8)]
for i in range(8):
    for j in _Q3_NEIGHBORS[i]:
        if j < i:
            _Q3_PRIOR_NEIGHBORS[i].append(j)

def _compute_backtrack_order(a, b):
    """Compute optimal backtracking order for the 6 remaining Q3 vertices given fixed edge (a,b)."""
    remaining = [v for v in range(8) if v != a and v != b]
    
    order = []
    placed = {a, b}
    for _ in range(6):
        best_v = -1
        best_count = -1
        for v in remaining:
            if v in placed:
                continue
            count = sum(1 for n in _Q3_NEIGHBORS[v] if n in placed)
            if count > best_count:
                best_count = count
                best_v = v
        order.append(best_v)
        placed.add(best_v)
    
    return order

_Q3_EDGE_INFO = {}
for a, b in _Q3_EDGES:
    order = _compute_backtrack_order(a, b)
    neighbor_info = []
    for i, v in enumerate(order):
        prior = set([a, b]) | set(order[:i])
        nbrs = [n for n in _Q3_NEIGHBORS[v] if n in prior]
        neighbor_info.append(nbrs)
    _Q3_EDGE_INFO[(a, b)] = (order, neighbor_info)


def _has_q3_with_edge(adj_bits, u, v):
    """
    Check if adding edge (u,v) to the current graph creates a Q3.
    Only checks Q3 embeddings that include edge (u,v).
    """
    full_mask = (1 << _N) - 1
    
    for (a, b) in _Q3_EDGES:
        order, neighbor_info = _Q3_EDGE_INFO[(a, b)]
        
        if _check_q3_with_fixed_edge(adj_bits, u, v, a, b, order, neighbor_info, full_mask):
            return True
        if _check_q3_with_fixed_edge(adj_bits, v, u, a, b, order, neighbor_info, full_mask):
            return True
    
    return False

def _check_q3_with_fixed_edge(adj_bits, gu, gv, a, b, order, neighbor_info, full_mask):
    """
    Check if there's a Q3 embedding where graph vertex gu maps to Q3 vertex a,
    gv maps to Q3 vertex b.
    """
    used_mask = (1 << gu) | (1 << gv)
    assigned = {}
    assigned[a] = gu
    assigned[b] = gv
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == 6:
            return True
        
        qv = order[idx]
        candidates = (~used_mask) & full_mask
        
        for nqv in neighbor_info[idx]:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv2 = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv2
            used_mask |= (1 << gv2)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv2)
            del assigned[qv]
        
        return False
    
    return backtrack(0)

def _find_q3_witness(adj_bits, u, v):
    """
    Find a specific Q3 embedding that includes edge (u,v), if one exists.
    Returns a list of 8 (qv, gv) pairs or None.
    """
    full_mask = (1 << _N) - 1
    
    for (a, b) in _Q3_EDGES:
        order, neighbor_info = _Q3_EDGE_INFO[(a, b)]
        
        for mapping in [(u, v, a, b), (v, u, a, b)]:
            gu, gv, qa, qb = mapping
            result = _find_q3_with_fixed_edge(adj_bits, gu, gv, qa, qb, order, neighbor_info, full_mask)
            if result is not None:
                return result
    
    return None

def _find_q3_with_fixed_edge(adj_bits, gu, gv, a, b, order, neighbor_info, full_mask):
    """Find a specific Q3 embedding or return None."""
    used_mask = (1 << gu) | (1 << gv)
    assigned = {}
    assigned[a] = gu
    assigned[b] = gv
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == 6:
            return dict(assigned)
        
        qv = order[idx]
        candidates = (~used_mask) & full_mask
        
        for nqv in neighbor_info[idx]:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return None
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv2 = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv2
            used_mask |= (1 << gv2)
            result = backtrack(idx + 1)
            if result is not None:
                return result
            used_mask &= ~(1 << gv2)
            del assigned[qv]
        
        return None
    
    return backtrack(0)


def _has_q3_fast(adj_bits):
    """
    Full Q3 subgraph detection using bitmask operations.
    """
    assigned = [-1] * 8
    used_mask = 0
    full_mask = (1 << _N) - 1
    
    def backtrack(qv):
        nonlocal used_mask
        if qv == 8:
            return True
        
        candidates = (~used_mask) & full_mask
        
        for pv in _Q3_PRIOR_NEIGHBORS[qv]:
            pgv = assigned[pv]
            candidates &= adj_bits[pgv]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(qv + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return False
    
    for gv0 in range(_N):
        assigned[0] = gv0
        used_mask = 1 << gv0
        if backtrack(1):
            return True
        used_mask = 0
        assigned[0] = -1
    
    return False


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    best_edges = []
    best_count = 0
    
    # Phase 1: Greedy to get a good initial lower bound
    num_trials = 20
    for trial in range(num_trials):
        if time.time() > deadline - 60: # Leave time for B&B
            break
        
        if trial == 0:
            edge_order = all_edges[:]
        else:
            random.seed(trial * 12345 + 67890)
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        
        adj = [0] * _N
        edges = []
        
        for u, v in edge_order:
            if time.time() > deadline - 55:
                break
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge(adj, u, v):
                edges.append([u, v])
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
    # Phase 2: Branch and Bound to improve
    global best_count, best_edges
    bnb_start = time.time()
    bnb_deadline = deadline - 5
    
    init_adj = [0] * _N
    init_kept_mask = 0
    for u, v in best_edges:
        init_adj[u] |= (1 << v)
        init_adj[v] |= (1 << u)
        idx = edge_to_idx[(u, v)]
        init_kept_mask |= (1 << idx)
        
    init_undecided_mask = ((1 << _E) - 1) & ~init_kept_mask
    
    sys.setrecursionlimit(10000)
    
    def branch_and_bound(kept_mask, undecided_mask, adj):
        nonlocal best_count, best_edges
        
        if time.time() > bnb_deadline:
            return
            
        num_kept = kept_mask.bit_count()
        num_undecided = undecided_mask.bit_count()
        
        if num_kept + num_undecided <= best_count:
            return
            
        if undecided_mask == 0:
            if num_kept > best_count:
                best_count = num_kept
                new_edges = []
                m = kept_mask
                while m:
                    lsb = m & (-m)
                    idx = lsb.bit_length() - 1
                    u, v = all_edges[idx]
                    new_edges.append([u, v])
                    m ^= lsb
                best_edges = new_edges
            return
            
        lsb = undecided_mask & (-undecided_mask)
        idx = lsb.bit_length() - 1
        u, v = all_edges[idx]
        
        # Branch 1: Include (u, v)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        witness = _find_q3_witness(adj, u, v)
        
        if witness is None:
            new_kept = kept_mask | lsb
            new_undecided = undecided_mask ^ lsb
            branch_and_bound(new_kept, new_undecided, adj)
        else:
            witness_edges = []
            for (qa, qb) in _Q3_EDGES:
                if qa in witness and qb in witness:
                    ga, gb = witness[qa], witness[qb]
                    if ga < gb:
                        edge_tuple = (ga, gb)
                    else:
                        edge_tuple = (gb, ga)
                    witness_edges.append(edge_tuple)
            
            for rem_u, rem_v in witness_edges:
                if (rem_u, rem_v) == (u, v):
                    continue
                
                rem_idx = edge_to_idx[(rem_u, rem_v)]
                rem_bit = 1 << rem_idx
                
                new_kept = (kept_mask | lsb) & ~rem_bit
                new_undecided = (undecided_mask ^ lsb) & ~rem_bit
                
                if kept_mask & rem_bit:
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    
                    branch_and_bound(new_kept, new_undecided, adj)
                    
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                else:
                    branch_and_bound(new_kept, new_undecided, adj)
                    
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Branch 2: Exclude (u, v)
        new_kept = kept_mask
        new_undecided = undecided_mask ^ lsb
        branch_and_bound(new_kept, new_undecided, adj)
        
    branch_and_bound(init_kept_mask, init_undecided_mask, init_adj)
    
    if time.time() < deadline - 1:
        adj = [0] * _N
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        if _has_q3_fast(adj):
            pass
            
    return best_edges
# EVOLVE-BLOCK-END
