"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_N = 16
_E = _N * (_N - 1) // 2  # 120 edges
_DEADLINE_MARGIN = 25  # seconds to spare before the 150s kill

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

# Precompute for fast Q3 detection with fixed edge
# When we fix edge (u,v) as Q3 vertices 0-1, we need to assign Q3 vertices 2-7
# Q3 vertex labels: 0=000, 1=001, 2=010, 3=011, 4=100, 5=101, 6=110, 7=111
# 
# For the fixed-edge check, we precompute the constraint structure
# Ordering: 2, 4, 3, 5, 6, 7
# 
# For each qv in order, store:
#   - which previously-assigned qvs it must be adjacent to
#   - which external vertex (u or v) it must be adjacent to
#   - which later qvs it must be adjacent to (for final check)

_FIXED_EDGE_ORDER = [2, 4, 3, 5, 6, 7]

# Constraints for each qv in the fixed-edge backtracking
# qv -> (list of prev_qvs it must connect to, ext: u/v/None, list of later_qvs it connects to)
_FIXED_CONSTRAINTS = {
    2: ([], 'u', [6]),
    4: ([], 'u', [6]),
    3: ([2], 'v', [7]),
    5: ([4], 'v', [7]),
    6: ([2, 4], None, [7]),
    7: ([3, 5], None, []),
}

# Precompute for each pair of Q3 vertices that must be adjacent,
# a bitmask indicating which positions in _FIXED_EDGE_ORDER they occupy
# This helps with the final edge checks

def _has_q3_with_edge_fast(adj_bits, u, v):
    """
    Very fast Q3 detection checking only Q3 embeddings containing edge (u,v).
    Uses precomputed constraints and bitmask operations.
    """
    order = _FIXED_EDGE_ORDER
    assigned = [-1] * 8
    assigned[0] = u
    assigned[1] = v
    used_mask = (1 << u) | (1 << v)
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == len(order):
            # Check all remaining edges among assigned vertices
            # Edges to check: (6,7) and any others not yet verified
            # Actually, we need to verify all Q3 edges among {2,3,4,5,6,7}
            # that weren't checked during backtracking
            # During backtracking, when assigning qv, we check edges to prev_qvs
            # So we only need to check edges between qvs where both are "later"
            # Let's just check all edges among assigned[2..7]
            for i in range(2, 8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):  # Q3 edge
                        if not (adj_bits[assigned[i]] >> assigned[j]) & 1:
                            return False
            return True
        
        qv = order[idx]
        prev_qvs, ext, later_qvs = _FIXED_CONSTRAINTS[qv]
        
        # Candidates: unused graph vertices
        candidates = (~used_mask) & ((1 << _N) - 1)
        
        # Must connect to external vertex if specified
        if ext == 'u':
            candidates &= adj_bits[u]
        elif ext == 'v':
            candidates &= adj_bits[v]
        
        # Must connect to already-assigned Q3 neighbors
        for nqv in prev_qvs:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return False
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            if backtrack(idx + 1):
                return True
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return False
    
    return backtrack(0)


def _find_q3_witness(adj_bits, u, v):
    """
    Find a specific Q3 embedding containing edge (u,v), if one exists.
    Returns the assignment dict or None.
    """
    order = _FIXED_EDGE_ORDER
    assigned = [-1] * 8
    assigned[0] = u
    assigned[1] = v
    used_mask = (1 << u) | (1 << v)
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == len(order):
            # Check all Q3 edges among {2,3,4,5,6,7}
            for i in range(2, 8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):
                        if not (adj_bits[assigned[i]] >> assigned[j]) & 1:
                            return None
            return assigned[:]
        
        qv = order[idx]
        prev_qvs, ext, later_qvs = _FIXED_CONSTRAINTS[qv]
        
        candidates = (~used_mask) & ((1 << _N) - 1)
        
        if ext == 'u':
            candidates &= adj_bits[u]
        elif ext == 'v':
            candidates &= adj_bits[v]
        
        for nqv in prev_qvs:
            candidates &= adj_bits[assigned[nqv]]
        
        if candidates == 0:
            return None
        
        c = candidates
        while c:
            lsb = c & (-c)
            gv = lsb.bit_length() - 1
            c ^= lsb
            
            assigned[qv] = gv
            used_mask |= (1 << gv)
            result = backtrack(idx + 1)
            if result is not None:
                return result
            used_mask &= ~(1 << gv)
            assigned[qv] = -1
        
        return None
    
    return backtrack(0)


def _greedy_construction(adj, edge_order, deadline):
    """
    Greedy edge addition with Q3 check.
    """
    edges = []
    adj = adj[:]
    
    for u, v in edge_order:
        if time.time() > deadline:
            break
        
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        if not _has_q3_with_edge_fast(adj, u, v):
            edges.append([u, v])
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    return edges, adj


def _local_search(adj, edges, all_edges, deadline):
    """
    Local search: try to add missing edges, and if blocked, try edge swaps.
    """
    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
    missing = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding missing edges
        random.shuffle(missing)
        for u, v in missing[:]:
            if time.time() > deadline:
                break
            if (u, v) in edge_set:
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if not _has_q3_with_edge_fast(adj, u, v):
                edges.append([u, v])
                edge_set.add((u, v))
                if (u, v) in missing:
                    missing.remove((u, v))
                improved = True
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # Try edge swaps: remove one edge, add two
        if time.time() > deadline - 5:
            break
            
        # For each missing edge that's blocked, find a witness Q3
        # and try removing an edge from that Q3 to unblock
        blocked_missing = [(u, v) for u, v in missing if (u, v) not in edge_set]
        random.shuffle(blocked_missing)
        
        for u, v in blocked_missing[:10]:  # Limit work per iteration
            if time.time() > deadline - 3:
                break
            if (u, v) in edge_set:
                continue
            
            # Find a Q3 witness containing (u,v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            witness = _find_q3_witness(adj, u, v)
            
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            if witness is None:
                continue
            
            # Get the edges of this Q3 in the graph
            q3_edges = []
            for i in range(8):
                for j in range(i + 1, 8):
                    if (i ^ j) in (1, 2, 4):
                        gi, gj = witness[i], witness[j]
                        if gi < gj:
                            q3_edges.append((gi, gj))
                        else:
                            q3_edges.append((gj, gi))
            
            # Try removing each edge from the Q3 (except u-v itself)
            for rem_u, rem_v in q3_edges:
                if time.time() > deadline - 2:
                    break
                if rem_u == u and rem_v == v:
                    continue
                if rem_u == v and rem_v == u:
                    continue
                
                # Check if this edge exists
                if not (adj[rem_u] >> rem_v) & 1:
                    continue
                
                # Try removing this edge and adding (u,v)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                # Check if the new graph is Q3-free
                if not _has_q3_with_edge_fast(adj, u, v):
                    # Update edges list
                    edges = [[e[0], e[1]] for e in edges if not (
                        (e[0] == rem_u and e[1] == rem_v) or 
                        (e[0] == rem_v and e[1] == rem_u)
                    )]
                    edges.append([u, v])
                    edge_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in edges)
                    missing = [(uu, vv) for uu, vv in all_edges if (uu, vv) not in edge_set]
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
    
    return edges, adj


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    all_edges = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    
    best_edges = []
    best_count = 0
    best_adj = [0] * _N
    
    # Phase 1: Multiple greedy constructions with different orderings
    num_trials = 80
    
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        # Create diverse orderings
        if trial == 0:
            edge_order = all_edges[:]
        elif trial == 1:
            edge_order = all_edges[::-1]
        elif trial == 2:
            # Prefer edges within "halves" first
            edge_order = [(u, v) for u, v in all_edges if u < 8 and v < 8] + \
                         [(u, v) for u, v in all_edges if u >= 8 and v >= 8] + \
                         [(u, v) for u, v in all_edges if (u < 8) != (v < 8)]
        elif trial == 3:
            # Prefer edges across "halves" first
            edge_order = [(u, v) for u, v in all_edges if (u < 8) != (v < 8)] + \
                         [(u, v) for u, v in all_edges if u < 8 and v < 8] + \
                         [(u, v) for u, v in all_edges if u >= 8 and v >= 8]
        else:
            random.seed(trial * 7919 + 104729)
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        
        adj = [0] * _N
        edges, adj = _greedy_construction(adj, edge_order, deadline - 10)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj[:]
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5:
        adj = best_adj[:]
        edges = best_edges[:]
        edges, adj = _local_search(adj, edges, all_edges, deadline - 2)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
    
    # Phase 3: More local search iterations if time permits
    if time.time() < deadline - 3:
        adj = best_adj[:]
        edges = best_edges[:]
        edges, adj = _local_search(adj, edges, all_edges, deadline - 1)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
    
    return best_edges
# EVOLVE-BLOCK-END
