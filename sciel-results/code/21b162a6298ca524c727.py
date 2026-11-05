import random
import time

_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute for each Q3 edge and orientation: the 6 unassigned vertices in best-first order
# and for each position, which assigned vertices it must be adjacent to.
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (ga_q, gb_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != ga_q and i != gb_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, ga_q, gb_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((ga_q, gb_q, tuple(unassigned)))

def _compute_backtrack_order(adj, n, full_mask):
    """Precompute a good vertex ordering for Q3 search based on degrees."""
    degs = [bin(adj[i]).count('1') for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda x: degs[x], reverse=True)
    return order

def _has_q3_with_edge(adj, n, u, v, full_mask):
    """Fast check: does adding edge (u,v) create a Q3? Uses precomputed orientations."""
    # Temporarily add edge
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    found = False
    for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
        for (gu, gv) in [(u, v), (v, u)]:
            assignment = [-1] * 8
            assignment[qa] = gu
            assignment[qb] = gv
            used = (1 << gu) | (1 << gv)
            
            def bt(idx):
                if idx == 6:
                    for a, b in _Q3_EDGES:
                        if not (adj[assignment[a]] & (1 << assignment[b])):
                            return False
                    return True
                x = unassigned[idx]
                candidates = full_mask & ~used
                for y in range(8):
                    if assignment[y] >= 0 and (_Q3_ADJ[x] & (1 << y)):
                        candidates &= adj[assignment[y]]
                while candidates:
                    bit = candidates & (-candidates)
                    g = bit.bit_length() - 1
                    candidates ^= bit
                    assignment[x] = g
                    if bt(idx + 1):
                        return True
                    assignment[x] = -1
                return False
            
            if bt(0):
                found = True
                break
        if found:
            break
    
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    return found

def _find_q3_witness(adj, n, u, v, full_mask):
    """Find an explicit Q3 embedding using edge (u,v). Returns list of 8 graph vertices or None."""
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
        for (gu, gv) in [(u, v), (v, u)]:
            assignment = [-1] * 8
            assignment[qa] = gu
            assignment[qb] = gv
            used = (1 << gu) | (1 << gv)
            
            def bt(idx):
                if idx == 6:
                    for a, b in _Q3_EDGES:
                        if not (adj[assignment[a]] & (1 << assignment[b])):
                            return False
                    return True
                x = unassigned[idx]
                candidates = full_mask & ~used
                for y in range(8):
                    if assignment[y] >= 0 and (_Q3_ADJ[x] & (1 << y)):
                        candidates &= adj[assignment[y]]
                while candidates:
                    bit = candidates & (-candidates)
                    g = bit.bit_length() - 1
                    candidates ^= bit
                    assignment[x] = g
                    if bt(idx + 1):
                        return True
                    assignment[x] = -1
                return False
            
            if bt(0):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                return assignment
    
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)
    return None

def _find_q3_with_fixed_edge(adj, n, u, v, full_mask):
    """Find Q3 using fixed edge (u,v), return witness or None."""
    return _find_q3_witness(adj, n, u, v, full_mask)

def _has_q3_fast(adj, n, full_mask):
    """Check if graph contains any Q3. Returns True if found."""
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                if _has_q3_with_edge(adj, n, i, j, full_mask):
                    return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 16
    full_mask = (1 << n) - 1

    def make_adj_from_parts(parts):
        adj = [0] * n
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                for u in parts[i]:
                    for v in parts[j]:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    # Start with K_{3,13}
    adj = make_adj_from_parts([[0,1,2], list(range(3,16))])
    best_adj = list(adj)
    best_count = count_edges(adj)

    # Phase 1: Greedy edge addition
    missing = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                missing.append((i, j))
    random.shuffle(missing)

    for u, v in missing:
        if time.time() > deadline - 40:
            break
        if not _has_q3_with_edge(adj, n, u, v, full_mask):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    cnt = count_edges(adj)
    if cnt > best_count:
        best_count = cnt
        best_adj = list(adj)

    # Phase 2: Targeted edge swaps using Q3 witnesses
    # For each missing edge, find the Q3 that blocks it, then try removing one edge from that Q3
    # to unblock the missing edge.
    swap_budget = deadline - time.time() - 15
    if swap_budget > 10:
        for iteration in range(20):
            if time.time() > deadline - 15:
                break
            
            missing = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        missing.append((i, j))
            
            random.shuffle(missing)
            improved = False
            
            for u, v in missing:
                if time.time() > deadline - 15:
                    break
                
                # Find Q3 witness blocking this edge
                witness = _find_q3_witness(adj, n, u, v, full_mask)
                if witness is None:
                    # Shouldn't happen, but add the edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
                    continue
                
                # Try removing each edge in the witness Q3 (except the new edge u-v)
                # to see if we can add u-v
                witness_set = set(witness)
                removed_ok = False
                for a in range(8):
                    for b in range(a + 1, 8):
                        ga, gb = witness[a], witness[b]
                        if (ga, gb) == (u, v) or (ga, gb) == (v, u):
                            continue
                        if not (adj[ga] & (1 << gb)):
                            continue  # This edge isn't in the graph
                        # Try removing (ga,gb) and adding (u,v)
                        adj[ga] &= ~(1 << gb)
                        adj[gb] &= ~(1 << ga)
                        # Check if u-v can now be added without creating new Q3
                        if not _has_q3_with_edge(adj, n, u, v, full_mask):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            removed_ok = True
                            improved = True
                            break
                        # Restore
                        adj[ga] |= (1 << gb)
                        adj[gb] |= (1 << ga)
                    if removed_ok:
                        break
                
                if improved:
                    cnt = count_edges(adj)
                    if cnt > best_count:
                        best_count = cnt
                        best_adj = list(adj)
            
            if not improved:
                # Try a few more random restarts of the greedy phase
                for restart in range(3):
                    if time.time() > deadline - 15:
                        break
                    adj = list(best_adj)
                    missing2 = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if not (adj[i] & (1 << j)):
                                missing2.append((i, j))
                    random.shuffle(missing2)
                    for u, v in missing2:
                        if time.time() > deadline - 15:
                            break
                        if not _has_q3_with_edge(adj, n, u, v, full_mask):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                    cnt = count_edges(adj)
                    if cnt > best_count:
                        best_count = cnt
                        best_adj = list(adj)
                        improved = True
                if not improved:
                    break

    # Build edge list from best
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return edges
