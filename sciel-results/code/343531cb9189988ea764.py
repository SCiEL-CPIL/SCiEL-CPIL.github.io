import time
import random

# Q3 edges (canonical, vertices 0..7)
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def _bits(mask):
    while mask:
        lsb = mask & (-mask)
        yield lsb.bit_length() - 1
        mask ^= lsb

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin

    full_mask = (1 << n) - 1

    def has_q3_after_add(adj, u, v, deadline):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                def bt(idx, used_mask):
                    if time.time() > deadline - 2:
                        return False
                    if idx == 6:
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]

                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit

                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    best_edges = None
    best_count = -1

    # Strategy 1: Start with K3,13
    adj = [0] * n
    for a in range(3):
        for b in range(3, 16):
            adj[a] |= (1 << b)
            adj[b] |= (1 << a)
    
    cnt = count_edges(adj)
    best_count = cnt
    best_edges = edge_list(adj)

    # Greedy add
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
    
    # Multiple random restarts of the greedy phase
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Reset to K3,13
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        cnt = count_edges(adj)
        
        if attempt == 0:
            # Sorted order
            pass
        else:
            random.shuffle(missing)
            
        for u, v in missing:
            if time.time() > deadline - 10:
                break
            if not (adj[u] & (1 << v)):
                if not has_q3_after_add(adj, u, v, deadline):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    cnt += 1
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

    # Strategy 2: Local Search on the best found graph
    # Load best graph
    adj = [0] * n
    for u, v in best_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Rebuild missing list
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
    
    # Try to add more edges greedily (in case previous greedy missed some due to order)
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 5:
            break
        if not (adj[u] & (1 << v)):
            if not has_q3_after_add(adj, u, v, deadline):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                best_count += 1
                
    best_edges = edge_list(adj)
    
    # Perturbation: Remove random edges and re-greedy
    if time.time() < deadline - 20:
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Start from current best
            adj = [0] * n
            for u, v in best_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Remove 1-3 random edges
            current_edges = [(u, v) for u, v in best_edges]
            random.shuffle(current_edges)
            for u, v in current_edges[:2]:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                best_count -= 1 # Temporary reduction
            
            # Re-greedy
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                if not (adj[u] & (1 << v)):
                    if not has_q3_after_add(adj, u, v, deadline):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        best_count += 1
            
            if best_count > 0: # Just check validity, count is tracked
                 # We need to recalculate count properly if we messed up the tracker
                 pass
            
            # Recalculate count for comparison
            real_count = count_edges(adj)
            if real_count > best_count:
                best_count = real_count
                best_edges = edge_list(adj)

    return best_edges
