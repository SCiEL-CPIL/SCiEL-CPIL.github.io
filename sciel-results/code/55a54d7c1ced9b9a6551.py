import time
import random
import numpy as np

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

    def try_add_edge(adj, u, v, deadline):
        """Try to add edge (u,v), return True if added."""
        if adj[u] & (1 << v):
            return False
        if not has_q3_after_add(adj, u, v, deadline):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        return False

    def greedy_fill(adj, deadline):
        """Greedily add edges in random order."""
        cnt = count_edges(adj)
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if try_add_edge(adj, u, v, deadline):
                cnt += 1
        return cnt

    def local_search(adj, deadline):
        """Local search with edge swaps and additions."""
        cnt = count_edges(adj)
        # Build edge set for quick lookup
        edge_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    edge_set.add((u, v))
        
        # Phase 1: Try to add missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 5:
                break
            if try_add_edge(adj, u, v, deadline):
                edge_set.add((u, v))
                cnt += 1
        
        # Phase 2: Edge swaps - remove one edge, add another
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(all_edges)
            for rem_u, rem_v in all_edges:
                if time.time() > deadline - 5:
                    break
                if (rem_u, rem_v) not in edge_set:
                    continue
                
                # Remove the edge
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                edge_set.remove((rem_u, rem_v))
                
                # Try to add a different edge
                added = False
                for add_u, add_v in all_edges:
                    if (add_u, add_v) == (rem_u, rem_v):
                        continue
                    if (add_u, add_v) in edge_set:
                        continue
                    if try_add_edge(adj, add_u, add_v, deadline):
                        edge_set.add((add_u, add_v))
                        cnt += 1
                        added = True
                        improved = True
                        break
                
                if not added:
                    # Put the edge back
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                    edge_set.add((rem_u, rem_v))
        
        return cnt

    best_edges = None
    best_count = -1

    # Strategy 1: Multiple random restarts with greedy fill
    for attempt in range(50):
        if time.time() > deadline - 30:
            break
        
        # Random initial graph
        adj = [0] * n
        # Start with a random sparse graph
        num_initial = random.randint(20, 50)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs[:num_initial]:
            if not has_q3_after_add(adj, u, v, deadline):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Greedy fill
        cnt = greedy_fill(adj, deadline)
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)
        
        # Local search
        cnt = local_search(adj, deadline)
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

    # Strategy 2: Start from K3,13 and improve
    if time.time() < deadline - 20:
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        
        cnt = greedy_fill(adj, deadline)
        cnt = local_search(adj, deadline)
        
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

    # Strategy 3: Start from a random dense graph and remove edges to make Q3-free
    if time.time() < deadline - 15:
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            # Start with random dense graph
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.7:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Remove edges until Q3-free
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            
            changed = True
            while changed and time.time() < deadline - 5:
                changed = False
                for u, v in all_edges:
                    if time.time() > deadline - 5:
                        break
                    if not (adj[u] & (1 << v)):
                        continue
                    # Check if removing this edge is needed
                    # Try removing it and see if it helps
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    
                    # Check if graph is Q3-free
                    is_free = True
                    for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS[:4]:
                        for (gu, gv) in [(0, 1), (1, 0)]:
                            if not (adj[gu] & (1 << gv)):
                                continue
                            assignment = [None] * 8
                            assignment[qa] = gu
                            assignment[qb] = gv
                            used_mask = (1 << gu) | (1 << gv)
                            
                            found = False
                            def bt_check(idx, um):
                                nonlocal found
                                if found:
                                    return
                                if idx == 6:
                                    for a, b in _Q3_EDGES:
                                        ga = assignment[a]
                                        gb = assignment[b]
                                        if not (adj[ga] & (1 << gb)):
                                            return
                                    found = True
                                    return
                                
                                x = unassigned[idx]
                                candidates = full_mask & ~um
                                for y in range(8):
                                    if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                                        hy = assignment[y]
                                        candidates &= adj[hy]
                                
                                while candidates:
                                    bit = candidates & (-candidates)
                                    g = bit.bit_length() - 1
                                    candidates ^= bit
                                    assignment[x] = g
                                    bt_check(idx + 1, um | (1 << g))
                                    assignment[x] = None
                                    if found:
                                        return
                            
                            bt_check(0, used_mask)
                            if found:
                                is_free = False
                                break
                        if not is_free:
                            break
                    
                    if not is_free:
                        # Put edge back
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    # Try to add edges back
                    cnt = greedy_fill(adj, deadline)
                    
                    if cnt > best_count:
                        best_count = cnt
                        best_edges = edge_list(adj)
                    break

    return best_edges
