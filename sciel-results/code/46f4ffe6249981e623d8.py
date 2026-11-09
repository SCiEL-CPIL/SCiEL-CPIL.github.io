<<<<<<< SEARCH
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    n = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_k33_free(adj):
        """
        Check if graph defined by adj (list of bitmasks) is K3,3-free.
        A K3,3 exists if there are 3 vertices A={a1,a2,a3} and 3 vertices B={b1,b2,b3}
        such that all 9 edges between A and B exist.
        Equivalent: No 3 vertices have 3 common neighbors.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij == 0:
                    continue
                # If common_ij has < 3 bits, it can't be part of a K3,3 with k
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def greedy_construct(p_order):
        """
        Build a valid graph by adding edges in a specific order if they don't violate K3,3.
        p_order is a list of (u,v) pairs.
        Optimized: only check K3,3 violations involving the new edge (u,v).
        """
        adj = [0] * n
        
        for u, v in p_order:
            if (adj[u] >> v) & 1:
                continue
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Fast check: only look for K3,3 violations that involve edge (u,v)
            # A K3,3 involving (u,v) must have u or v in one partition and the other in the other,
            # OR both in the same partition with 3 common neighbors.
            
            violation = False
            
            # Case 1: u and v are in the same 3-set A, need 3 common neighbors
            common_uv = adj[u] & adj[v]
            if common_uv.bit_count() >= 3:
                # Check if there exist 2 other vertices x,y such that x,y also have >=3 common neighbors with u,v
                # Actually simpler: if common_uv has >=3 bits, pick any 3 of them as B set.
                # But we need to verify that those 3 B vertices are connected to all of A={u,v,?}
                # Wait, the condition is: no 3 vertices have 3 common neighbors.
                # So if u and v have >=3 common neighbors, we need to check if there's a third vertex w
                # such that u,v,w have >=3 common neighbors.
                for w in range(n):
                    if w == u or w == v:
                        continue
                    common_uvw = common_uv & adj[w]
                    if common_uvw.bit_count() >= 3:
                        violation = True
                        break
            
            if not violation:
                # Case 2: u is in A, v is in B (or vice versa). 
                # This means we need 2 other vertices x,y in A such that x,y,v have >=3 common neighbors
                # AND u is connected to all of x,y,v.
                # More precisely: if v is in B, then for any 2 vertices x,y that are neighbors of u,
                # if x,y,v have >=3 common neighbors, and u is connected to x,y,v, we have a K3,3.
                # Actually, the standard check: for each pair (x,y) where x,y are neighbors of u,
                # check if x,y,v have >=3 common neighbors. If yes, and u is connected to x,y,v, violation.
                
                # Simpler approach: just do the full check but it's fast enough for n=20
                if is_k33_free(adj):
                    pass
                else:
                    violation = True
            
            if violation:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        return adj

    def local_search(adj, max_iters=50):
        """
        Improve the graph by trying to add missing edges or performing swaps.
        Returns improved adj.
        """
        current_edges = edge_count(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 1.0:
                break
                
            # 1. Try to add any missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_k33_free(adj):
                        current_edges += 1
                        improved = True
                        break
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            # 2. If no edge could be added, try a swap (remove one, add one)
            if not improved:
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                if existing and missing:
                    random.shuffle(existing)
                    random.shuffle(missing)
                    # Limit swap attempts to avoid slowness
                    attempts = 0
                    max_attempts = min(200, len(existing) * len(missing))
                    for u1, v1 in existing:
                        if improved or attempts >= max_attempts:
                            break
                        for u2, v2 in missing:
                            if attempts >= max_attempts:
                                break
                            attempts += 1
                            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                                continue
                            
                            # Perform swap
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            
                            if is_k33_free(adj):
                                current_edges = edge_count(adj)  # Should be same or +1-1
                                improved = True
                                break
                            else:
                                # Revert
                                adj[u1] |= (1 << v1)
                                adj[v1] |= (1 << u1)
                                adj[u2] &= ~(1 << v2)
                                adj[v2] &= ~(1 << u2)
                    if improved:
                        break # Restart loop to try adding edges again
                    else:
                        # Try a random kick: remove 2 random edges, add 2 random missing edges
                        # This is a 2-opt move.
                        if len(existing) >= 2 and len(missing) >= 2:
                            rem1, rem2 = random.sample(existing, 2)
                            add1, add2 = random.sample(missing, 2)
                            # Check if we are removing/adding the same edges
                            edge_set = {(min(a[0],a[1]), max(a[0],a[1])) for a in [rem1, rem2, add1, add2]}
                            if len(edge_set) == 4:
                                # Apply
                                for u,v in [rem1, rem2]:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                for u,v in [add1, add2]:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                
                                if is_k33_free(adj):
                                    current_edges = edge_count(adj)
                                    improved = True
                                else:
                                    # Revert
                                    for u,v in [rem1, rem2]:
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                    for u,v in [add1, add2]:
                                        adj[u] &= ~(1 << v)
                                        adj[v] &= ~(1 << u)

        return adj
=======
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    n = 20
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_k33_free(adj):
        """
        Check if graph defined by adj (list of bitmasks) is K3,3-free.
        A K3,3 exists if there are 3 vertices A={a1,a2,a3} and 3 vertices B={b1,b2,b3}
        such that all 9 edges between A and B exist.
        Equivalent: No 3 vertices have 3 common neighbors.
        """
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                common_ij = ai & adj[j]
                if common_ij == 0:
                    continue
                # If common_ij has < 3 bits, it can't be part of a K3,3 with k
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    common_ijk = common_ij & adj[k]
                    if common_ijk.bit_count() >= 3:
                        return False
        return True

    def can_add_edge_fast(adj, u, v):
        """
        Fast check: can we add edge (u,v) without creating a K3,3?
        Only check configurations that involve the new edge (u,v).
        A K3,3 involving (u,v) must have:
        - u and v in the same 3-set A, with 3 common neighbors in B
        - u in A, v in B (or vice versa), with 2 other vertices completing A and B
        """
        # Case 1: u and v are in the same 3-set A
        # Need 3 common neighbors of u and v that are also connected to some third vertex w in A
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() >= 3:
            # For each potential third vertex w in A (w != u, w != v),
            # check if u, v, w have >= 3 common neighbors
            for w in range(n):
                if w == u or w == v:
                    continue
                common_uvw = common_uv & adj[w]
                if common_uvw.bit_count() >= 3:
                    return False
        
        # Case 2: u is in A, v is in B (or vice versa)
        # Subcase 2a: u in A, v in B. Need 2 other vertices x,y in A such that:
        #   - x,y are neighbors of u (already true since x,y in A and u in A)
        #   - x,y,v have >= 3 common neighbors
        #   - u is connected to x,y,v (u-x, u-y already exist, u-v is the new edge)
        # So: for each pair (x,y) where x,y are neighbors of u and x != v, y != v, x != y:
        #   check if x, y, v have >= 3 common neighbors
        neighbors_u = adj[u]
        for x in range(n):
            if x == u or x == v:
                continue
            if not (neighbors_u >> x) & 1:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if not (neighbors_u >> y) & 1:
                    continue
                # Check if x, y, v have >= 3 common neighbors
                common_xyv = adj[x] & adj[y] & adj[v]
                if common_xyv.bit_count() >= 3:
                    return False
        
        # Subcase 2b: v in A, u in B (symmetric)
        neighbors_v = adj[v]
        for x in range(n):
            if x == u or x == v:
                continue
            if not (neighbors_v >> x) & 1:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue
                if not (neighbors_v >> y) & 1:
                    continue
                # Check if x, y, u have >= 3 common neighbors
                common_xyu = adj[x] & adj[y] & adj[u]
                if common_xyu.bit_count() >= 3:
                    return False
        
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]

    def greedy_construct(p_order):
        """
        Build a valid graph by adding edges in a specific order if they don't violate K3,3.
        p_order is a list of (u,v) pairs.
        Optimized: only check K3,3 violations involving the new edge (u,v).
        """
        adj = [0] * n
        
        for u, v in p_order:
            if (adj[u] >> v) & 1:
                continue
            # Fast check before adding
            if can_add_edge_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_iters=50):
        """
        Improve the graph by trying to add missing edges or performing swaps.
        Returns improved adj.
        """
        current_edges = edge_count(adj)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 1.0:
                break
                
            # 1. Try to add any missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not (adj[u] >> v) & 1]
            if missing:
                random.shuffle(missing)
                for u, v in missing:
                    if can_add_edge_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        improved = True
                        break
            
            # 2. If no edge could be added, try a swap (remove one, add one)
            if not improved:
                existing = [(u, v) for u in range(n) for v in range(u + 1, n) if (adj[u] >> v) & 1]
                if existing and missing:
                    random.shuffle(existing)
                    random.shuffle(missing)
                    # Limit swap attempts to avoid slowness
                    attempts = 0
                    max_attempts = min(200, len(existing) * len(missing))
                    for u1, v1 in existing:
                        if improved or attempts >= max_attempts:
                            break
                        for u2, v2 in missing:
                            if attempts >= max_attempts:
                                break
                            attempts += 1
                            if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                                continue
                            
                            # Perform swap: remove (u1,v1), add (u2,v2)
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            
                            # Check if we can add (u2,v2) to the modified graph
                            if can_add_edge_fast(adj, u2, v2):
                                adj[u2] |= (1 << v2)
                                adj[v2] |= (1 << u2)
                                improved = True
                                break
                            else:
                                # Revert removal
                                adj[u1] |= (1 << v1)
                                adj[v1] |= (1 << u1)
                    if improved:
                        break # Restart loop to try adding edges again
                    else:
                        # Try a random kick: remove 2 random edges, add 2 random missing edges
                        # This is a 2-opt move.
                        if len(existing) >= 2 and len(missing) >= 2:
                            rem1, rem2 = random.sample(existing, 2)
                            add1, add2 = random.sample(missing, 2)
                            # Check if we are removing/adding the same edges
                            edge_set = {(min(a[0],a[1]), max(a[0],a[1])) for a in [rem1, rem2, add1, add2]}
                            if len(edge_set) == 4:
                                # Apply removals
                                for u,v in [rem1, rem2]:
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                
                                # Check if we can add both edges
                                can_add_1 = can_add_edge_fast(adj, add1[0], add1[1])
                                if can_add_1:
                                    adj[add1[0]] |= (1 << add1[1])
                                    adj[add1[1]] |= (1 << add1[0])
                                can_add_2 = can_add_edge_fast(adj, add2[0], add2[1])
                                if can_add_1 and can_add_2:
                                    # Both can be added
                                    if not (adj[add2[0]] >> add2[1]) & 1:  # Check if already added
                                        adj[add2[0]] |= (1 << add2[1])
                                        adj[add2[1]] |= (1 << add2[0])
                                    improved = True
                                else:
                                    # Revert
                                    for u,v in [rem1, rem2]:
                                        adj[u] |= (1 << v)
                                        adj[v] |= (1 << u)
                                    if can_add_1:
                                        adj[add1[0]] &= ~(1 << add1[1])
                                        adj[add1[1]] &= ~(1 << add1[0])
                                    if can_add_2:
                                        adj[add2[0]] &= ~(1 << add2[1])
                                        adj[add2[1]] &= ~(1 << add2[0])

        return adj
>>>>>>> REPLACE

