import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def is_valid(n, adj):
        # Check for K4
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Check if any edge exists in common
                    # Iterate through bits in common
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if adj[x] & common & ~(1 << x):
                            return False
        # Check for Independent 6
        # This is harder to do incrementally in a simple loop, 
        # but for verification we can just check all 6-subsets if n is small enough
        # Or use the same logic: if we found a graph via search, we trust the count.
        # For the final check, we do a rigorous check.
        
        # Rigorous K4 check
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Any edge in common implies K4
                    c = common
                    while c:
                        lsb = c & -c
                        x = lsb.bit_length() - 1
                        c ^= lsb
                        if (adj[x] >> (common.bit_length() - 1)) & 1: # This logic is flawed, just check pair
                            pass
                        # Correct check:
                        if adj[x] & (common & ~(1 << x)):
                            return False
        
        # Rigorous I6 check
        for combo in combinations(range(n), 6):
            is_indep = True
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return False
        return True

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges_in_subset(mask, adj):
        """Count edges in the induced subgraph defined by mask"""
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            # Count neighbors of u within mask
            count += (adj[u] & mask & ~(1 << u)).bit_count() // 2 # Wait, this counts each edge twice?
            # adj[u] & mask gives neighbors in mask. 
            # We sum degrees in the subgraph. Sum of degrees = 2 * E.
        # Actually, simpler:
        total_deg = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            total_deg += (adj[u] & mask).bit_count()
        return total_deg // 2

    def count_indep_4_in_subset(mask, adj):
        """Count independent sets of size 4 in the induced subgraph defined by mask"""
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            vertices.append(lsb.bit_length() - 1)
            m ^= lsb
        
        count = 0
        n_v = len(vertices)
        if n_v < 4:
            return 0
        
        for i in range(n_v):
            for j in range(i+1, n_v):
                if (adj[vertices[i]] >> vertices[j]) & 1:
                    continue
                for k in range(j+1, n_v):
                    if (adj[vertices[i]] >> vertices[k]) & 1 or (adj[vertices[j]] >> vertices[k]) & 1:
                        continue
                    for l in range(k+1, n_v):
                        if (adj[vertices[i]] >> vertices[l]) & 1 or (adj[vertices[j]] >> vertices[l]) & 1 or (adj[vertices[k]] >> vertices[l]) & 1:
                            continue
                        count += 1
        return count

    def delta_violations(u, v, adj, n, is_edge):
        """
        Returns the change in total violations if edge (u,v) is flipped.
        is_edge: True if (u,v) is currently an edge.
        If is_edge, we are removing it. Delta = -E(N(u)capN(v)) + I4(I(u,v))
        If not is_edge, we are adding it. Delta = E(N(u)capN(v)) - I4(I(u,v))
        """
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        # Independent set candidates: vertices not adjacent to u or v, and not u,v
        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
        
        e_count = count_edges_in_subset(common, adj)
        i4_count = count_indep_4_in_subset(non_common, adj)
        
        if is_edge:
            # Removing edge: 
            # Destroys K4s: -e_count
            # Creates I6s: +i4_count (for each I4 in non_common, adding u,v makes I6)
            return -e_count + i4_count
        else:
            # Adding edge:
            # Creates K4s: +e_count
            # Destroys I6s: -i4_count
            return e_count - i4_count

    best_graph = None
    best_n = 0
    
    # Try to find valid graphs for n from 35 down to 18
    # We prioritize higher n.
    for target_n in range(35, 17, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple restarts for each n
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            adj = [0] * target_n
            # Density around 0.5
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                best_delta = 0
                best_u = -1
                best_v = -1
                
                # Evaluate all edges
                for u in range(target_n):
                    for v in range(u+1, target_n):
                        is_edge = bool((adj[u] >> v) & 1)
                        d = delta_violations(u, v, adj, target_n, is_edge)
                        if d < best_delta:
                            best_delta = d
                            best_u = u
                            best_v = v
                            improved = True
                            
                if best_u != -1:
                    # Flip edge
                    if (adj[best_u] >> best_v) & 1:
                        adj[best_u] &= ~(1 << best_v)
                        adj[best_v] &= ~(1 << best_u)
                    else:
                        adj[best_u] |= (1 << best_v)
                        adj[best_v] |= (1 << best_u)
            
            # Check if valid
            # We need to check if violations are 0.
            # Instead of recalculating from scratch, we can just check is_valid
            # But is_valid is O(n^4) for K4 and O(n^6) for I6.
            # For n=35, is_valid might be slow.
            # Let's optimize is_valid or just trust the local search if it converged?
            # No, we must be sure.
            
            # Quick check: Count violations?
            # We can reuse the delta logic to sum up current violations?
            # No, delta is for flips.
            
            # Let's just run is_valid. It's only called a few times.
            if is_valid(target_n, adj):
                best_graph = adj
                best_n = target_n
                found = True
                break
        
        if found:
            break

    if best_graph is None:
        # Fallback to the seed graph (Paley 17)
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
