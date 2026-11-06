import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def count_edges_in_mask(mask, adj):
        """Count edges in the induced subgraph defined by mask"""
        count = 0
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m ^= lsb
            # Count neighbors of u within mask, excluding u itself
            count += (adj[u] & mask).bit_count()
        return count // 2

    def count_ind_sets_4_in_mask(mask, adj):
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
        
        # Iterate all 4-subsets
        for i in range(n_v):
            u = vertices[i]
            for j in range(i+1, n_v):
                v = vertices[j]
                if (adj[u] >> v) & 1:
                    continue
                for k in range(j+1, n_v):
                    w = vertices[k]
                    if (adj[u] >> w) & 1 or (adj[v] >> w) & 1:
                        continue
                    for l in range(k+1, n_v):
                        x = vertices[l]
                        if (adj[u] >> x) & 1 or (adj[v] >> x) & 1 or (adj[w] >> x) & 1:
                            continue
                        count += 1
        return count

    def total_violations(adj, n):
        """Calculate total number of K4s and I6s"""
        k4_count = 0
        # Each K4 has 6 edges. For each edge (u,v), count edges in common neighborhood.
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    k4_count += count_edges_in_mask(common, adj)
        k4_count //= 6
        
        i6_count = 0
        # Count I6s directly? O(n^6) is too slow for full calc every time.
        # But we only need it for initialization or final check.
        # For delta calculation, we use count_ind_sets_4_in_mask on non-neighbors.
        
        # For the initial total, we can sum deltas? No, just compute once.
        # O(n^6) for n=35 is ~1.6M * 15 checks = 24M ops. Acceptable once.
        for combo in combinations(range(n), 6):
            is_indep = True
            for i in range(6):
                u = combo[i]
                for j in range(i+1, 6):
                    v = combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                i6_count += 1
                
        return k4_count + i6_count

    def delta_violations(u, v, adj, n, is_edge):
        """
        Returns change in total violations if edge (u,v) is flipped.
        is_edge: True if (u,v) is currently an edge.
        """
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        
        non_common = ((1 << n) - 1) & ~(adj[u] | adj[v]) & ~(1 << u) & ~(1 << v)
        
        e_count = count_edges_in_mask(common, adj)
        i4_count = count_ind_sets_4_in_mask(non_common, adj)
        
        if is_edge:
            # Removing edge:
            # Destroys K4s: -e_count (each edge in common forms a K4 with u,v)
            # Creates I6s: +i4_count (each I4 in non_common forms an I6 with u,v)
            return -e_count + i4_count
        else:
            # Adding edge:
            # Creates K4s: +e_count
            # Destroys I6s: -i4_count
            return e_count - i4_count

    best_graph = None
    best_n = 0
    
    # Try to find valid graphs for n from 35 down to 20
    for target_n in range(35, 19, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random graph
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u+1, target_n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Calculate initial violations
            curr_viol = total_violations(adj, target_n)
            
            # Local search
            improved = True
            no_improve_count = 0
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
                            
                if best_u != -1:
                    # Flip edge
                    if (adj[best_u] >> best_v) & 1:
                        adj[best_u] &= ~(1 << best_v)
                        adj[best_v] &= ~(1 << best_u)
                    else:
                        adj[best_u] |= (1 << best_v)
                        adj[best_v] |= (1 << best_u)
                    
                    curr_viol += best_delta
                    improved = True
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    # If stuck, try a random flip to escape local minimum
                    if no_improve_count > 10:
                        ru = random.randint(0, target_n-1)
                        rv = random.randint(0, target_n-1)
                        if ru != rv:
                            is_edge = bool((adj[ru] >> rv) & 1)
                            d = delta_violations(ru, rv, adj, target_n, is_edge)
                            if (adj[ru] >> rv) & 1:
                                adj[ru] &= ~(1 << rv)
                                adj[rv] &= ~(1 << ru)
                            else:
                                adj[ru] |= (1 << rv)
                                adj[rv] |= (1 << ru)
                            curr_viol += d
                            no_improve_count = 0
                            improved = True

            # Check if valid
            if curr_viol == 0:
                # Double check with rigorous is_valid just in case of bug
                # (Though delta logic is exact)
                if total_violations(adj, target_n) == 0:
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
        
        if found:
            break

    if best_graph is None:
        # Fallback to a known small valid graph if all fails
        # Paley 17 is K4-free? No, Paley 17 has K4s.
        # We need a K4-free and I6-free graph.
        # The smallest such graph is smaller.
        # Let's just return n=1 empty graph? No, n>=1.
        # A valid graph on 5 vertices: K5 is not K4-free? K5 contains K4.
        # We need no K4. So max clique size 3.
        # No I6.
        # Any graph with n<6 has no I6.
        # So we just need a K4-free graph.
        # Complete graph K3 is K4-free. n=3.
        # Let's construct a K4-free graph on n=5.
        # Cycle C5 is K4-free.
        adj = [0] * 5
        for i in range(5):
            adj[i] |= (1 << ((i+1)%5))
            adj[i] |= (1 << ((i-1)%5))
        return {'n': 5, 'edges': get_edges(adj, 5)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
