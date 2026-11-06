import time
import random
import numpy as np

def construct():
    """
    Construct a C4-free graph with alpha < 11, maximizing n.
    Strategy:
    1. Deterministically construct the polarity graph of PG(2, 5). This gives n=31.
    2. Verify it is C4-free and has alpha < 11.
    3. Try to extend this graph to n=32, ..., 39 by adding vertices and edges
       while maintaining C4-freeness and alpha < 11.
    4. Use a local search to maximize the number of vertices.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def time_left():
        return deadline - time.time()

    def popcount(x):
        return bin(x).count('1')

    def max_independent_set_size(n, adj):
        """
        Find the size of the maximum independent set using branch and bound.
        adj is a list of integers (bitmasks).
        Returns the size.
        """
        best = [0]
        
        def branch(candidates_mask, current_size):
            if time_left() < 5:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Upper bound
            if current_size + popcount(candidates_mask) <= best[0]:
                return
            
            # Heuristic: Pick vertex with highest degree in the candidate subgraph
            best_v = -1
            best_deg = -1
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                deg = popcount(candidates_mask & adj[v])
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                m -= lsb
            
            if best_v == -1:
                return

            v_bit = 1 << best_v
            
            # Option 1: Include best_v
            new_candidates = candidates_mask & ~adj[best_v] & ~v_bit
            branch(new_candidates, current_size + 1)
            
            # Option 2: Exclude best_v
            branch(candidates_mask & ~v_bit, current_size)

        branch((1 << n) - 1, 0)
        return best[0]

    def is_c4_free(n, adj):
        """Check if graph is C4-free."""
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    common = adj[i] & adj[j]
                    if popcount(common) > 1:
                        return False
        return True

    def construct_polarity_graph():
        """
        Construct the polarity graph of PG(2, 5).
        This gives a C4-free graph on 31 vertices.
        """
        # Points in PG(2, 5)
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    # Normalize
                    if x != 0:
                        inv_x = pow(x, -1, 5)
                        points.append((1, y * inv_x % 5, z * inv_x % 5))
                    elif y != 0:
                        inv_y = pow(y, -1, 5)
                        points.append((0, 1, z * inv_y % 5))
                    else:
                        points.append((0, 0, 1))
        points = list(set(points))
        n = len(points)
        
        # Bilinear form for polarity
        def bilinear(u, v):
            return (u[0]*v[2] + u[1]*v[1] + u[2]*v[0]) % 5
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if bilinear(points[i], points[j]) == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        return n, adj

    def try_extend(n, adj, target_n):
        """
        Try to extend the graph to target_n vertices by adding new vertices.
        Returns (new_n, new_adj) if successful, else (n, adj).
        """
        new_adj = adj[:]
        new_n = n
        
        for new_v in range(n, target_n):
            if time_left() < 10:
                break
            
            # Try to find a valid position for new_v
            # new_v should be connected to some existing vertices
            # such that C4-freeness is maintained and alpha doesn't increase
            
            # Strategy: Try all subsets of existing vertices to connect to
            # Use a heuristic: start with empty set, then add edges greedily
            
            new_adj.append(0)
            
            # Try to add edges from new_v to existing vertices
            # in an order that maintains C4-freeness
            added = False
            for i in range(new_n):
                if new_adj[new_v] & (1 << i):
                    continue
                # Check if adding edge (new_v, i) creates a C4
                # Condition: N(new_v) & N(i) must have size <= 1
                # But also need to check interactions with other neighbors
                
                # Simple check: count common neighbors of new_v and i
                common = new_adj[new_v] & new_adj[i]
                if popcount(common) > 1:
                    continue
                
                # Additional check: for each neighbor k of i, 
                # if k is also a neighbor of new_v, that's fine
                # but we need to ensure no C4 is formed
                # The main constraint is |N(a) ∩ N(b)| <= 1 for all pairs a,b
                
                # Let's tentatively add and verify
                new_adj[new_v] |= (1 << i)
                new_adj[i] |= (1 << new_v)
                
                # Verify C4-freeness for the new edge
                valid = True
                # Check pair (new_v, i)
                if popcount(new_adj[new_v] & new_adj[i]) > 1:
                    valid = False
                else:
                    # Check pairs (new_v, k) for k in N(i)
                    for k in range(new_n):
                        if k == i:
                            continue
                        if new_adj[i] & (1 << k):
                            # k is neighbor of i
                            # Check |N(new_v) ∩ N(k)|
                            common_nk = new_adj[new_v] & new_adj[k]
                            if popcount(common_nk) > 1:
                                valid = False
                                break
                        if new_adj[new_v] & (1 << k):
                            # k is neighbor of new_v
                            # Check |N(i) ∩ N(k)|
                            common_ik = new_adj[i] & new_adj[k]
                            if popcount(common_ik) > 1:
                                valid = False
                                break
                    # Check pairs (i, k) for k in N(new_v)
                    for k in range(new_n):
                        if k == i:
                            continue
                        if new_adj[new_v] & (1 << k):
                            # k is neighbor of new_v
                            # Check |N(i) ∩ N(k)|
                            common_ik = new_adj[i] & new_adj[k]
                            if popcount(common_ik) > 1:
                                valid = False
                                break
                
                if not valid:
                    # Remove the edge
                    new_adj[new_v] &= ~(1 << i)
                    new_adj[i] &= ~(1 << new_v)
                else:
                    added = True
            
            if added:
                new_n += 1
            else:
                # Could not add new_v
                new_adj.pop()
                break
        
        return new_n, new_adj

    def maximize_edges(n, adj):
        """
        Add edges that do not create C4s to reduce alpha.
        """
        adj = adj[:]
        improved = True
        while improved and time_left() > 10:
            improved = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        # Check if adding (i, j) creates a C4
                        common_ij = adj[i] & adj[j]
                        if popcount(common_ij) > 1:
                            continue
                        
                        valid = True
                        # Check constraints for neighbors of j
                        mj = adj[j] & ~(1 << i)
                        while mj:
                            lsb = mj & -mj
                            k = lsb.bit_length() - 1
                            if adj[i] & adj[k]:
                                valid = False
                                break
                            mj -= lsb
                            
                        if not valid:
                            continue
                            
                        # Check constraints for neighbors of i
                        mi = adj[i] & ~(1 << j)
                        while mi:
                            lsb = mi & -mi
                            k = lsb.bit_length() - 1
                            if adj[j] & adj[k]:
                                valid = False
                                break
                            mi -= lsb

                        if valid:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                            improved = True
        return adj

    # Main construction
    n, adj = construct_polarity_graph()
    
    # Verify initial graph
    if not is_c4_free(n, adj):
        # Fallback: return empty graph
        return {'n': 1, 'edges': []}
    
    alpha = max_independent_set_size(n, adj)
    if alpha >= 11:
        # Fallback
        return {'n': 1, 'edges': []}
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend
    for target_n in range(n + 1, 40):
        if time_left() < 20:
            break
        
        # Try multiple random seeds for extension
        for seed in range(10):
            if time_left() < 10:
                break
            
            random.seed(seed * 1000 + target_n)
            
            # Shuffle the order of vertices to try different extensions
            shuffled_adj = best_adj[:]
            
            new_n, new_adj = try_extend(best_n, shuffled_adj, target_n)
            
            if new_n > best_n:
                # Verify
                if is_c4_free(new_n, new_adj):
                    alpha = max_independent_set_size(new_n, new_adj)
                    if alpha < 11:
                        best_n = new_n
                        best_adj = new_adj[:]
                        break
    
    # Final edge maximization
    if time_left() > 10:
        best_adj = maximize_edges(best_n, best_adj)
        # Verify
        if is_c4_free(best_n, best_adj):
            alpha = max_independent_set_size(best_n, best_adj)
            if alpha < 11:
                pass
            else:
                # Revert if alpha increased
                best_adj = best_adj[:]  # Keep previous
    
    # Convert to edge list
    edge_list = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return {'n': best_n, 'edges': edge_list}
