import time
import sys
import heapq

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # We want to maximize n. 
    # Strategy: Try n=17 first? Or increment?
    # Directive: "starting from n=10 and incrementally increasing n until no solution is found or n=17 is reached"
    # This implies we find a solution for 10, then 11, etc.
    # If we find 17, we return it.
    # If we time out, we return the largest n for which we found a solution.
    
    best_sol = None
    best_n = 0
    
    # Fallback known solution for n=9 just in case
    # The seed provided n=9. Let's include a hardcoded n=9 or n=10 if easy.
    # Actually, n=10 is very likely to be found quickly.
    
    for n in range(10, 18):
        if time.time() - start_time > time_limit:
            break
            
        # Allocate time based on n
        # Higher n is harder.
        if n <= 12:
            limit_for_n = 2.0
        elif n <= 14:
            limit_for_n = 5.0
        elif n <= 16:
            limit_for_n = 10.0
        else:
            limit_for_n = 90.0 # Most time for 17
            
        # Check if we have enough time
        if time.time() - start_time + limit_for_n > time_limit + 5: # 5s buffer
             # If we can't finish this n and still return, maybe skip to next?
             # No, we must return a valid solution.
             # If we are running out of time, we should probably just return the best found so far.
             if best_sol is not None:
                 break
             # If no solution found yet, we MUST find one.
             # Force a quick search or use fallback.
             pass

        sol = solve_c6_fast(n, limit_for_n)
        if sol is not None:
            best_sol = sol
            best_n = n
            # If we found 17, we are done
            if n == 17:
                return {'n': n, 'colors': sol}
        else:
            # If we failed to find a solution for n, 
            # it doesn't mean n+1 is impossible, but it's unlikely if n is hard.
            # However, the constraint is local. 
            # The directive says "until no solution is found". 
            # So we break if we fail?
            # "incrementally increasing n until no solution is found"
            # This implies if solve_c6_fast(n) returns None, we stop increasing n.
            # But wait, if we fail n=10 (unlikely), we stop.
            # If we fail n=17, we return best_n (which is 16 if we found it).
            # This aligns with the directive.
            break

    if best_sol is None:
        # Fallback to a small known valid coloring
        # n=9 solution from seed: min(j - i, n - j + i) - 1
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
        
    return {'n': best_n, 'colors': best_sol}

def solve_c6_fast(n, time_limit):
    """
    Backtracking solver with forward checking and MRV.
    """
    start_time = time.time()
    
    # adj[c][v] is a bitmask of neighbors of v with color c
    adj = [[0] * n for _ in range(4)]
    
    # edges list
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_set = set(edges)
    
    # assigned[i][j] = color or -1
    assigned = [[-1] * n for _ in range(n)]
    
    # Unassigned edges set
    unassigned = set(edges)
    
    def get_color_mask(color, v):
        return adj[color][v]
    
    def is_valid(c, u, v):
        """
        Check if adding edge (u,v) with color c creates a C6.
        A C6 containing (u,v) is u-a-b-d-e-v-u.
        Edges (u,a), (a,b), (b,d), (d,e), (e,v) must be color c.
        We need to check if there is a path of length 4 between u and v in G_c.
        """
        # Neighbors of u in color c
        mask_u = adj[c][u]
        # Neighbors of v in color c
        mask_v = adj[c][v]
        
        # Iterate a in mask_u
        # Iterate e in mask_v
        # Check if dist(a, e) == 2 in G_c
        
        # Optimization: 
        # If a == e, we have a path u-a-v, length 2. Not a C6.
        # We need a-b-e.
        
        # Let's iterate a
        a = mask_u
        while a:
            a_bit = a & -a
            a_idx = a_bit.bit_length() - 1
            a ^= a_bit
            
            if a_idx == v: continue # a cannot be v? 
            # Actually, the path is u-a-b-d-e-v. 
            # a is neighbor of u. e is neighbor of v.
            # a and e must be distinct from u, v.
            # In the mask, u and v are not neighbors of themselves, so they are not in mask_u or mask_v.
            # But a could be v? No, edge (u,v) is being added, so v is not in adj[c][u] yet.
            # Wait, adj[c][u] contains neighbors. v is not a neighbor of u in G_c before adding (u,v).
            # So a != v and e != u.
            
            # Iterate e in mask_v
            e = mask_v
            while e:
                e_bit = e & -e
                e_idx = e_bit.bit_length() - 1
                e ^= e_bit
                
                if a_idx == e_idx:
                    continue # a and e must be distinct for a simple cycle?
                    # If a == e, we have u-a-v. This is a path of length 2.
                    # A C6 requires 6 distinct vertices.
                    # So a != e is required.
                
                # Check if there is a path of length 2 from a to e in G_c.
                # Path a-b-e.
                # b must be distinct from u, v, a, e.
                
                # Common neighbors of a and e in G_c
                # mask_a = adj[c][a]
                # mask_e = adj[c][e]
                # common = mask_a & mask_e
                
                # We need a vertex b in common such that b != u, b != v, b != a, b != e.
                # Since a and e are neighbors of u and v respectively, 
                # and we are checking for a cycle, b cannot be u or v?
                # If b == u, then a-u-e... but u is connected to a and e?
                # If b == u, then edges (a,u) and (u,e) are color c.
                # Then the cycle is u-a-u-e-v-u? No, vertices must be distinct.
                # So b cannot be u or v.
                
                mask_a = adj[c][a_idx]
                mask_e = adj[c][e_idx]
                
                common = mask_a & mask_e
                
                # Remove u, v, a, e from common
                # Bitmask for u, v, a, e
                forbidden = (1 << u) | (1 << v) | (1 << a_idx) | (1 << e_idx)
                
                if common & ~forbidden:
                    return False # Found a C6
        
        return True

    def backtrack():
        if time.time() - start_time > time_limit:
            return False
            
        if not unassigned:
            return True
            
        # MRV: Find edge with fewest valid colors
        best_edge = None
        min_valids = 5
        best_valids = []
        
        # To speed up, we can sample or just iterate all.
        # N_edges is ~136 max. Iterating all is fine.
        
        for edge in unassigned:
            u, v = edge
            valids = []
            for c in range(4):
                if is_valid(c, u, v):
                    valids.append(c)
            if len(valids) < min_valids:
                min_valids = len(valids)
                best_edge = edge
                best_valids = valids
                if min_valids == 0:
                    return False # Dead end
                if min_valids == 1:
                    break # Can't do better than 1
            
        if best_edge is None:
            return False
            
        u, v = best_edge
        unassigned.remove(best_edge)
        
        for c in best_valids:
            # Assign
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            assigned[u][v] = c
            assigned[v][u] = c
            
            if backtrack():
                return True
            
            # Unassign
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            assigned[u][v] = -1
            assigned[v][u] = -1
            
        unassigned.add(best_edge)
        return False

    # Symmetry breaking: 
    # Edge (0,1) must be 0.
    # We can force this by initially setting it?
    # Or just let MRV handle it?
    # MRV will likely pick (0,1) early if we break symmetry?
    # No, MRV picks based on constraints. Initially all edges have 4 valid colors.
    # So it picks the first one it finds.
    # To break symmetry, we can manually assign (0,1)=0, (0,2) in {0,1}, etc.
    # But MRV is dynamic.
    # Let's just run MRV. The first edge picked will be arbitrary (first in set iteration? No, set is unordered).
    # To ensure determinism and symmetry breaking, let's pick the "most constrained" edge.
    # Initially, all are equivalent.
    # We can force the first few edges.
    
    # Let's modify the loop to handle the first few edges with symmetry breaking.
    # This is tricky with MRV.
    # Alternative: Just use a fixed order for the first k edges, then MRV.
    # Or, simpler: The problem is symmetric under vertex permutation.
    # We can fix vertex 0 to be "special" by ordering edges incident to 0 first?
    
    # Let's just run the backtrack. 
    # To avoid symmetry issues causing timeout, we can add a check:
    # If we are at the start, force (0,1)=0.
    
    # Let's implement a simple symmetry breaking in the initial state.
    # Assign (0,1) = 0
    # Assign (0,2) = 0 or 1
    # Assign (0,3) = 0, 1, 2
    
    # This reduces the search space significantly.
    
    # We will do this by manually assigning these edges and removing them from unassigned.
    
    # Edge (0,1)
    e01 = (0, 1)
    if e01 in unassigned:
        unassigned.remove(e01)
        adj[0][0] |= (1 << 1)
        adj[0][1] |= (1 << 0)
        assigned[0][1] = 0
        assigned[1][0] = 0
        
    # Edge (0,2)
    e02 = (0, 2)
    if e02 in unassigned:
        unassigned.remove(e02)
        # Try 0 and 1? 
        # We can't easily branch here in the main flow without recursion.
        # Let's just let MRV handle it, but the first edge picked might be (0,1) if we sort?
        # Set iteration is arbitrary.
        
        # Let's force a deterministic order for the first few edges to break symmetry.
        # We'll create a list of "forced" assignments.
        pass

    # Actually, the simplest robust way is to use the standard backtracking order 
    # but with the is_valid check optimized.
    # The MRV approach above is good. 
    # To handle symmetry, we can just rely on the fact that finding *any* solution is enough.
    # The search space is huge, but symmetry breaking helps.
    # Let's add a specific symmetry breaking step:
    # Before calling backtrack, we can try the first edge (0,1) with color 0.
    # If that fails, we don't need to try other colors for (0,1) due to symmetry.
    
    # Let's modify backtrack to accept a "forced" edge/color for the first step?
    # Or just do it outside.
    
    # Force (0,1) = 0
    # If n >= 2
    if n >= 2:
        u, v = 0, 1
        if (u, v) in unassigned:
            unassigned.remove((u, v))
            # Check validity
            if is_valid(0, u, v):
                adj[0][u] |= (1 << v)
                adj[0][v] |= (1 << u)
                assigned[u][v] = 0
                assigned[v][u] = 0
                
                # Force (0,2) in {0, 1} if n>=3
                if n >= 3:
                    u2, v2 = 0, 2
                    if (u2, v2) in unassigned:
                        unassigned.remove((u2, v2))
                        # We need to branch here. 
                        # This complicates the single backtrack call.
                        # Let's just let MRV handle it. 
                        # The symmetry breaking for (0,1) is the most important one.
                        # (0,2) can be 0,1,2,3 but 2,3 are symmetric to 0,1? 
                        # Colors are symmetric. So (0,2) can be 0 or 1 (representing same/diff).
                        # MRV will pick the best edge.
                        
                if backtrack():
                    # Extract colors
                    colors = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            colors.append(assigned[i][j])
                    return colors
                else:
                    # Undo (0,1)
                    adj[0][u] &= ~(1 << v)
                    adj[0][v] &= ~(1 << u)
                    assigned[u][v] = -1
                    assigned[v][u] = -1
                    unassigned.add((u, v))
                    return None
            else:
                # Should not happen
                return None
    else:
        # n=1, no edges
        return []

    return None

# EVOLVE-BLOCK-END
