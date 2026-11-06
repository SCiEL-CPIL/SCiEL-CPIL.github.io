import time
import sys

def construct():
    start_time = time.time()
    timeout = 60.0  # Safety margin before the 90s hard kill
    
    n = 256
    # conflict[i] is a bitmask of indices j such that (i+1) and (j+1) conflict
    # i, j in [0, 255] corresponding to values 1..256
    conflict = [0] * n
    
    # Precompute conflict masks
    # (a*b) % (a+b) == 0
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # We want to find the Maximum Independent Set.
    # Independent Set in G is Clique in Complement G.
    # However, Tomita's algorithm is typically described for MIS directly 
    # using coloring on the candidate set in the original graph context 
    # (where edges are conflicts).
    # Wait, standard Tomita finds Max Clique. 
    # Max Independent Set in G is Max Clique in G_complement.
    # Let's stick to MIS formulation directly if possible, or convert to Clique.
    # It is often easier to implement Max Clique algorithms (like Tomita) 
    # and apply them to the complement graph.
    # Let's define comp_conflict[i] as the mask of vertices NON-adjacent to i in the conflict graph.
    # i.e., comp_conflict[i] = ~conflict[i] & ((1<<n)-1) & ~(1<<i)
    
    # Actually, let's just implement MIS directly with coloring bound.
    # In MIS, we select vertices such that no two are connected by an edge.
    # The coloring bound for MIS: 
    # We color the candidate vertices such that no two vertices of the same color are adjacent in the conflict graph.
    # The number of colors used is an upper bound on the size of the MIS in the candidate set.
    # This is the same as finding the Chromatic Number (approx) of the candidate subgraph.
    # Greedy coloring:
    # 1. Sort candidates by degree (descending) or other heuristic.
    # 2. Assign colors greedily.
    
    # Let's use the standard approach: Convert to Max Clique problem on the complement graph.
    # Let H be the complement of the conflict graph.
    # Edge (u,v) in H iff (u,v) NOT in conflict graph.
    # We want Max Clique in H.
    # Tomita's algorithm (BBMC) is very efficient for this.
    
    # Precompute adjacency masks for H (complement of conflict)
    # adj[i] in H is the set of j such that j is not in conflict[i] and j != i.
    adj = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        # ~conflict[i] gives 1s where there is NO conflict.
        # We need to exclude i itself and bits outside n.
        adj[i] = (~conflict[i]) & all_mask
        adj[i] &= ~(1 << i) # Remove self

    # Order vertices by degree in H (descending) to help coloring?
    # Tomita usually processes vertices in a specific order.
    # Let's precompute the order of vertices to process.
    # Higher degree in H (fewer conflicts) is often a good heuristic for Clique.
    degrees = [adj[i].bit_count() for i in range(n)]
    # Sort indices by degree descending
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Map original index to position in order? 
    # Or just iterate through candidates in the order of 'order' list?
    # The BBMC algorithm typically takes a set of candidates and picks the next vertex 
    # from a fixed ordering.
    
    # Let's implement a recursive BBMC.
    # P: set of candidates
    # X: set of vertices already processed (not in P) - used for coloring bound in some variants, 
    #     but the standard "coloring bound" uses the greedy coloring of P.
    
    best_clique_size = 0
    best_clique_mask = 0

    # Greedy Coloring for Upper Bound
    # Input: set of candidates P
    # Returns: list of colors (integers) for each vertex in P, and the max color used.
    # The max color is the upper bound.
    # We need to be careful with performance. 
    # A simple greedy coloring:
    # 1. Get list of vertices in P.
    # 2. Sort them by degree in P (descending).
    # 3. Assign smallest available color not used by neighbors.
    
    # Optimization: Pre-compute static ordering? 
    # Dynamic coloring is expensive. 
    # However, for N=256, it's manageable if we prune well.
    
    def get_color_bound(P):
        # Returns max color used in greedy coloring of P
        if P == 0:
            return 0
            
        # Extract vertices
        verts = []
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            verts.append(i)
            p ^= lsb
            
        # Sort by degree in P descending
        # Degree of v in P is popcount(adj[v] & P)
        # To save time, we can just sort by global degree? 
        # Global degree might be a decent proxy and is precomputed.
        # Let's try sorting by global degree first. If too slow, optimize.
        verts.sort(key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        max_color = 0
        
        for v in verts:
            # Find colors used by neighbors of v in P
            used_colors = set()
            neigh = adj[v] & P
            # Iterate neighbors
            n_p = neigh
            while n_p:
                lsb = n_p & -n_p
                u = lsb.bit_length() - 1
                if u in colors:
                    used_colors.add(colors[u])
                n_p ^= lsb
            
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            
            colors[v] = c
            if c > max_color:
                max_color = c
            
        return max_color

    def bbmc(P, X, current_clique):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() - start_time > timeout:
            return
            
        if P == 0:
            if len(current_clique) > best_clique_size:
                best_clique_size = len(current_clique)
                # Reconstruct mask from current_clique
                mask = 0
                for v in current_clique:
                    mask |= (1 << v)
                best_clique_mask = mask
            return
            
        # Upper bound
        bound = get_color_bound(P)
        
        # If current size + bound <= best, prune
        if len(current_clique) + bound <= best_clique_size:
            return
            
        # Select next vertex from P
        # Heuristic: Pick vertex with highest degree in P
        # Or just follow the global order?
        # Let's pick the vertex from P that is first in our global 'order' list.
        # This ensures consistent branching.
        
        # Find the vertex in P that is earliest in 'order'
        # 'order' is a list of indices sorted by global degree desc.
        # We need to find v in P such that v is in order.
        # Since P is a bitmask, we can iterate P and check against a rank array?
        # Or simply iterate P and pick the one with max global degree?
        
        # Let's just pick the vertex in P with the highest global degree.
        v_idx = -1
        max_deg = -1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            if degrees[i] > max_deg:
                max_deg = degrees[i]
                v_idx = i
            p ^= lsb
            
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        
        # Branch 1: Include v_idx
        # New P = P & adj[v_idx] (neighbors of v in H)
        # Note: In Clique, if we include v, all other vertices in the clique must be neighbors of v.
        new_P = P & adj[v_idx]
        # Remove v from P for the next step? 
        # The standard BBMC removes v from P and adds to X, but since we branch on P & adj[v], 
        # v is effectively removed.
        # We also need to remove v from the "available" set for the sibling branch.
        
        current_clique.append(v_idx)
        bbmc(new_P, X, current_clique)
        current_clique.pop()
        
        # Branch 2: Exclude v_idx
        # New P = P without v_idx
        # X is updated but X is not used in the simple coloring bound version 
        # (some variants use X to tighten bound, but simple P coloring is standard Tomita).
        new_P_exc = P & ~v_bit
        bbmc(new_P_exc, X, current_clique)

    # Initial call
    initial_P = all_mask
    current_clique = []
    bbmc(initial_P, 0, current_clique)
    
    # Convert best_clique_mask to list of values
    if best_clique_size == 0:
        # Fallback: return a single element
        return [1]
        
    result = []
    for i in range(n):
        if (best_clique_mask >> i) & 1:
            result.append(i + 1)
            
    return result

# To ensure we don't hit the hard timeout, we can wrap the call in construct
# but the internal timeout handles it.
# The function construct is the entry point.

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a = res[i]
            b = res[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
