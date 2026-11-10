import time
import sys

def construct():
    """
    Finds a maximum cardinality subset of [1, 512] such that 
    lcm(a, b) <= 512 for all pairs a, b in the subset.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    N = 512
    
    # 1. Build the graph
    # vertices are 0 to N-1 representing 1 to N
    # neighbors[i] is a bitmask of neighbors of vertex i
    
    # Precompute gcd or just use math.lcm? math.lcm is fast enough for 512^2
    from math import gcd
    
    # neighbors[i] will be an integer where bit j is set if lcm(i+1, j+1) <= 512
    neighbors = [0] * N
    
    for i in range(N):
        ui = i + 1
        mask = 0
        for j in range(N):
            vj = j + 1
            # lcm(a, b) = a * b // gcd(a, b)
            # We want a * b // gcd(a, b) <= 512
            # This is equivalent to a * b <= 512 * gcd(a, b)
            # Or simply compute lcm and check.
            # Since 512 is small, direct computation is fine.
            g = gcd(ui, vj)
            lcm_val = (ui // g) * vj
            if lcm_val <= 512:
                mask |= (1 << j)
        neighbors[i] = mask
        
    # 2. Vertex Dominance Reduction
    # If N(u) \subseteq N(v), we can remove u.
    # Note: We should be careful not to remove all vertices.
    # We iterate and mark vertices for removal.
    
    # To be safe and simple, we can skip this if the graph is small enough, 
    # but 512 nodes is small. Let's do a simple reduction.
    # Actually, Bron-Kerbosch is fast enough on 512 nodes even without reduction.
    # Let's try running BK directly. If it's too slow, we add reduction.
    # Given 512 nodes, BK with pivoting is extremely fast (usually < 1s).
    
    # 3. Bron-Kerbosch with Pivoting
    best_clique_size = 0
    best_clique_vertices = []
    
    # We use a list to track the current clique to easily reconstruct it
    # But for speed, we pass bitmasks for P and X.
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_vertices
        
        if time.time() > deadline:
            return
            
        if len(P) == 0 and len(X) == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_vertices = R.copy()
            return
            
        # If P is empty, we found a maximal clique
        if len(P) == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_vertices = R.copy()
            return
            
        # Select a pivot u from P U X to minimize branching
        # We want u with maximum |P \cap N(u)|
        # P and X are lists of vertex indices
        # To make this efficient, we can use bitmasks for P and X.
        
    # Let's use bitmasks for P and X to make intersection fast.
    # R is a list of vertices.
    
    def bk_bitmask(R, P_mask, X_mask):
        nonlocal best_clique_size, best_clique_vertices
        
        if time.time() > deadline:
            return
            
        if P_mask == 0 and X_mask == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_vertices = R.copy()
            return
            
        if P_mask == 0:
            if len(R) > best_clique_size:
                best_clique_size = len(R)
                best_clique_vertices = R.copy()
            return
            
        # Find pivot u in P_mask | X_mask
        # We want to maximize the number of neighbors of u in P_mask
        # Let U = P_mask | X_mask
        # We iterate through vertices in U to find the best pivot.
        # Since N=512, iterating through all bits in U is acceptable.
        
        U = P_mask | X_mask
        if U == 0:
            return
            
        # Find pivot
        best_pivot = -1
        max_count = -1
        
        # Iterate through bits in U
        u_temp = U
        while u_temp:
            # Get lowest set bit
            lsb = u_temp & (-u_temp)
            u = lsb.bit_length() - 1
            
            # Count neighbors of u in P_mask
            # neighbors[u] is the bitmask of neighbors of u
            # Count bits in (P_mask & neighbors[u])
            count = (P_mask & neighbors[u]).bit_count()
            
            if count > max_count:
                max_count = count
                best_pivot = u
                
            u_temp &= u_temp - 1 # Clear lowest set bit
            
        # Iterate over vertices in P \ N(best_pivot)
        # These are the candidates for the next step
        candidates_mask = P_mask & ~neighbors[best_pivot]
        
        while candidates_mask:
            if time.time() > deadline:
                return
                
            # Pick a vertex v from candidates_mask
            lsb = candidates_mask & (-candidates_mask)
            v = lsb.bit_length() - 1
            
            # Recurse
            # New R = R + [v]
            # New P = P & N(v)
            # New X = X & N(v)
            new_P = P_mask & neighbors[v]
            new_X = X_mask & neighbors[v]
            
            bk_bitmask(R + [v], new_P, new_X)
            
            # Remove v from P and add to X
            P_mask &= ~lsb
            X_mask |= lsb
            candidates_mask &= ~lsb
            
    # Initial call
    # P is all vertices, X is empty
    initial_P = (1 << N) - 1
    initial_X = 0
    
    bk_bitmask([], initial_P, initial_X)
    
    # If no clique found (should not happen, at least {1} is valid), return {1}
    if not best_clique_vertices:
        return [1]
        
    # Convert back to 1-indexed values
    result = [v + 1 for v in best_clique_vertices]
    
    # Final safety check (optional, but good for correctness verification if time permits)
    # The problem asks for a valid subset. The algorithm guarantees validity.
    
    return result

# Keep the EVOLVE-BLOCK markers as per instructions? 
# The prompt says: "Keep the function names and signatures of the seed program."
# And "Output the COMPLETE program in a single python code block."
# The seed program had EVOLVE-BLOCK comments. I should probably include them to be safe 
# if the parser looks for them, but the instruction says "Rewrite the program from scratch".
# Usually, in these tasks, the code block is the whole file. 
# I will include the comments to match the structure if possible, or just the code.
# The prompt says: "Keep the program's entry-point function name(s) and their return interface EXACTLY as in the current program above"
# The current program is:
# """Initial program..."""
# # EVOLVE-BLOCK-START
# def construct(): ...
# # EVOLVE-BLOCK-END
#
# I will provide the code wrapped in these markers to be safe.
