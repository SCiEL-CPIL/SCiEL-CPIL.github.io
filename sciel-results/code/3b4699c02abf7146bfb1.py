import random
import time

def construct():
    N = 16
    # List of all K4s as tuples of 4 vertices
    # We need the edges for each K4 to check colors
    k4s = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                for l in range(k+1, N):
                    k4s.append((i, j, k, l))
                    
    # Precompute edges for each K4
    # k4_edges[ idx ] = list of 6 (u,v) pairs
    k4_edges = []
    for idx, (i, j, k, l) in enumerate(k4s):
        edges = [(i, j), (i, k), (i, l), (j, k), (j, l), (k, l)]
        k4_edges.append(edges)
        
    # Map edge to list of K4 indices containing it
    edge_to_k4s = {}
    for idx, edges in enumerate(k4_edges):
        for e in edges:
            if e not in edge_to_k4s:
                edge_to_k4s[e] = []
            edge_to_k4s[e].append(idx)
            
    # Initialize with a proper edge coloring (1-factorization)
    # Using the standard circle method for K_16
    # Vertices 0..14, 15 is infinity
    # Colors 0..14
    colors = [[-1]*N for _ in range(N)]
    
    # Generate 1-factorization
    # For K_{2m}, m=8.
    # Fix vertex 15.
    # Other vertices 0..14.
    # For step s in 0..14:
    #   Edge (15, s) has color s? No, that's not a matching.
    #   Standard construction:
    #   Vertices v_0, ..., v_{2m-2}, v_inf.
    #   For i in 0..2m-2:
    #      Edge (v_inf, v_i) is in factor i? No.
    #      
    #   Let's use a simpler random initialization that is proper.
    #   Or just random assignment and repair?
    #   Random proper coloring is hard to generate directly without backtracking.
    #   Let's use the algebraic construction for K_16.
    #   Colors are in Z_15? No, 15 colors.
    #   Vertices in Z_15 + {inf}.
    #   Edge (i, j) for i,j in Z_15: color = (i+j)/2 mod 15? 
    #   If i != j.
    #   Edge (inf, i): color = i.
    #   Check properness:
    #   (i, j) and (i, k): (i+j)/2 != (i+k)/2 if j!=k.
    #   (i, j) and (j, k): (i+j)/2 != (j+k)/2 if i!=k.
    #   (inf, i) and (inf, j): i != j.
    #   (inf, i) and (i, j): i != (i+j)/2 => 2i != i+j => i != j.
    #   So this is a proper edge coloring using 15 colors.
    
    # Map vertices 0..15 to Z_15 + inf
    # Let 15 be inf.
    # 0..14 are Z_15.
    
    for i in range(N):
        for j in range(i+1, N):
            if i == 15 or j == 15:
                # One is inf
                other = j if i == 15 else i
                colors[i][j] = other % 15
                colors[j][i] = other % 15
            else:
                # Both in Z_15
                # Color = (i + j) * inv(2) mod 15
                # 2 * 8 = 16 = 1 mod 15. So inv(2) = 8.
                c = ((i + j) * 8) % 15
                colors[i][j] = c
                colors[j][i] = c

    # Function to get distinct colors in a K4
    def get_distinct_colors(k4_idx):
        edges = k4_edges[k4_idx]
        cols = set()
        for u, v in edges:
            cols.add(colors[u][v])
        return len(cols)
        
    # Function to check if a K4 is bad
    def is_bad(k4_idx):
        return get_distinct_colors(k4_idx) < 5

    # Calculate initial bad count
    bad_count = 0
    bad_status = [False] * len(k4s)
    for idx in range(len(k4s)):
        if is_bad(idx):
            bad_count += 1
            bad_status[idx] = True
            
    if bad_count == 0:
        # Convert to list in lexicographic order
        res = []
        for i in range(N):
            for j in range(i+1, N):
                res.append(colors[i][j])
        return res

    # Simulated Annealing
    start_time = time.time()
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We want to minimize bad_count
    # Energy = bad_count
    
    while T > min_T:
        if time.time() - start_time > 110:
            break
            
        # Pick a random edge
        u = random.randint(0, N-1)
        v = random.randint(0, N-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        old_c = colors[u][v]
        
        # Pick a random new color
        new_c = random.randint(0, 14)
        if new_c == old_c:
            continue
            
        # Calculate Delta E
        # Affected K4s are those containing (u,v)
        affected = edge_to_k4s[(u, v)]
        delta = 0
        
        # Store old states to revert if rejected
        old_states = []
        
        for k4_idx in affected:
            old_distinct = get_distinct_colors(k4_idx)
            old_bad = old_distinct < 5
            old_states.append((k4_idx, old_bad))
            
            # Temporarily change
            colors[u][v] = new_c
            colors[v][u] = new_c
            
            new_distinct = get_distinct_colors(k4_idx)
            new_bad = new_distinct < 5
            
            if old_bad and not new_bad:
                delta -= 1
            elif not old_bad and new_bad:
                delta += 1
                
            colors[u][v] = old_c
            colors[v][u] = old_c
            
        # Acceptance
        if delta < 0:
            # Accept
            colors[u][v] = new_c
            colors[v][u] = new_c
            bad_count += delta
            for k4_idx, old_bad in old_states:
                # Recalculate status? Or just update
                # We know the transition
                # We need to update bad_status array
                # Actually, we can just recompute bad_status for affected
                pass
            # Update bad_status properly
            for k4_idx, old_bad in old_states:
                # We changed the color, so we need to know the new bad status
                # We calculated new_bad in the loop above, but didn't store it.
                # Let's just recompute for these few K4s
                bad_status[k4_idx] = is_bad(k4_idx)
        else:
            # Metropolis
            if random.random() < pow(2.71828, -delta / T):
                colors[u][v] = new_c
                colors[v][u] = new_c
                bad_count += delta
                for k4_idx, old_bad in old_states:
                    bad_status[k4_idx] = is_bad(k4_idx)
                    
        T *= cooling
        
        # Optional: If bad_count == 0, break
        if bad_count == 0:
            break
            
    # Final check
    if bad_count == 0:
        res = []
        for i in range(N):
            for j in range(i+1, N):
                res.append(colors[i][j])
        return res
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
