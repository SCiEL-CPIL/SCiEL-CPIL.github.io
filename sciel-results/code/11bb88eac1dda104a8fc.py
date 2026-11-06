import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Fallback: n=11 (Known valid: K11 decomposed into 5 Hamiltonian cycles)
    # The formula min(j-i, n-j+i)-1 creates a cyclic decomposition.
    # For n=11, this works because 11 is prime and 5 divides 11-1? No, 10/5=2.
    # Actually, a known C4-free 5-coloring of K_11 exists.
    # Let's use a simple valid construction for small n as fallback.
    # A valid 5-coloring of K_11 can be constructed by taking vertices 0..10.
    # Color(i, j) = (i + j) % 5? No, that creates C4s.
    # Let's just hardcode a known valid one or generate via search for small n if needed.
    # For now, n=11 fallback using a simple pattern that is likely valid or verified.
    # Actually, the seed program's fallback was n=11. Let's ensure it's valid.
    # Pattern: color(i,j) = (i+j) % 5 is NOT valid for C4-free.
    # Let's use a search for small n to get a valid fallback quickly if algebraic fails.
    
    best_n = 11
    # Generate a valid coloring for n=11 using a quick backtracking search
    # This is fast and ensures a valid baseline.
    colors_fallback = _backtrack_search(11, time.time() + 5.0)
    if colors_fallback is not None:
        best_colors = colors_fallback
    else:
        # If even n=11 fails (unlikely), use random and hope? No, must be valid.
        # Let's try a specific known valid construction for K_11.
        # Vertices 0..10.
        # One valid 5-coloring of K_11 is based on the Paley graph of order 11?
        # Paley(11) is a 5-regular graph. Its complement is also 5-regular.
        # We need 5 colors, each C4-free.
        # Let's just trust the backtracking for n=11.
        best_colors = [0] * (11*10//2) # Placeholder, will be overwritten by search below if needed

    # Helper: Index of edge (i, j) in lexicographic list
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Backtracking search with forward checking
    # State:
    #   colors: list of current colors for edges assigned so far
    #   n: number of vertices
    #   assigned: boolean mask or set of assigned edges?
    #   neighbor_counts[c][u][v]: number of common neighbors of u and v in color c.
    #   Actually, it's easier to maintain:
    #   adj[c][u]: bitmask of neighbors of u in color c.
    #   When assigning edge (u, v) color c:
    #   Check if (u, v) already has >1 common neighbor in c? No, (u,v) is the edge.
    #   The constraint is: No two vertices x, y can have >1 common neighbor in any color.
    #   If we add edge (u, v) in color c:
    #   1. For any w, if w is a common neighbor of u and v in c, then adding (u,v) doesn't change common neighbors of (u,v)?
    #      Wait. The C4 condition is about common neighbors.
    #      If we add edge (u, v) in color c:
    #      - u and v become neighbors in c.
    #      - For any pair (x, y), their common neighbors in c might change.
    #      Specifically, u is a neighbor of v. v is a neighbor of u.
    #      So for any x, if x is a neighbor of u in c, then x and v now share u as a common neighbor?
    #      No. Common neighbors of x and v are N_c(x) ∩ N_c(v).
    #      Since u ∈ N_c(v) (new edge), if u ∈ N_c(x), then u is a common neighbor of x and v.
    #      So we must ensure that for all x, |N_c(x) ∩ N_c(v)| <= 1 after adding u to N_c(v).
    #      This means: Before adding, |N_c(x) ∩ N_c(v)| must be 0 if u ∈ N_c(x), because adding u would make it 1.
    #      If |N_c(x) ∩ N_c(v)| was already 1, and u is NOT in that intersection, it stays 1.
    #      If |N_c(x) ∩ N_c(v)| was 1 and u IS in that intersection? Impossible, u is new neighbor.
    #      Wait, u is being added to N_c(v).
    #      So for any x, the new common neighbors of x and v are (OldCommon(x,v) ∪ {u}) if u ∈ N_c(x).
    #      So we need:
    #      If u ∈ N_c(x), then OldCommon(x,v) must be empty. i.e., N_c(x) ∩ N_c(v) = 0.
    #      Similarly for y and u.
    #      
    #      Also, does adding (u,v) affect common neighbors of other pairs (a,b)?
    #      Only if u or v is one of a,b.
    #      So the check is local to pairs involving u or v.
    #      
    #      Algorithm for adding edge (u, v) with color c:
    #      1. Check if u and v are already connected in c? (Shouldn't happen in complete graph coloring).
    #      2. For all x in N_c(u):
    #         Check if x and v have any common neighbor in c.
    #         If yes, then adding u to N_c(v) would make u a second common neighbor?
    #         No. u is a neighbor of x (by definition x in N_c(u)).
    #         So u is a candidate common neighbor of x and v.
    #         If x and v already have a common neighbor w (w != u), then adding u makes it 2. Invalid.
    #         So: For all x in N_c(u), we must have N_c(x) ∩ N_c(v) == 0.
    #      3. For all y in N_c(v):
    #         Similarly, N_c(y) ∩ N_c(u) == 0.
    #      4. Additionally, check if u and v themselves have common neighbors?
    #         The pair (u, v) is not constrained to have 0 common neighbors, they are the edge.
    #         But wait, the constraint is "No two vertices with two common neighbours".
    #         This applies to ANY pair of vertices, including adjacent ones?
    #         Yes. If u and v are connected by color c, and they share two other neighbors w, z in color c,
    #         then {u, v, w, z} forms a C4?
    #         Edges: (u,v) in c. (u,w) in c? (w,v) in c?
    #         If w is a common neighbor, (u,w) in c and (v,w) in c.
    #         If z is a common neighbor, (u,z) in c and (v,z) in c.
    #         Then u-w-v-z-u is a C4 in color c.
    #         So yes, even adjacent vertices cannot have >1 common neighbor in the same color.
    #         So we must also check: |N_c(u) ∩ N_c(v)| <= 1.
    #         But since we are adding (u,v), u and v are now neighbors.
    #         The common neighbors are N_c(u) ∩ N_c(v).
    #         This count must be <= 1.
    #         However, the check in step 2/3 covers the "new" common neighbors created by the edge?
    #         No. Step 2 checks if adding u to N_c(v) creates a C4 involving x, v, u, w.
    #         The C4 is x-u-v-w-x?
    #         Edges: (x,u) in c, (u,v) in c (new), (v,w) in c, (w,x) in c.
    #         This requires w to be a common neighbor of x and v.
    #         So if N_c(x) ∩ N_c(v) is non-empty, and x is a neighbor of u, then adding (u,v) creates a C4.
    #         This matches the condition: For all x in N_c(u), N_c(x) ∩ N_c(v) must be empty.
    #         
    #         What about the pair (u, v) themselves?
    #         Do we need to check |N_c(u) ∩ N_c(v)| <= 1?
    #         If there is a common neighbor w, then (u,w) in c and (v,w) in c.
    #         With (u,v) in c, we have triangle u-v-w-u.
    #         A C4 requires 4 distinct vertices.
    #         If there are TWO common neighbors w, z, then we have C4 u-w-v-z-u.
    #         So yes, |N_c(u) ∩ N_c(v)| must be <= 1.
    #         But note: N_c(u) and N_c(v) do NOT include each other yet in the "before" state.
    #         So we check the intersection of current neighbors.
    
    def _backtrack_search(n, deadline):
        # State
        colors = [-1] * (n * (n - 1) // 2)
        adj = [[0] * n for _ in range(5)]  # adj[c][u] is bitmask of neighbors in color c
        
        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Heuristic: Order edges? 
        # Random order with restarts is often better for SAT-like problems.
        # But for backtracking, a fixed order allows for more efficient pruning if we use symmetry breaking.
        # Let's try a random permutation of edges for each restart.
        
        def is_valid_assignment(u, v, c):
            # Check if adding edge (u, v) in color c is valid
            # 1. Check intersection of N_c(u) and N_c(v)
            #    Current common neighbors: adj[c][u] & adj[c][v]
            #    Must have bit_count <= 1
            if (adj[c][u] & adj[c][v]).bit_count() > 1:
                return False
            
            # 2. For all x in N_c(u), check if N_c(x) intersects N_c(v)
            #    If yes, then adding u to N_c(v) creates a C4 x-u-v-w-x where w is common neighbor of x,v.
            #    Wait, the C4 is x-u-v-w-x.
            #    Edges: (x,u) in c, (u,v) in c (new), (v,w) in c, (w,x) in c.
            #    This requires w in N_c(v) and w in N_c(x).
            #    So if N_c(x) & N_c(v) != 0, it's invalid.
            mask_u = adj[c][u]
            mask_v = adj[c][v]
            
            # Iterate over bits in mask_u
            temp = mask_u
            while temp:
                lsb = temp & -temp
                x = lsb.bit_length() - 1
                if adj[c][x] & mask_v:
                    return False
                temp ^= lsb
            
            # 3. For all y in N_c(v), check if N_c(y) intersects N_c(u)
            temp = mask_v
            while temp:
                lsb = temp & -temp
                y = lsb.bit_length() - 1
                if adj[c][y] & mask_u:
                    return False
                temp ^= lsb
            
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == len(edges):
                return True
            
            u, v = edges[idx]
            e_idx = get_idx(u, v, n)
            
            # Try colors 0-4
            # Randomize color order to break symmetry
            color_order = [0, 1, 2, 3, 4]
            random.shuffle(color_order)
            
            for c in color_order:
                if is_valid_assignment(u, v, c):
                    # Assign
                    colors[e_idx] = c
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    
                    if backtrack(idx + 1):
                        return True
                    
                    # Unassign
                    colors[e_idx] = -1
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
            
            return False

        # Try with random edge orderings
        for attempt in range(100):
            if time.time() > deadline:
                break
            random.shuffle(edges)
            # Reset state
            for i in range(n):
                for c in range(5):
                    adj[c][i] = 0
            for i in range(len(colors)):
                colors[i] = -1
            
            if backtrack(0):
                return colors
        
        return None

    # Strategy:
    # 1. Try to find a valid coloring for n=26 using backtracking.
    #    This is likely to fail quickly due to the large search space, but it's worth a shot with a tight time limit.
    #    However, backtracking on 325 edges is too slow for random order.
    #    We need a better initialization or a different search.
    
    # Alternative Strategy:
    # Use the algebraic construction for n=26 if we can derive it.
    # The Lazebnik-Woldar construction for R_5(C4) > 26.
    # Let's try to implement a specific known construction.
    # One such construction uses the finite field GF(5).
    # Vertices: 26 = 5^2 + 1? No.
    # Actually, there is a construction for n = q^2 + q + 1? No, that's projective plane.
    # For q=5, q^2+q+1 = 31.
    # The known bound is R_5(C4) > 26.
    # A construction for 26 vertices:
    # Take the affine plane AG(2, 5) which has 25 points.
    # Add one point at infinity?
    # Or use a different geometry.
    
    # Let's try a hybrid:
    # 1. Use backtracking for n up to 15 (fast).
    # 2. For n > 15, use a local search (Simulated Annealing) but with a much better initialization.
    #    Initialization: Use the result of backtracking for n-1 and extend?
    #    Or use a structured initialization based on modular arithmetic.
    
    # Let's try to improve the local search approach from the seed, but with:
    # 1. Faster violation counting using bitmasks (already in seed).
    # 2. Better initialization: 
    #    - Try to find a valid coloring for n=20, 21, ... via backtracking (might work for smaller n).
    #    - Use that as a seed for larger n? No, n changes.
    #    - Use a "greedy" initialization: Assign colors one by one, choosing the color that minimizes immediate C4 creation.
    
    def greedy_init(n):
        colors = [-1] * (n * (n - 1) // 2)
        adj = [[0] * n for _ in range(5)]
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        # Shuffle edges to break symmetry? Or use a specific order?
        # A specific order might be better for greedy.
        # Let's use lexicographic order.
        
        for u, v in edges:
            e_idx = get_idx(u, v, n)
            best_c = -1
            best_score = float('inf')
            
            for c in range(5):
                # Score: Number of potential C4s created
                # 1. Common neighbors of u and v in c
                common_uv = (adj[c][u] & adj[c][v]).bit_count()
                if common_uv > 1:
                    continue # Invalid
                
                # 2. For x in N_c(u), check if N_c(x) intersects N_c(v)
                score = 0
                mask_u = adj[c][u]
                mask_v = adj[c][v]
                
                temp = mask_u
                while temp:
                    lsb = temp & -temp
                    x = lsb.bit_length() - 1
                    # If x and v have a common neighbor, adding u-v creates a C4?
                    # No, it creates a "threat".
                    # We want to minimize the number of pairs (x, v) such that x in N_c(u) and N_c(x) & N_c(v) != 0.
                    if adj[c][x] & mask_v:
                        score += 1
                    temp ^= lsb
                
                temp = mask_v
                while temp:
                    lsb = temp & -temp
                    y = lsb.bit_length() - 1
                    if adj[c][y] & mask_u:
                        score += 1
                    temp ^= lsb
                
                if score < best_score:
                    best_score = score
                    best_c = c
            
            if best_c == -1:
                # No valid color, pick random
                best_c = random.randint(0, 4)
            
            colors[e_idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
        
        return colors

    # Local Search with Optimized Violation Count
    def count_violations_fast(colors, n):
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        violations = 0
        for c in range(5):
            for u in range(n):
                mask_u = adj[c][u]
                if mask_u == 0:
                    continue
                for v in range(u + 1, n):
                    mask_v = adj[c][v]
                    if mask_v == 0:
                        continue
                    common = mask_u & mask_v
                    if common:
                        cnt = common.bit_count()
                        if cnt > 1:
                            violations += cnt - 1
        return violations

    def local_search(colors, n, max_time):
        current_violations = count_violations_fast(colors, n)
        if current_violations == 0:
            return colors, True
        
        best_colors = colors[:]
        best_violations = current_violations
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        
        temp = 5.0
        cooling = 0.99
        min_temp = 0.01
        
        while temp > min_temp and time.time() < max_time:
            e_idx = random.randint(0, num_edges - 1)
            u, v = edges[e_idx]
            
            old_color = colors[e_idx]
            new_color = random.randint(0, 4)
            if new_color == old_color:
                continue
            
            colors[e_idx] = new_color
            new_violations = count_violations_fast(colors, n)
            
            delta = new_violations - current_violations
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                colors[e_idx] = old_color
            
            temp *= cooling
            
        if best_violations == 0:
            return best_colors, True
        return best_colors, False

    # Main Execution
    # 1. Try backtracking for n=20 (might be too slow, so limit time)
    # 2. Try greedy init + local search for n=26, 25, ...
    
    # Try backtracking for n=15 first to see if it's fast enough
    # If n=15 works, try n=16, 17...
    for n_target in range(15, 26, -1):
        if time.time() > deadline - 20:
            break
        
        # Try backtracking with short time limit
        t_limit = time.time() + 5.0
        bt_colors = _backtrack_search(n_target, t_limit)
        if bt_colors is not None:
            best_n = n_target
            best_colors = bt_colors
            break
    
    if best_n < 26:
        # Try local search for larger n
        for n_target in range(26, best_n, -1):
            if time.time() > deadline - 10:
                break
            
            # Initialize using greedy
            init_colors = greedy_init(n_target)
            
            # Run local search
            t_limit = time.time() + 10.0
            result_colors, is_valid = local_search(init_colors, n_target, t_limit)
            
            if is_valid:
                best_n = n_target
                best_colors = result_colors
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
