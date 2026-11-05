import time
import random
import sys

def construct():
    deadline = time.time() + 115
    
    # --- Field F_49 Arithmetic ---
    # F_49 = F_7[x] / (x^2 - 3)
    # Elements are (a, b) representing a + bx
    def f49_add(a, b):
        return ((a[0] + b[0]) % 7, (a[1] + b[1]) % 7)
    
    def f49_mul(a, b):
        return ((a[0]*b[0] + 3*a[1]*b[1]) % 7, (a[0]*b[1] + a[1]*b[0]) % 7)
        
    def f49_neg(a):
        return ((-a[0]) % 7, (-a[1]) % 7)
        
    def f49_eq(a, b):
        return a[0] == b[0] and a[1] == b[1]
        
    # Generate all elements
    elements = [(a, b) for a in range(7) for b in range(7)]
    zero = (0, 0)
    
    # Find non-zero elements
    non_zero = [e for e in elements if not f49_eq(e, zero)]
    
    # We need to partition non_zero into 4 sets S0..S3
    # Such that for each S, there are no x,y,z in S with x+y=z
    # Note: x,y,z must be distinct in the context of the graph, 
    # but x+y=z implies distinctness if x,y,z non-zero (x=y=0 is excluded).
    # If x=y, 2x=z. If 2x=z and x,z in S, is that a triangle?
    # Triangle vertices u,v,w. u-v=x, v-w=y, w-u=z.
    # If x=y, then u-v = v-w => u+w = 2v.
    # The colors are determined by differences.
    # If S contains x, y, z and x+y=z, then we have a mono triangle.
    
    # Local search to find partition
    # Initialize randomly
    sets = [set() for _ in range(4)]
    for e in non_zero:
        sets[random.randint(0, 3)].add(e)
        
    # Precompute sum map for quick lookup?
    # sum_map[s] = list of (x, y) such that x+y=s? No, just check x+y=z.
    
    def count_bad(sets):
        bad = 0
        for s in sets:
            s_list = list(s)
            # Check all pairs x, y in s
            # If x+y is in s, it's a bad triple.
            # To avoid double counting, we can just count triples.
            # For efficiency, use a set for O(1) lookup.
            s_set = s
            for i in range(len(s_list)):
                x = s_list[i]
                for j in range(i + 1, len(s_list)):
                    y = s_list[j]
                    z = f49_add(x, y)
                    if z in s_set:
                        bad += 1
        return bad
        
    best_bad = count_bad(sets)
    current_bad = best_bad
    
    # Local search
    max_iter = 5000
    for _ in range(max_iter):
        if current_bad == 0:
            break
        # Pick a random element to move
        e = random.choice(non_zero)
        # Find current set
        cur_set_idx = -1
        for i in range(4):
            if e in sets[i]:
                cur_set_idx = i
                break
        
        # Try moving to another set
        target_idx = random.choice([i for i in range(4) if i != cur_set_idx])
        
        # Calculate delta
        # Remove e from cur, add to target
        # We just recompute or do incremental.
        # Given small size, recompute is fast enough for 5000 iters?
        # 48 elements, 12 per set. 12*11/2 = 66 pairs per set. 4*66 = 264 checks.
        # 5000 * 264 = 1.3M ops. Very fast.
        
        sets[cur_set_idx].remove(e)
        sets[target_idx].add(e)
        
        new_bad = count_bad(sets)
        
        if new_bad < current_bad:
            current_bad = new_bad
            if current_bad < best_bad:
                best_bad = current_bad
                best_sets = [set(s) for s in sets]
        else:
            # Revert
            sets[target_idx].remove(e)
            sets[cur_set_idx].add(e)
            
        if time.time() > deadline - 20:
            break

    if current_bad > 0:
        # Use best found
        sets = best_sets
        current_bad = best_bad
        
    if current_bad > 0:
        # Fallback to seed logic if partition fails
        # But it should succeed.
        pass

    # Build K_49 coloring
    n_49 = 49
    # Map element to index 0..48
    elem_to_idx = {e: i for i, e in enumerate(non_zero)}
    
    # colors_49[i][j] for i,j in 0..48
    # We store as a list of lists or a flat array
    # Let's use a 2D list for clarity in construction, then flatten
    colors_49 = [[-1] * n_49 for _ in range(n_49)]
    
    for i in range(n_49):
        for j in range(i + 1, n_49):
            ei = non_zero[i]
            ej = non_zero[j]
            diff = f49_add(ei, f49_neg(ej))
            # Find which set diff belongs to
            for c in range(4):
                if diff in sets[c]:
                    colors_49[i][j] = c
                    colors_49[j][i] = c
                    break
    
    # Now add vertex 49
    N = 50
    num_edges = N * (N - 1) // 2
    final_colors = [0] * num_edges
    
    # Helper to get edge index
    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        # Lexicographic order
        # (0,1), (0,2), ..., (0,49), (1,2), ...
        # Index = sum_{k=0}^{u-1} (N-1-k) + (v-u-1)
        # Or simpler:
        idx = 0
        for k in range(u):
            idx += N - 1 - k
        idx += v - u - 1
        return idx

    # Precompute edge colors for K_49 in final array
    for i in range(n_49):
        for j in range(i + 1, n_49):
            idx = get_edge_idx(i, j)
            final_colors[idx] = colors_49[i][j]
    
    # Now assign colors to edges (49, i) for i=0..48
    # Let c_i be the color of edge (49, i)
    # Constraint: For all i < j, if c_i == c_j == colors_49[i][j], then BAD.
    # We want to minimize violations.
    
    # Local search on c[0..48]
    c = [random.randint(0, 3) for _ in range(n_49)]
    
    def count_violations(c):
        violations = 0
        for i in range(n_49):
            for j in range(i + 1, n_49):
                if c[i] == c[j] == colors_49[i][j]:
                    violations += 1
        return violations
        
    best_c = c[:]
    best_v = count_violations(c)
    current_v = best_v
    
    max_iter_2 = 20000
    for _ in range(max_iter_2):
        if current_v == 0:
            break
        if time.time() > deadline - 5:
            break
            
        i = random.randint(0, n_49 - 1)
        old_c = c[i]
        new_c = random.choice([x for x in range(4) if x != old_c])
        
        c[i] = new_c
        
        # Recalculate violations involving i
        # We can just recompute total for simplicity if N is small
        # 49*48/2 = 1176 checks. 20000 * 1176 = 23M. Might be tight in Python.
        # Let's do incremental.
        
        # Actually, let's just recompute if it's fast enough.
        # 20k iterations * 1k ops = 20M ops. In Python, ~1-2 seconds. Acceptable.
        
        new_v = count_violations(c)
        
        if new_v < current_v:
            current_v = new_v
            if current_v < best_v:
                best_v = current_v
                best_c = c[:]
        else:
            c[i] = old_c
            
    if best_v == 0:
        # Success! Assign best_c to final_colors
        for i in range(n_49):
            idx = get_edge_idx(i, 49)
            final_colors[idx] = best_c[i]
            
        return {'n': N, 'colors': final_colors}
        
    # If local search failed, try backtracking on the 49 variables?
    # Or just return n=49? No, K_49 is valid, but we want max n.
    # If we can't color vertex 49, we return n=49.
    # But K_49 is valid.
    
    # Construct n=49 output
    colors_49_flat = [0] * (n_49 * (n_49 - 1) // 2)
    idx = 0
    for i in range(n_49):
        for j in range(i + 1, n_49):
            colors_49_flat[idx] = colors_49[i][j]
            idx += 1
            
    return {'n': n_49, 'colors': colors_49_flat}

