import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115  # safety margin before 120s limit

    # --- Greenwood-Gleason base: Clebsch graph (16 vertices, 3 colors) ---
    # Clebsch graph: vertices are 4-bit strings; edge if Hamming distance 1 or 3
    # This is a 3-colorable triangle-free graph (actually it's the complement of the
    # Clebsch graph that's triangle-free... let me use the known construction)
    
    # Known: The Clebsch graph on 16 vertices can be 3-colored without monochromatic triangles.
    # Actually, the standard construction: GF(16) multiplicative group, partition into 3 cosets
    # of the subgroup of cubes. This gives a 3-coloring of K_16 with no mono triangle.
    
    # Let me use the bitset approach for the base 16-vertex coloring
    # Clebsch graph coloring: vertex i (0..15), color based on log_2(i) mod 3 for i in 1..15, 
    # and vertex 0 needs special handling. Actually the standard is:
    # Work in GF(16)*, which is cyclic of order 15. Split into 3 cosets of the order-5 subgroup.
    
    # GF(16) with primitive polynomial x^4 + x + 1 (0b10011)
    def gf16_mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        # reduce
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Build log table for GF(16)*
    gf16_log = {}
    gf16_exp = [0] * 16
    x = 1
    for e in range(15):
        gf16_log[x] = e
        gf16_exp[e] = x
        x = gf16_mul(x, 2)
    # gf16_exp[15] should be 1 again
    
    # The subgroup of cubes: {g^(3k) : k=0,1,2} where g=2 is generator
    # Cubes: 2^0=1, 2^3=8, 2^6=64 mod... wait, in GF(16)*, 2^5=1? No, order is 15.
    # 2^0=1, 2^3=8, 2^6=2^6. Let me compute: 2^1=2, 2^2=4, 2^3=8, 2^4=16^... 
    # In GF(16) with x^4+x+1: alpha=2, alpha^2=4, alpha^3=8, alpha^4=alpha+1=3, 
    # alpha^5=2*3=6, alpha^6=2*6=12, alpha^7=2*12=24 mod (x^4+x+1)... 
    # Let me just use the log table.
    
    # Cubes in GF(16)*: elements whose discrete log is divisible by 3
    # So cosets are: {e: e%3==0}, {e: e%3==1}, {e: e%3==2}
    # Color of edge (i,j) for i,j in 1..15: log[i^j] % 3
    # For vertex 0: need to handle separately
    
    # Actually, the standard Greenwood-Gleason construction for K_16:
    # Vertices 0..15. For i,j != 0: color(i,j) = log[i^j] % 3
    # For i=0, j!=0: we need a color. The Clebsch graph is self-complementary and...
    # Let me just use the known fact that this works and verify.
    
    # Build base 16-vertex coloring
    base_colors = {}
    for i in range(16):
        for j in range(i+1, 16):
            if i == 0 and j == 0:
                continue
            if i == 0:
                # vertex 0 to vertex j: use a fixed scheme
                # In the Clebsch graph construction, vertex 0 connects to all others
                # We need color(0,j) such that no mono triangle
                # Standard: color(0,j) = log[j] % 3 (or some fixed assignment)
                # Let me try: the full construction uses the fact that GF(16) \ {0} 
                # with the cubic residue coloring, and 0 is connected with a 4th "color" pattern
                # Actually for 3-coloring of K16, we can use:
                # color(0, j) = 0 for all j? No, that creates mono triangles if 0-j1-j2 all color 0.
                # 
                # Let me use a different approach: the known 3-coloring of K_16
                # from the Clebsch graph. The Clebsch graph is triangle-free and 3-colorable.
                # Its complement is also triangle-free and 3-colorable? No.
                # 
                # Actually, R(3,3,3) = 17, so K_16 can be 3-colored without mono triangle.
                # The Greenwood-Gleason construction: 
                # For GF(16), the multiplicative group has order 15. 
                # Partition {1,...,15} into 3 sets of 5 (cubic residues, non-residues cosets)
                # color(i,j) = which coset i*j^{-1} is in, for i,j in 1..15
                # For vertex 0: color(0,j) for j in 1..15
                # We need to assign these so no mono triangle.
                # A triangle involving 0 and two vertices a,b: edges (0,a), (0,b), (a,b)
                # We need these 3 edges to not all be same color.
                
                # One known assignment: color(0, j) = 0 for all j in 1..15
                # Then triangle (0,a,b): edges are color 0, color 0, color(a,b)
                # This is mono only if color(a,b) = 0, i.e., a*b^{-1} in cubic residues.
                # So this doesn't work directly.
                
                # Better: use the fact that the Clebsch graph provides the structure.
                # Let me just hardcode a valid 3-coloring of K_16.
                # I'll compute it via the standard method and verify.
                
                pass
            elif j == 0:
                continue
            else:
                base_colors[(i,j)] = gf16_log[i ^ j] % 3
    
    # For vertex 0, use a greedy/known assignment
    # color(0, j) for j=1..15: we'll determine by checking constraints
    # Actually, let me use a known valid coloring. The Greenwood-Gleason 3-coloring of K_16:
    # It's known that if we take the 15 nonzero elements and color by cubic residues,
    # and add vertex 0 with a specific pattern, it works.
    
    # Let me try: color(0, j) = (gf16_log[j] + 1) % 3
    # Check: triangle (0, a, b): colors are (log[a]+1)%3, (log[b]+1)%3, log[a^b]%3
    # a^b in GF(16): if a=2^i, b=2^j, then a^b = 2^(i+j) (since GF(16)* is cyclic)
    # Wait, a XOR b is not a*b in GF(16). i^j as integers is XOR, not field multiplication.
    # I'm confusing things. Let me restart the base construction.
    
    # Correct approach: GF(16) elements are 0..15. Multiplication is gf16_mul.
    # For i, j in 1..15: color(i,j) = gf16_log[gf16_mul(i, gf16_inv(j))] % 3
    # where gf16_inv(j) = gf16_exp[(15 - gf16_log[j]) % 15]
    
    def gf16_inv(a):
        if a == 0:
            return 0
        return gf16_exp[(15 - gf16_log[a]) % 15]
    
    # Rebuild base colors properly
    base_colors = {}
    for i in range(1, 16):
        for j in range(i+1, 16):
            ratio = gf16_mul(i, gf16_inv(j))
            base_colors[(i,j)] = gf16_log[ratio] % 3
    
    # For vertex 0: try color(0,j) = 0 and check
    # Actually, let me just use a brute-force check for vertex 0 colors
    # There are 3^15 possibilities, but we can be smarter.
    # For each j in 1..15, color(0,j) must be such that for all pairs (a,b) with a<b, a,b != 0:
    #   not (color(0,a) == color(0,b) == base_colors[(a,b)])
    
    # This is a constraint satisfaction. Let me try a simple pattern first.
    # Pattern: color(0, j) = (gf16_log[j]) % 3
    c0 = {}
    valid = True
    for j in range(1, 16):
        c0[j] = gf16_log[j] % 3
    
    # Verify
    for a in range(1, 16):
        for b in range(a+1, 16):
            if c0[a] == c0[b] == base_colors[(a,b)]:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Try other patterns
        found = False
        for offset in range(3):
            for mult in range(1, 3):
                c0 = {}
                for j in range(1, 16):
                    c0[j] = (gf16_log[j] * mult + offset) % 3
                valid = True
                for a in range(1, 16):
                    for b in range(a+1, 16):
                        if c0[a] == c0[b] == base_colors[(a,b)]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    found = True
                    break
            if found:
                break
        if not found:
            # Brute force a few random assignments
            for attempt in range(10000):
                c0 = {}
                for j in range(1, 16):
                    c0[j] = random.randint(0, 2)
                valid = True
                for a in range(1, 16):
                    for b in range(a+1, 16):
                        if c0[a] == c0[b] == base_colors[(a,b)]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    break
    
    # Build full 16-vertex coloring
    base16 = {}
    for i in range(16):
        for j in range(i+1, 16):
            if i == 0:
                base16[(i,j)] = c0[j]
            elif j == 0:
                continue
            else:
                base16[(i,j)] = base_colors[(i,j)]
    
    # --- Now extend to larger n using the "blow-up" or "add vertices" approach ---
    # Strategy: Start with K_16 (3 colors). Add vertices one by one using color 3
    # for edges to the new vertex, but we need to ensure no mono triangle in color 3.
    # Color 3 edges form a graph that must be triangle-free.
    # 
    # Better strategy: Use the known construction that gives larger colorings.
    # 
    # Actually, the best known is 50 vertices (Chung). Let me try to construct a larger one
    # by starting with the 16-vertex base and adding vertices with a 4th color class.
    #
    # Approach: 
    # 1. Start with the 3-coloring of K_16.
    # 2. Add vertices 16, 17, ... using a strategy where new vertices connect to old ones
    #    primarily with color 3, but carefully to avoid mono triangles.
    # 3. Use local search to fix violations.
    
    # Let me use a different, more powerful approach:
    # Start with a valid coloring of K_n (n=16) and try to extend to K_{n+1}
    # by adding a new vertex and assigning colors to its edges, then repairing.
    
    def get_color(colors_dict, i, j):
        if i > j:
            i, j = j, i
        return colors_dict.get((i,j))
    
    def set_color(colors_dict, i, j, c):
        if i > j:
            i, j = j, i
        colors_dict[(i,j)] = c
    
    def count_mono_triangles(colors_dict, n):
        """Count monochromatic triangles."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    c1 = get_color(colors_dict, i, j)
                    c2 = get_color(colors_dict, i, k)
                    c3 = get_color(colors_dict, j, k)
                    if c1 == c2 == c3:
                        count += 1
        return count
    
    def has_mono_triangle(colors_dict, n):
        """Check if any monochromatic triangle exists."""
        for i in range(n):
            for j in range(i+1, n):
                c_ij = get_color(colors_dict, i, j)
                for k in range(j+1, n):
                    if get_color(colors_dict, i, k) == c_ij and get_color(colors_dict, j, k) == c_ij:
                        return True, (i, j, k)
        return False, None
    
    # Start with the 16-vertex coloring
    current_colors = dict(base16)
    current_n = 16
    
    # Try to extend by adding vertices
    # When adding vertex v, assign colors to edges (v, u) for u in 0..v-1
    # Then repair any mono triangles involving v
    
    def try_extend(colors_dict, n, max_iter=5000):
        """Try to add vertex n to the coloring of K_n."""
        if time.time() - start_time > DEADLINE:
            return None
        
        new_colors = dict(colors_dict)
        new_v = n
        
        # Initial assignment: random
        for u in range(n):
            new_colors[(u, new_v)] = random.randint(0, 3)
        
        # Repair: iteratively fix mono triangles
        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Find a mono triangle
            found = False
            # Check triangles involving new_v
            for i in range(n):
                for j in range(i+1, n):
                    c_ij = new_colors[(i,j)]
                    if new_colors[(i, new_v)] == c_ij and new_colors[(j, new_v)] == c_ij:
                        # Mono triangle (i, j, new_v) with color c_ij
                        # Change one of the edges to new_v
                        # Pick the edge with most "freedom" (fewest constraints)
                        options = []
                        # Change (i, new_v)
                        for new_c in range(4):
                            if new_c != c_ij:
                                # Check if this creates a new mono triangle
                                ok = True
                                for k in range(n):
                                    if k != i and k != new_v:
                                        if new_colors[(i,k)] == new_c and new_colors[(k, new_v)] == new_c:
                                            ok = False
                                            break
                                if ok:
                                    options.append((i, new_v, new_c))
                        # Change (j, new_v)
                        for new_c in range(4):
                            if new_c != c_ij:
                                ok = True
                                for k in range(n):
                                    if k != j and k != new_v:
                                        if new_colors[(j,k)] == new_c and new_colors[(k, new_v)] == new_c:
                                            ok = False
                                            break
                                if ok:
                                    options.append((j, new_v, new_c))
                        
                        if options:
                            # Pick randomly or greedily
                            opt = random.choice(options)
                            new_colors[(opt[0], opt[1])] = opt[2]
                            found = True
                            break
                        else:
                            # No safe option, pick random change
                            if random.random() < 0.5:
                                new_colors[(i, new_v)] = random.randint(0, 3)
                            else:
                                new_colors[(j, new_v)] = random.randint(0, 3)
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # No mono triangles involving new_v
                # Check all triangles (should be fine if we didn't break old ones)
                has_mono, _ = has_mono_triangle(new_colors, n + 1)
                if not has_mono:
                    return new_colors
                else:
                    # There's a mono triangle not involving new_v (shouldn't happen if we only changed new_v edges)
                    # But just in case, break
                    return None
        
        return None
    
    # Try to extend multiple times
    for target_n in range(17, 63):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        success = False
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            new_colors = try_extend(current_colors, target_n - 1, max_iter=3000)
            if new_colors is not None:
                current_colors = new_colors
                current_n = target_n
                success = True
                break
        
        if not success:
            break
    
    # Local search improvement on the current coloring
    if time.time() - start_time < DEADLINE - 10:
        # Try to improve by flipping edges
        best_n = current_n
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Try flipping a random edge
            i = random.randint(0, best_n - 1)
            j = random.randint(0, best_n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = current_colors[(i,j)]
            new_c = (old_c + 1) % 4
            current_colors[(i,j)] = new_c
            
            # Check if this created a mono triangle
            bad = False
            for k in range(best_n):
                if k != i and k != j:
                    c_ik = get_color(current_colors, i, k)
                    c_jk = get_color(current_colors, j, k)
                    if c_ik == new_c and c_jk == new_c:
                        bad = True
                        break
            
            if bad:
                current_colors[(i,j)] = old_c
    
    # Build output
    colors_list = []
    for i in range(current_n):
        for j in range(i+1, current_n):
            colors_list.append(current_colors[(i,j)])
    
    return {'n': current_n, 'colors': colors_list}
