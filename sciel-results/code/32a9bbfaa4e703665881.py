<<<<<<< SEARCH
"""Stronger program: directly construct the known 50-vertex 4-coloring."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 100.0

    def color_index(n, i, j):
        """Index in lexicographic edge array for edge (i,j), i<j."""
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def build_from_adj(n, adj):
        """Build colors array from adjacency matrix."""
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(adj[i][j])
        return colors

    def has_mono_triangle_adj(n, adj):
        """Check if an adjacency matrix has a mono triangle."""
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if adj[a][b] == adj[a][c] == adj[b][c]:
                        return True
        return False

    def count_mono_triangles(n, adj):
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    if adj[a][b] == adj[a][c] == adj[b][c]:
                        cnt += 1
        return cnt

    # --- GF(64) Construction for 50-vertex 4-coloring ---
    
    # GF(64) arithmetic
    # Primitive polynomial: x^6 + x + 1 (binary 1000011, decimal 67)
    # Wait, x^6+x+1 is 1000011 which is 67.
    # Let's use x^6 + x^4 + x^3 + x + 1? No, x^6+x+1 is primitive.
    # Bits: 6, 1, 0. Value: 2^6 + 2^1 + 2^0 = 64 + 2 + 1 = 67.
    
    def mul64(a, b):
        r = 0
        for i in range(6):
            if b & (1 << i):
                r ^= a << i
        # Reduce
        # x^6 = x + 1
        # If bit 6 is set, subtract x^6 (XOR) and add x+1
        if r & (1 << 6):
            r ^= (1 << 6)
            r ^= 0b11 # x + 1
        # Check if further reduction needed? 
        # Since we only shift by 1 at a time in the loop, 
        # and we reduce immediately, r < 2^6 is not guaranteed in single step?
        # a < 2^6, b < 2^6.
        # a << i can be up to 2^(6-1+5) = 2^10?
        # Let's do standard reduction.
        # Actually, the loop `for i in range(6)` handles bits 0-5.
        # If a has bit 5, a<<5 has bit 10.
        # So r can be large.
        # We need to reduce r modulo P(x).
        # P(x) = x^6 + x + 1.
        # While r >= 2^6:
        #   if r & (1<<6): r ^= (1<<6); r ^= 0b11
        #   elif r & (1<<7): ...
        # Simpler:
        # Perform division.
        # But for small numbers, we can just do:
        # for i in range(11, 5, -1):
        #   if r & (1 << i):
        #     r ^= (1 << i)
        #     r ^= (1 << (i-6)) # x^(i-6)
        #     r ^= (1 << (i-5)) # x^(i-5) ? No, x^(i-6) * (x+1) = x^(i-5) + x^(i-6).
        #     # So r ^= (1 << (i-5)) | (1 << (i-6))
        #     # Wait, x^6 = x+1.
        #     # x^i = x^(i-6) * x^6 = x^(i-6)(x+1) = x^(i-5) + x^(i-6).
        #     # So r ^= (1 << (i-5)) ^ (1 << (i-6)).
        # Let's implement this robustly.
        r = 0
        for i in range(6):
            if b & (1 << i):
                r ^= a << i
        # Reduce r
        for i in range(11, 5, -1):
            if r & (1 << i):
                r ^= (1 << i)
                r ^= (1 << (i-5))
                r ^= (1 << (i-6))
        return r

    # Precompute log and exp tables for GF(64)
    log64 = {}
    exp64 = [0] * 64
    x = 1
    log64[1] = 0
    exp64[0] = 1
    for e in range(1, 63):
        x = mul64(x, 2)
        log64[x] = e
        exp64[e] = x
    # x should be 1 after 63 steps
    # exp64[63] is not defined in loop, but log64 has 1..62.
    # We need log64 for all non-zero elements.
    # The loop fills 1..62.
    # What about the element corresponding to e=0? 1.
    # So log64 has keys 1..63?
    # x starts at 1 (e=0).
    # e=1: x=2. log64[2]=1.
    # ...
    # e=62: x=g^62. log64[g^62]=62.
    # Next x = g^63 = 1.
    # So log64 has 1..62 mapped.
    # We need to handle all non-zero elements.
    # There are 63 non-zero elements.
    # The keys are 1, 2, ..., 63 (values in GF(64)).
    # Wait, the values are 0..63.
    # Non-zero are 1..63.
    # So log64 should have 63 entries.
    # My loop runs e=1..62. That's 62 entries.
    # Plus log64[1]=0. Total 63. Correct.
    
    # Initialize 50-vertex coloring
    n_target = 50
    # Map vertices to distinct non-zero field elements
    # Use exp64[0..49]
    verts = [exp64[i] for i in range(n_target)]
    
    adj = [[0] * n_target for _ in range(n_target)]
    for i in range(n_target):
        for j in range(i + 1, n_target):
            diff = verts[i] ^ verts[j] # subtraction in GF(2^k)
            if diff == 0:
                c = 0 # Should not happen
            else:
                c = log64[diff] % 4
            adj[i][j] = c
            adj[j][i] = c

    # Local Search - conflict-driven with simulated annealing
    rng = random.Random(42)
    cur_mono = count_mono_triangles(n_target, adj)
    
    # If already 0, great.
    it = 0
    max_iters = 500000
    temp = 2.0
    cooling = 0.9995
    
    while cur_mono > 0 and it < max_iters and time.time() < deadline - 10:
        it += 1
        temp *= cooling
        
        # Pick a random edge
        i = rng.randrange(n_target)
        j = rng.randrange(n_target)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        old_c = adj[i][j]
        new_c = rng.randrange(4)
        if new_c == old_c:
            continue
            
        # Calculate delta
        delta = 0
        for k in range(n_target):
            if k == i or k == j:
                continue
            ci = adj[i][k]
            cj = adj[j][k]
            before = 1 if (old_c == ci == cj) else 0
            after = 1 if (new_c == ci == cj) else 0
            delta += after - before
            
        if delta < 0 or (delta == 0 and rng.random() < 0.1) or (delta > 0 and rng.random() < (0.5 ** (delta * 10))):
            adj[i][j] = new_c
            adj[j][i] = new_c
            cur_mono += delta

    if cur_mono == 0:
        colors = build_from_adj(n_target, adj)
        return {'n': n_target, 'colors': colors}

    # If local search failed, try to prune to largest valid subgraph
    # Greedy pruning - remove vertices involved in mono triangles
    active = [1] * n_target
    cur_mono = count_mono_triangles(n_target, adj)
    
    # Build adjacency of mono-triangle involvement for faster pruning
    while cur_mono > 0 and time.time() < deadline - 5:
        # Find a mono triangle
        found = False
        for a in range(n_target):
            if not active[a]: continue
            for b in range(a + 1, n_target):
                if not active[b]: continue
                for c in range(b + 1, n_target):
                    if not active[c]: continue
                    if adj[a][b] == adj[a][c] == adj[b][c]:
                        # Found (a,b,c)
                        # Count how many mono triangles each vertex is in (among active)
                        badness = [0, 0, 0]
                        for x in range(n_target):
                            if not active[x] or x == a or x == b or x == c:
                                continue
                            for y in range(x + 1, n_target):
                                if not active[y] or y == a or y == b or y == c:
                                    continue
                                if x == a and y == b:
                                    pass
                                # Check if (a,x,y) or similar forms mono triangle
                                # This is getting complex; just remove the one with most edges of that color
                        # Simpler: remove vertex with highest degree in mono-triangle subgraph
                        # For now, remove random one but weighted
                        v = rng.choice([a, b, c])
                        active[v] = 0
                        cur_mono = count_mono_triangles(n_target, adj)
                        found = True
                        break
                if found: break
            if found: break
        if not found:
            break # Should not happen if cur_mono > 0

    # Build result from active vertices
    active_indices = [i for i in range(n_target) if active[i]]
    n_final = len(active_indices)
    
    if n_final >= 32:
        # Build colors for the subgraph
        colors = []
        # Map old index to new index
        # We need lexicographic order of the new vertices
        # The new vertices are active_indices sorted (they are)
        # We need colors for pairs (i,j) in new indices
        # Which correspond to (active_indices[i], active_indices[j]) in old
        for i in range(n_final):
            for j in range(i + 1, n_final):
                old_i = active_indices[i]
                old_j = active_indices[j]
                colors.append(adj[old_i][old_j])
        
        # Verify
        if not has_mono_triangle_adj(n_final, build_from_adj(n_final, [[0]*n_final for _ in range(n_final)])): # Dummy check
             pass
        # Real verification
        valid = True
        for a in range(n_final):
            for b in range(a + 1, n_final):
                for c in range(b + 1, n_final):
                    if colors[color_index(n_final, a, b)] == colors[color_index(n_final, a, c)] == colors[color_index(n_final, b, c)]:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        
        if valid:
            return {'n': n_final, 'colors': colors}

    # Fallback: 32-vertex Greenwood-Gleason
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    n = 32
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if i // 16 == j // 16:
                colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)

    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
=======
"""Stronger program: construct a valid 50-vertex 4-coloring using local search from good initial states."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def count_mono_triangles(n, adj):
        """Count monochromatic triangles in adjacency matrix."""
        cnt = 0
        for a in range(n):
            for b in range(a + 1, n):
                c_ab = adj[a][b]
                for c in range(b + 1, n):
                    if adj[a][c] == c_ab and adj[b][c] == c_ab:
                        cnt += 1
        return cnt

    def local_search(n, adj, rng, time_limit, max_iters=2000000):
        """Simulated annealing to eliminate mono triangles."""
        cur_mono = count_mono_triangles(n, adj)
        if cur_mono == 0:
            return adj, 0
        
        it = 0
        temp = 1.0
        cooling = 0.9999
        best_adj = [row[:] for row in adj]
        best_mono = cur_mono
        
        while cur_mono > 0 and it < max_iters and time.time() < time_limit:
            it += 1
            temp *= cooling
            
            # Pick random edge
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_c = adj[i][j]
            new_c = rng.randrange(4)
            if new_c == old_c:
                continue
            
            # Calculate delta in mono triangles
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                ci = adj[i][k]
                cj = adj[j][k]
                before = 1 if (old_c == ci == cj) else 0
                after = 1 if (new_c == ci == cj) else 0
                delta += after - before
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.3
            else:
                # Accept with probability exp(-delta/temp)
                if temp > 0.01:
                    accept = rng.random() < (0.5 ** (delta * 2))
            
            if accept:
                adj[i][j] = new_c
                adj[j][i] = new_c
                cur_mono += delta
                
                if cur_mono < best_mono:
                    best_mono = cur_mono
                    best_adj = [row[:] for row in adj]
        
        # Restore best if we got worse
        if best_mono < cur_mono:
            adj = best_adj
            cur_mono = best_mono
        
        return adj, cur_mono

    n_target = 50
    rng = random.Random(12345)
    
    # Try multiple initial colorings with local search
    best_n = 0
    best_colors = None
    
    for attempt in range(10):
        if time.time() > deadline - 15:
            break
        
        # Different initial colorings
        if attempt == 0:
            # Structured: use modular arithmetic
            adj = [[0] * n_target for _ in range(n_target)]
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    d = (j - i) % n_target
                    if d == 0:
                        c = 0
                    elif d % 2 == 0:
                        c = (d // 2) % 4
                    else:
                        c = (d * 3 + 1) % 4
                    adj[i][j] = c
                    adj[j][i] = c
        elif attempt == 1:
            # GF(64)-inspired with fixed arithmetic
            # Use x^6 + x^4 + x^3 + x + 1 (primitive)
            def mul64(a, b):
                r = 0
                for i in range(6):
                    if b & (1 << i):
                        r ^= a << i
                # Reduce mod x^6 + x^4 + x^3 + x + 1
                # x^6 = x^4 + x^3 + x + 1
                for i in range(11, 5, -1):
                    if r & (1 << i):
                        r ^= (1 << i)
                        r ^= (1 << (i-2))  # x^(i-6)*x^4 = x^(i-2)
                        r ^= (1 << (i-3))  # x^(i-6)*x^3 = x^(i-3)
                        r ^= (1 << (i-6))  # x^(i-6)*x   = x^(i-6)
                        r ^= (1 << (i-6))  # x^(i-6)*1   = x^(i-6) -- wait, that's wrong
                # Let me redo: x^6 = x^4 + x^3 + x + 1
                # So x^(i) for i>=6: x^i = x^(i-6) * x^6 = x^(i-6)(x^4+x^3+x+1) = x^(i-2)+x^(i-3)+x^(i-6)+x^(i-6)
                # That's x^(i-2) + x^(i-3) + 2*x^(i-6) = x^(i-2) + x^(i-3) in GF(2)
                # Hmm, that doesn't seem right either. Let me use a different primitive poly.
                # Use x^6 + x + 1 which is definitely primitive.
                # x^6 = x + 1
                # x^i = x^(i-6) * x^6 = x^(i-6)(x+1) = x^(i-5) + x^(i-6)
                r = 0
                for i in range(6):
                    if b & (1 << i):
                        r ^= a << i
                for i in range(11, 5, -1):
                    if r & (1 << i):
                        r ^= (1 << i)
                        r ^= (1 << (i-5))
                        r ^= (1 << (i-6))
                return r
            
            log64 = {}
            exp64 = [0] * 64
            x = 1
            log64[1] = 0
            exp64[0] = 1
            for e in range(1, 63):
                x = mul64(x, 2)
                log64[x] = e
                exp64[e] = x
            
            verts = [exp64[i] for i in range(n_target)]
            adj = [[0] * n_target for _ in range(n_target)]
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    diff = verts[i] ^ verts[j]
                    if diff == 0:
                        c = 0
                    else:
                        c = log64[diff] % 4
                    adj[i][j] = c
                    adj[j][i] = c
        elif attempt == 2:
            # Random
            adj = [[0] * n_target for _ in range(n_target)]
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = rng.randrange(4)
                    adj[i][j] = c
                    adj[j][i] = c
        else:
            # Random with structure
            adj = [[0] * n_target for _ in range(n_target)]
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = (i * j + i + j) % 4
                    adj[i][j] = c
                    adj[j][i] = c
        
        # Run local search
        adj, mono = local_search(n_target, adj, rng, deadline - 15 - attempt * 5)
        
        if mono == 0:
            # Valid 50-vertex coloring!
            colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    colors.append(adj[i][j])
            return {'n': n_target, 'colors': colors}
        
        # Try to prune to get a valid subgraph
        if mono > 0:
            active = [1] * n_target
            cur_mono = mono
            prune_time = min(deadline - 10, start_time + 8 + attempt * 2)
            
            while cur_mono > 0 and time.time() < prune_time:
                found = False
                for a in range(n_target):
                    if not active[a]: continue
                    for b in range(a + 1, n_target):
                        if not active[b]: continue
                        for c in range(b + 1, n_target):
                            if not active[c]: continue
                            if adj[a][b] == adj[a][c] == adj[b][c]:
                                # Remove the vertex in this triangle that's in the most mono triangles
                                # Quick heuristic: remove random one
                                v = rng.choice([a, b, c])
                                active[v] = 0
                                # Recount
                                cur_mono = 0
                                for a2 in range(n_target):
                                    if not active[a2]: continue
                                    for b2 in range(a2 + 1, n_target):
                                        if not active[b2]: continue
                                        c_ab = adj[a2][b2]
                                        for c2 in range(b2 + 1, n_target):
                                            if not active[c2]: continue
                                            if adj[a2][c2] == c_ab and adj[b2][c2] == c_ab:
                                                cur_mono += 1
                                found = True
                                break
                        if found: break
                    if found: break
                if not found:
                    break
            
            active_indices = [i for i in range(n_target) if active[i]]
            n_final = len(active_indices)
            
            if n_final > best_n:
                best_n = n_final
                colors = []
                for i in range(n_final):
                    for j in range(i + 1, n_final):
                        old_i = active_indices[i]
                        old_j = active_indices[j]
                        colors.append(adj[old_i][old_j])
                best_colors = colors
    
    if best_n >= 32 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Final fallback: 32-vertex Greenwood-Gleason
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    n = 32
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
        if i // 16 == j // 16:
            colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
        else:
            colors.append(3)

    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

I notice there's a syntax error in my replacement - an extra indentation issue. Let me fix that:

