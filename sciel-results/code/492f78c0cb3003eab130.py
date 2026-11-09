import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Helper to count set bits in an integer (Python 3.10+ has int.bit_count())
    def popcount(x):
        return x.bit_count()

    def generate_seed_coloring(n):
        # Use F_16 construction for the first 48 vertices (3 blocks of 16)
        # F_16 arithmetic
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        
        colors = [0] * (n * (n - 1) // 2)
        # Map vertex index to block and local index
        # Blocks 0, 1, 2 cover vertices 0-47
        # Vertices 48, 49 are "extra"
        
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
                # This formula is tricky, let's just use a simple counter
                pass
                
        # Simpler: just fill a list
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                block_i = i // 16
                block_j = j // 16
                local_i = i % 16
                local_j = j % 16
                
                if block_i == block_j and block_i < 3:
                    # Inside a block, use F_16 coloring
                    c = log[local_i ^ local_j] % 3
                elif block_i < 3 and block_j < 3:
                    # Between blocks 0,1,2
                    # Use a simple deterministic pattern to avoid triangles
                    # Color = (block_i + block_j) % 3 ? 
                    # Let's try: if block_i < block_j, color = (block_i * 2 + block_j) % 3
                    c = (block_i * 2 + block_j) % 3
                else:
                    # Involves extra vertices 48, 49 or between extra
                    # Random
                    c = random.randint(0, 3)
                
                colors.append(c)
        return colors

    def colors_to_bitsets(colors, n):
        # bitsets[v][c] is an integer where bit u is 1 if color(v,u) == c
        bitsets = [[0]*4 for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                bitsets[i][c] |= (1 << j)
                bitsets[j][c] |= (1 << i)
                idx += 1
        return bitsets

    def count_triangles(bitsets, n):
        total = 0
        for c in range(4):
            # For each edge (u,v) of color c, count common neighbors of color c
            # We can just iterate all pairs
            # This is O(N^2) per color
            for u in range(n):
                # Only count v > u to avoid double counting?
                # The triangle (u,v,w) is counted 3 times in sum over edges?
                # No, if we sum |N_c(u) & N_c(v)| for all u<v, 
                # each triangle (u,v,w) is counted 3 times (once for each edge).
                # So total triangles = sum / 3.
                mask_u = bitsets[u][c]
                # Iterate v > u
                # We can just iterate all v != u and divide by 6?
                # Let's stick to u < v
                v_mask = mask_u & ~(1 << u)
                # We need to sum popcount(mask_u & bitsets[v][c]) for v > u
                # But bitsets[v][c] includes u.
                # popcount(mask_u & bitsets[v][c]) counts w such that w~u and w~v.
                # This includes w=v? No, v not in N_c(v).
                # This includes w=u? No, u not in N_c(u).
                # So it counts exactly the triangles containing edge (u,v).
                # Summing over all u < v counts each triangle 3 times.
                # So divide by 3.
                
                # Optimization: iterate v > u
                # We can extract bits > u
                # But simple loop is fine for N=50
                s = 0
                for v in range(u + 1, n):
                    if (mask_u >> v) & 1: # If u-v is color c
                        s += popcount(mask_u & bitsets[v][c])
                total += s
        return total // 3

    def local_search(bitsets, n, max_iters=20000):
        current_tris = count_triangles(bitsets, n)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            if current_tris == 0:
                return bitsets, True
            
            # Pick random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old_c = -1
            for c in range(4):
                if (bitsets[u][c] >> v) & 1:
                    old_c = c
                    break
            
            if old_c == -1:
                continue # Should not happen
            
            best_gain = 0
            best_c = -1
            
            mask_u = bitsets[u]
            mask_v = bitsets[v]
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                # Gain = (triangles removed) - (triangles added)
                # Removed: triangles of color old_c involving (u,v)
                # Added: triangles of color new_c involving (u,v)
                
                # Note: When we change (u,v) to new_c, 
                # the neighbors of u in new_c and neighbors of v in new_c 
                # are used to count added triangles.
                # But the edge (u,v) itself is not a neighbor.
                # The sets bitsets[u][new_c] and bitsets[v][new_c] 
                # do NOT include each other if the edge is not that color.
                # If new_c != old_c, then v is NOT in bitsets[u][new_c] and u is NOT in bitsets[v][new_c].
                # So the intersection is clean.
                
                added = popcount(mask_u[new_c] & mask_v[new_c])
                removed = popcount(mask_u[old_c] & mask_v[old_c])
                
                gain = removed - added
                if gain > best_gain:
                    best_gain = gain
                    best_c = new_c
            
            if best_c != -1:
                # Update bitsets
                bitsets[u][old_c] &= ~(1 << v)
                bitsets[v][old_c] &= ~(1 << u)
                bitsets[u][best_c] |= (1 << v)
                bitsets[v][best_c] |= (1 << u)
                current_tris -= best_gain
            else:
                # Random kick
                new_c = random.randint(0, 3)
                if new_c != old_c:
                    bitsets[u][old_c] &= ~(1 << v)
                    bitsets[v][old_c] &= ~(1 << u)
                    bitsets[u][new_c] |= (1 << v)
                    bitsets[v][new_c] |= (1 << u)
                    # Recalculate triangles? Too slow.
                    # Just update current_tris approximately or leave it?
                    # If we leave it, it might be wrong.
                    # Let's just re-count every 100 kicks?
                    # Or just accept the risk and check at the end?
                    # For now, let's just update the matrix and re-count if we get stuck.
                    # Actually, let's just re-count if no move found for a while?
                    # Simplest: Re-count on kick.
                    current_tris = count_triangles(bitsets, n)

        return bitsets, (current_tris == 0)

    def bitsets_to_colors(bitsets, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if (bitsets[i][c] >> j) & 1:
                        colors.append(c)
                        break
        return colors

    best_n = 0
    best_colors = []
    
    # Try to construct n=50
    target_n = 50
    if time.time() < deadline - 5:
        seed_colors = generate_seed_coloring(target_n)
        bitsets = colors_to_bitsets(seed_colors, target_n)
        bitsets, success = local_search(bitsets, target_n, max_iters=10000)
        if success:
            best_n = target_n
            best_colors = bitsets_to_colors(bitsets, target_n)
    
    # If we succeeded with 50, try 51
    if best_n == 50 and time.time() < deadline - 5:
        target_n = 51
        # Extend bitsets
        new_bitsets = [[0]*4 for _ in range(target_n)]
        for i in range(50):
            for c in range(4):
                new_bitsets[i][c] = bitsets[i][c]
        
        # Add vertex 50
        # Connect to 0..49 randomly
        for i in range(50):
            c = random.randint(0, 3)
            new_bitsets[50][c] |= (1 << i)
            new_bitsets[i][c] |= (1 << 50)
            
        bitsets = new_bitsets
        bitsets, success = local_search(bitsets, target_n, max_iters=5000)
        if success:
            best_n = target_n
            best_colors = bitsets_to_colors(bitsets, target_n)
            
    # If we didn't get 50, maybe try to salvage a smaller n?
    # The seed program returned 32. We should at least return 32 if we fail.
    if best_n == 0:
        # Fallback to seed logic or just return a small valid one
        # Let's just return the n=32 logic from the original if this fails
        # But for now, if we fail 50, we return 0? No, that's bad.
        # We should ensure we return at least the base seed.
        
        # Re-generate the n=32 seed from the original code
        def generate_seed_32():
            def mul(a, b):
                r = 0
                for i in range(4):
                    if b >> i & 1:
                        r ^= a << i
                for i in range(7, 3, -1):
                    if r >> i & 1:
                        r ^= 0b10011 << (i - 4)
                return r

            log, x = {}, 1
            for e in range(15):
                log[x] = e
                x = mul(x, 2)
            
            colors = []
            for i in range(32):
                for j in range(i + 1, 32):
                    if i // 16 == j // 16:
                        colors.append(log[(i % 16) ^ (j % 16)] % 3)
                    else:
                        colors.append(3)
            return colors
        
        best_n = 32
        best_colors = generate_seed_32()

    return {'n': best_n, 'colors': best_colors}
