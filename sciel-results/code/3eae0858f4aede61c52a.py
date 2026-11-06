import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Helper to get initial n=32 coloring from seed logic
    def get_seed_32():
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
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    # Initial state
    n = 32
    colors = get_seed_32()
    
    # Function to count monochromatic triangles
    # We maintain the count incrementally during search, but need initial count
    # For n=32 seed, it is 0.
    mono_tri_count = 0
    
    # Helper to get index of edge (i, j) in the flat list
    # edges are stored in lexicographic order (i, j) with i < j
    # index = i * (2*n - i - 1) // 2 + (j - i - 1)  -- this is for fixed n
    # Since n changes, it's easier to use a 2D array or a dictionary for lookup, 
    # or just recompute indices if n is small.
    # For n up to 50, a 2D list is fast and simple.
    
    # Let's switch representation to a 2D list for easier manipulation during growth
    # colors_2d[i][j] = color
    colors_2d = [[0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            colors_2d[i][j] = colors[idx]
            colors_2d[j][i] = colors[idx]
            idx += 1
            
    def count_mono_triangles(n, c2d):
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = c2d[i][j]
                for k in range(j + 1, n):
                    if c2d[i][k] == c and c2d[j][k] == c:
                        count += 1
        return count

    # Verify initial state (should be 0)
    # mono_tri_count = count_mono_triangles(n, colors_2d)
    # assert mono_tri_count == 0, f"Seed n={n} has {mono_tri_count} mono triangles"

    best_n = n
    best_colors = colors[:]
    
    # Try to extend n from 33 to 50
    for target_n in range(33, 51):
        if time.time() > deadline - 10: # Reserve 10s for finalization
            break
            
        # Attempt to extend
        success = False
        # Try multiple random initializations for the new vertex edges
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
                
            # Create new state
            new_n = target_n
            # Copy old 2d array and expand
            new_c2d = [row[:] for row in colors_2d]
            # Add new row/col
            new_c2d.append([0] * new_n)
            for i in range(new_n - 1):
                new_c2d[i].append(0)
                
            # Assign random colors to new edges (new_n-1, i)
            for i in range(new_n - 1):
                c = random.randint(0, 3)
                new_c2d[i][new_n - 1] = c
                new_c2d[new_n - 1][i] = c
                
            # Calculate initial energy for new state
            # Only triangles involving the new vertex are new.
            # Triangles (new, i, j) where i, j < new_n-1.
            # Energy = count of (i, j) such that c2d[i][j] == c2d[i][new] == c2d[j][new]
            energy = 0
            new_v = new_n - 1
            for i in range(new_v):
                ci = new_c2d[i][new_v]
                for j in range(i + 1, new_v):
                    if new_c2d[i][j] == ci and new_c2d[j][new_v] == ci:
                        energy += 1
                        
            # Run Simulated Annealing
            # We want to reduce energy to 0
            T = 1.0
            T_min = 0.01
            cooling_rate = 0.999
            max_steps = 50000 # Adjust based on time
            
            current_energy = energy
            current_c2d = new_c2d
            best_energy = current_energy
            best_c2d = [row[:] for row in current_c2d]
            
            # Precompute neighbors for faster triangle updates?
            # For a move on edge (u, v), affected triangles are (u, v, w) for w != u, v.
            
            steps = 0
            while steps < max_steps and T > T_min:
                if time.time() > deadline - 10:
                    break
                    
                # Pick random edge
                u = random.randint(0, new_n - 1)
                v = random.randint(0, new_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                    
                old_color = current_c2d[u][v]
                new_color = random.randint(0, 3)
                if new_color == old_color:
                    continue
                
                # Calculate delta energy
                # Triangles affected: (u, v, w) for w in [0, new_n) \ {u, v}
                delta = 0
                for w in range(new_n):
                    if w == u or w == v:
                        continue
                    
                    c_uv = old_color
                    c_uw = current_c2d[u][w]
                    c_vw = current_c2d[v][w]
                    
                    # Check if (u, v, w) was mono
                    was_mono = (c_uv == c_uw and c_uv == c_vw)
                    
                    # Check if (u, v, w) becomes mono with new_color
                    is_mono = (new_color == c_uw and new_color == c_vw)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                    
                # Acceptance
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_c2d[u][v] = new_color
                    current_c2d[v][u] = new_color
                    current_energy += delta
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_c2d = [row[:] for row in current_c2d]
                        
                    # If we reached 0, we can stop early for this attempt
                    if best_energy == 0:
                        break
                        
                T *= cooling_rate
                steps += 1
                
            if best_energy == 0:
                # Success! Update state
                n = new_n
                colors_2d = best_c2d
                success = True
                break
                
        if success:
            # Flatten colors_2d to list for next iteration or return
            # We keep colors_2d for next extension
            best_n = n
            # Update best_colors flat list? Not needed yet, only for return
        else:
            # Failed to extend to target_n
            # We can try to optimize the current n a bit more? 
            # Or just break. Since we started with valid n=32, and failed to add 33, 
            # we return n=32.
            # However, maybe we can try a different random seed for the extension?
            # The loop above tries 50 attempts. If all fail, we break.
            break

    # Convert final colors_2d to flat list
    final_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            final_colors.append(colors_2d[i][j])
            
    return {'n': n, 'colors': final_colors}

# EVOLVE-BLOCK-END
