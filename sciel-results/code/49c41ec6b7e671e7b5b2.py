import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def get_seed_32():
        """32-vertex coloring from GF(16) construction."""
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
        return {'n': 32, 'colors': colors}
    
    def get_seed_50():
        """
        Attempt to construct a 50-vertex 4-coloring.
        Uses a known construction based on finite geometry / difference sets.
        
        The 50-vertex construction can be built using the following approach:
        - Use 5 groups of 10 vertices
        - Within each group, use a 10-vertex 3-coloring (no mono triangles)
        - Between groups, use a systematic 4-coloring
        """
        # Known: The 50-vertex 4-coloring exists (Chung, 1996)
        # Let me try a construction based on Z_5 x Z_10 or similar
        
        n = 50
        M = np.zeros((n, n), dtype=np.int8)
        
        # Partition into 5 groups of 10
        # Within each group of 10, use a 3-coloring of K_10 with no mono triangles
        # K_10 can be 3-colored without mono triangles (R(3,3,3) = 17, so 10 is fine)
        
        # 10-vertex 3-coloring: use the known construction
        # Vertices 0..9, color by (i*j mod 7) % 3 for i!=j? Let me use a simpler approach.
        # Actually, let me use: for vertices a,b in same group, color = (a-b)^2 mod 7 mapped to {0,1,2}
        # Squares mod 7: 0,1,2,4. Non-zero squares: 1,2,4. Map 1->0, 2->1, 4->2
        
        def color_10(a, b):
            """Color edge (a,b) in a 10-vertex set with 3 colors, no mono triangles."""
            d = (a - b) % 10
            if d == 0:
                return 0
            # Use a known 10-vertex 3-coloring
            # One approach: cyclic construction
            # For K_10, we can use: color = (d^2 mod 7) mapped appropriately
            # But 10 is not prime. Let me use a different approach.
            
            # Simple: use the fact that R(3,3,3)=17, so K_10 is 3-colorable without mono triangles
            # Use a greedy or known pattern
            # Pattern: partition 10 vertices into 3 sets of sizes 4,3,3
            # Within each set: one color. Between sets: use 2 remaining colors carefully.
            
            # Actually, let me just use a random 3-coloring and verify
            # For now, use a deterministic pattern:
            # Vertices 0-3: color 0 internally
            # Vertices 4-6: color 1 internally  
            # Vertices 7-9: color 2 internally
            # Between groups: use a 2-coloring pattern
            
            a_set = 0 if a < 4 else (1 if a < 7 else 2)
            b_set = 0 if b < 4 else (1 if b < 7 else 2)
            
            if a_set == b_set:
                return a_set  # 0, 1, or 2
            else:
                # Between different sets, use colors from the remaining 2
                # This is getting complex. Let me use a simpler global approach.
                return (a + b) % 3
        
        # This approach is getting complicated. Let me try a different strategy:
        # Use simulated annealing starting from a good random initialization
        # with a strong bias toward the 32-vertex structure.
        
        return None  # We'll use SA instead
    
    def local_search(M, n, time_limit):
        """
        Local search to eliminate monochromatic triangles.
        Returns True if no mono triangles remain.
        """
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        def count_mono_triangles():
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = M[i, j]
                    if c < 0 or c > 3:
                        continue
                    for k in range(j + 1, n):
                        if M[i, k] == c and M[j, k] == c:
                            count += 1
            return count
        
        def find_one_mono_triangle():
            """Find one monochromatic triangle if exists."""
            for i in range(n):
                for j in range(i + 1, n):
                    c = M[i, j]
                    if c < 0 or c > 3:
                        continue
                    for k in range(j + 1, n):
                        if M[i, k] == c and M[j, k] == c:
                            return (i, j, k)
            return None
        
        current_count = count_mono_triangles()
        
        if current_count == 0:
            return True
        
        T = 5.0
        cooling = 0.9995
        min_T = 0.01
        no_improve = 0
        best_count = current_count
        
        while time.time() < time_limit:
            if current_count == 0:
                return True
            
            # Pick a random edge
            idx = random.randint(0, len(edges) - 1)
            i, j = edges[idx]
            old_c = M[i, j]
            
            if old_c < 0 or old_c > 3:
                continue
            
            # Calculate delta for changing (i,j) to new_c
            # Count k where M[i,k] == M[j,k] == c for each c
            matches = (M[i, :] == M[j, :])
            matches[i] = False
            matches[j] = False
            
            counts = np.zeros(4, dtype=np.int32)
            for k in range(n):
                if matches[k]:
                    c = M[i, k]
                    if 0 <= c <= 3:
                        counts[c] += 1
            
            best_delta = 0
            best_new_c = old_c
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                delta = counts[new_c] - counts[old_c]
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_delta <= 0:
                accept = True
            else:
                if T > min_T:
                    accept = random.random() < np.exp(-best_delta / T)
                else:
                    accept = False
            
            if accept and best_new_c != old_c:
                M[i, j] = best_new_c
                M[j, i] = best_new_c
                current_count += best_delta
                
                if current_count < best_count:
                    best_count = current_count
                    no_improve = 0
                else:
                    no_improve += 1
                
                if current_count == 0:
                    return True
            else:
                no_improve += 1
            
            T *= cooling
            
            # Restart if stuck too long
            if no_improve > 5000:
                # Try a random restart with perturbation
                for _ in range(10):
                    idx = random.randint(0, len(edges) - 1)
                    i, j = edges[idx]
                    M[i, j] = random.randint(0, 3)
                    M[j, i] = M[i, j]
                current_count = count_mono_triangles()
                no_improve = 0
                T = 5.0
        
        return current_count == 0
    
    def try_find_coloring(n, time_limit):
        """Try to find a valid 4-coloring of K_n."""
        if n < 2:
            return [0] * (n * (n - 1) // 2)
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Strategy: Start from the 32-vertex seed if n >= 32, extend it
        # Otherwise, start random
        
        best_result = None
        best_count = float('inf')
        
        # Multiple restarts
        max_restarts = 20
        restart_time = time_limit / max_restarts
        
        for r in range(max_restarts):
            if time.time() > time_limit:
                break
            
            M = np.zeros((n, n), dtype=np.int8)
            
            if n >= 32:
                # Initialize with 32-vertex structure in first 32 vertices
                seed = get_seed_32()
                idx = 0
                for i in range(32):
                    for j in range(i + 1, 32):
                        M[i, j] = M[j, i] = seed['colors'][idx]
                        idx += 1
                
                # Extend to n vertices: for vertices 32..n-1, assign random colors
                for i in range(32, n):
                    for j in range(i + 1, n):
                        M[i, j] = M[j, i] = random.randint(0, 3)
                    for j in range(32):
                        M[i, j] = M[j, i] = random.randint(0, 3)
            else:
                for i, j in edges:
                    M[i, j] = M[j, i] = random.randint(0, 3)
            
            # Run local search
            success = local_search(M, n, time.time() + restart_time * 0.9)
            
            if success:
                # Verify
                count = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = M[i, j]
                        for k in range(j + 1, n):
                            if M[i, k] == c and M[j, k] == c:
                                count += 1
                if count == 0:
                    res = []
                    for i, j in edges:
                        res.append(int(M[i, j]))
                    return res
        
        return None
    
    # Try n from 50 down to 33
    for n in range(50, 32, -1):
        time_left = deadline - time.time()
        if time_left < 3.0:
            break
        
        # Allocate time based on n
        if n >= 48:
            t_limit = time.time() + min(time_left * 0.6, 30.0)
        elif n >= 45:
            t_limit = time.time() + min(time_left * 0.4, 20.0)
        else:
            t_limit = time.time() + min(time_left * 0.3, 10.0)
        
        res = try_find_coloring(n, t_limit)
        
        if res is not None:
            # Final verification
            colors = res
            n_val = n
            edge_count = n_val * (n_val - 1) // 2
            if len(colors) != edge_count:
                continue
            if any(c not in (0, 1, 2, 3) for c in colors):
                continue
            
            # Check for mono triangles
            M = np.zeros((n_val, n_val), dtype=np.int8)
            idx = 0
            for i in range(n_val):
                for j in range(i + 1, n_val):
                    M[i, j] = M[j, i] = colors[idx]
                    idx += 1
            
            has_mono = False
            for i in range(n_val):
                for j in range(i + 1, n_val):
                    c = M[i, j]
                    for k in range(j + 1, n_val):
                        if M[i, k] == c and M[j, k] == c:
                            has_mono = True
                            break
                    if has_mono:
                        break
                if has_mono:
                    break
            
            if not has_mono:
                return {'n': n_val, 'colors': colors}
    
    # Fallback to 32-vertex seed
    return get_seed_32()

if __name__ == "__main__":
    res = construct()
    print(res['n'])
