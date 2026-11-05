import time
import random
import numpy as np

DEADLINE = 145.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Fallback: Known 32-vertex solution
    def get_base_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log_table, x = {}, 1
        for e in range(15):
            log_table[x] = e
            x = mul(x, 2)

        base_colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    base_colors.append(log_table[(i % 16) ^ (j % 16)] % 3)
                else:
                    base_colors.append(3)
        return 32, base_colors

    best_n, best_colors = get_base_32()
    
    def find_coloring(N, time_budget):
        """
        Finds a 4-coloring of K_N with no monochromatic triangles.
        Uses vectorized local search.
        """
        if N <= best_n:
            return None, None
            
        # M[c] is a NxN matrix of 0/1 for color c
        M = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        
        # Initialize randomly
        for i in range(N):
            for j in range(i + 1, N):
                c = random.randint(0, 3)
                M[c][i, j] = 1
                M[c][j, i] = 1
                
        # Calculate initial energy (number of mono triangles)
        def calc_total_energy():
            E = 0
            for c in range(4):
                # Triangles in color c: 1/3 * Tr(M_c^3)
                # Or simply: for each i<j<k, check if all 3 edges are c.
                # Vectorized: For each c, sum over i<j of dot(M_c[i], M_c[j]) * (i<j mask? No, just count unique triangles)
                # E_c = 0.5 * sum_{i!=j} (M_c @ M_c)[i,j] ? No.
                # Number of triangles = 1/3 * sum_{i,j} (M_c^3)[i,j] ? No, 1/6 * sum_{i,j,k} M[i,j]M[j,k]M[k,i].
                # Since M is symmetric, Tr(M^3) = 6 * (# triangles).
                # So E = sum_c (1/6) * np.trace(M[c] @ M[c] @ M[c])
                # This is O(N^3) but highly optimized in BLAS.
                # For N=50, 50^3 = 125k ops. Very fast.
                pass
            
            # Let's use a simpler O(N^3) loop for initialization check, but we only need it once.
            # Actually, we can just run the search and check energy at the end.
            # But we need to know if we are at 0.
            # Let's compute E efficiently.
            E = 0
            for c in range(4):
                # Count triangles in M[c]
                # For each pair (i,j) with i<j, count common neighbors k>j? 
                # Simpler: 
                # For each i, for each j>i, if M[c][i,j]==1:
                #   count k>j such that M[c][j,k]==1 and M[c][k,i]==1
                # This is O(N^3) in Python loops, which is slow.
                
                # Use matrix multiplication:
                # A = M[c]
                # Triangles = np.trace(A @ A @ A) / 6.0
                # A @ A is O(N^3) in C, very fast.
                E += np.trace(M[c] @ M[c] @ M[c]) / 6.0
            return int(E)

        E = calc_total_energy()
        
        if E == 0:
            return M, E

        # Local Search
        # State: M, E
        # Move: Change edge (u,v) from c_old to c_new
        # Delta E:
        #   Removed: Triangles in c_old using (u,v).
        #            These are triangles (u,v,w) where M[c_old][u,w]==1 and M[c_old][v,w]==1.
        #            Count = dot(M[c_old][u], M[c_old][v])
        #            Note: This counts w where w!=u,v. Since M[u,u]=0, it's correct.
        #   Added: Triangles in c_new using (u,v).
        #          Count = dot(M[c_new][u], M[c_new][v])
        
        T = 1.0
        T_min = 0.01
        alpha = 0.9999
        steps_since_improve = 0
        max_steps_since_improve = 5000
        
        while time.time() < start_time + time_budget:
            if E == 0:
                return M, E
            
            # Pick a random edge involved in a conflict?
            # Or just pick a random edge.
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            c_old = -1
            for c in range(4):
                if M[c][u, v] == 1:
                    c_old = c
                    break
            
            if c_old == -1:
                continue # Should not happen
            
            # Evaluate all 3 other colors
            best_delta = float('inf')
            best_c_new = c_old
            
            # Pre-fetch rows for speed
            row_u = [M[c][u] for c in range(4)]
            row_v = [M[c][v] for c in range(4)]
            
            removed_tri = np.dot(row_u[c_old], row_v[c_old])
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                added_tri = np.dot(row_u[c_new], row_v[c_new])
                delta = added_tri - removed_tri
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
            
            # Acceptance
            if best_delta < 0:
                accept = True
            else:
                if T > T_min:
                    accept = random.random() < np.exp(-best_delta / T)
                else:
                    accept = False
            
            if accept:
                # Apply change
                M[c_old][u, v] = 0
                M[c_old][v, u] = 0
                M[best_c_new][u, v] = 1
                M[best_c_new][v, u] = 1
                E += best_delta
                if E < 0: E = 0 # Safety
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                # If stuck, restart?
                if steps_since_improve > max_steps_since_improve:
                    break
            
            T *= alpha

        # If we didn't find 0, return None
        if E == 0:
            return M, E
        return None, E

    # Try N=51 first, then 50
    # Time budget allocation
    time_left = deadline - time.time()
    
    # Attempt 51
    if time_left > 10:
        budget_51 = min(30, time_left - 10)
        M, E = find_coloring(51, budget_51)
        if M is not None and E == 0:
            # Convert to list
            colors = []
            for i in range(51):
                for j in range(i + 1, 51):
                    for c in range(4):
                        if M[c][i, j] == 1:
                            colors.append(c)
                            break
            best_n = 51
            best_colors = colors
            return {'n': best_n, 'colors': best_colors}
            
    # Attempt 50
    time_left = deadline - time.time()
    if time_left > 5:
        budget_50 = time_left - 2
        # We might need multiple restarts for 50 if the first one fails
        # find_coloring does one run. Let's loop it.
        
        while time.time() < deadline - 2:
            M, E = find_coloring(50, 5.0) # 5 sec per attempt
            if M is not None and E == 0:
                colors = []
                for i in range(50):
                    for j in range(i + 1, 50):
                        for c in range(4):
                            if M[c][i, j] == 1:
                                colors.append(c)
                                break
                best_n = 50
                best_colors = colors
                return {'n': best_n, 'colors': best_colors}
                
    return {'n': best_n, 'colors': best_colors}
