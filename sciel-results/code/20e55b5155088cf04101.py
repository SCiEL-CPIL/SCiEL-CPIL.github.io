import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    def check_time():
        return time.time() - start_time < deadline

    # Fallback: Seed N=32 construction
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
        colors_seed = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors_seed}

    def run_numPy_sa(N, time_limit):
        num_edges = N * (N - 1) // 2
        # Initialize 4 matrices
        A = [np.zeros((N, N), dtype=np.int8) for _ in range(4)]
        
        # Random initialization
        # To ensure symmetry, we pick random colors for i<j
        for i in range(N):
            for j in range(i + 1, N):
                c = random.randint(0, 3)
                A[c][i, j] = 1
                A[c][j, i] = 1
                
        # Calculate initial T
        T = 0
        for c in range(4):
            # T_c = sum_{u < v} A_c[u,v] * dot(A_c[u,:], A_c[v,:])
            # This is 0.5 * sum_{u,v} A_c[u,v] * dot(...) because of symmetry?
            # Let's just sum u < v
            Ac = A[c]
            for u in range(N):
                for v in range(u + 1, N):
                    if Ac[u, v]:
                        T += np.dot(Ac[u, :], Ac[v, :])
                        
        best_T = T
        best_colors_flat = []
        # Store best state as flat list of colors for output
        def get_flat_colors():
            cols = [0] * num_edges
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    for c in range(4):
                        if A[c][i, j]:
                            cols[idx] = c
                            break
                    idx += 1
            return cols

        if T == 0:
            return get_flat_colors()

        best_colors_flat = get_flat_colors()

        current_temp = 1.0
        cooling_rate = 0.9995
        min_temp = 0.01
        
        # Precompute list of edges for faster random access?
        # Or just generate random i, j
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        while check_time():
            if current_temp < min_temp:
                # Restart from best
                for c in range(4):
                    A[c] = np.zeros((N, N), dtype=np.int8)
                for idx, c in enumerate(best_colors_flat):
                    i, j = edges[idx]
                    A[c][i, j] = 1
                    A[c][j, i] = 1
                T = 0
                for c in range(4):
                    Ac = A[c]
                    for u in range(N):
                        for v in range(u + 1, N):
                            if Ac[u, v]:
                                T += np.dot(Ac[u, :], Ac[v, :])
                current_temp = 1.0
                continue

            # SA Step
            idx = random.randint(0, num_edges - 1)
            u, v = edges[idx]
            
            # Find current color
            c_old = -1
            for c in range(4):
                if A[c][u, v]:
                    c_old = c
                    break
            
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            # Calculate delta
            # k_old = dot(A[c_old][u,:], A[c_old][v,:])
            # k_new = dot(A[c_new][u,:], A[c_new][v,:])
            k_old = np.dot(A[c_old][u, :], A[c_old][v, :])
            k_new = np.dot(A[c_new][u, :], A[c_new][v, :])
            delta = k_new - k_old
            
            if delta < 0 or (delta > 0 and random.random() < math.exp(-delta / current_temp)):
                # Accept
                A[c_old][u, v] = 0
                A[c_old][v, u] = 0
                A[c_new][u, v] = 1
                A[c_new][v, u] = 1
                T += delta
                
                if T < best_T:
                    best_T = T
                    best_colors_flat = get_flat_colors()
                    if T == 0:
                        return best_colors_flat
                        
            current_temp *= cooling_rate
            
        return None

    # Try N=50
    res = run_numPy_sa(50, deadline)
    if res:
        return {'n': 50, 'colors': res}
        
    # Try N=49 if time permits
    if check_time():
        res = run_numPy_sa(49, deadline)
        if res:
            return {'n': 49, 'colors': res}

    # Fallback
    return get_seed_32()
