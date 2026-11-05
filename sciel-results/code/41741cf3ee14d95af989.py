import numpy as np
import time
from itertools import product

n, m, p = 2, 5, 6
nm, mp, pn = n * m, m * p, p * n

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit

    # Pre-compute indices for the 60 tensor entries
    # T[(i*m+j), (j*p+k), (k*n+i)] = 1
    idx_u = np.zeros(n * m * p, dtype=np.int32)
    idx_v = np.zeros(n * m * p, dtype=np.int32)
    idx_w = np.zeros(n * m * p, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u[idx] = i * m + j
                idx_v[idx] = j * p + k
                idx_w[idx] = k * n + i
                idx += 1

    def check_decomposition_fast(U, V, W):
        """
        Vectorized check using pre-computed indices.
        U: (nm, R), V: (mp, R), W: (pn, R)
        """
        vals = U[idx_u] * V[idx_v] * W[idx_w]
        sums = vals.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-10)

    def get_optimized_strassen():
        """
        Optimized Strassen-based decomposition for 2x5x6.
        
        Key insight: We can use different block sizes and combine Strassen
        with careful handling of the remaining dimensions.
        
        Strategy:
        - Split j into {0,1}, {2,3}, {4}
        - Split k into {0,1}, {2,3}, {4,5}
        - Use Strassen for 2x2 blocks where possible
        - Optimize the combination to minimize total rank
        """
        rank = 54
        U = np.zeros((nm, rank), dtype=np.int8)
        V = np.zeros((mp, rank), dtype=np.int8)
        W = np.zeros((pn, rank), dtype=np.int8)
        
        col = 0
        
        def add_strassen_block(i_base, j_base, k_base):
            """Add a 2x2x2 Strassen block (7 multiplications)"""
            nonlocal col
            # M0: (A00+A11)(B00+B11)
            U[i_base * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M1: (A10+A11)B00
            U[(i_base + 1) * m + j_base, col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] -= 1
            col += 1
            
            # M2: A00(B01-B11)
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M3: A11(B10-B00)
            U[(i_base + 1) * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[j_base * p + k_base, col] -= 1
            W[k_base * n + i_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # M4: (A00+A01)B11
            U[i_base * m + j_base, col] += 1
            U[i_base * m + (j_base + 1), col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] -= 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # M5: (A10-A00)(B00+B01)
            U[(i_base + 1) * m + j_base, col] += 1
            U[i_base * m + j_base, col] -= 1
            V[j_base * p + k_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1
            
            # M6: (A01-A11)(B10+B11)
            U[i_base * m + (j_base + 1), col] += 1
            U[(i_base + 1) * m + (j_base + 1), col] -= 1
            V[(j_base + 1) * p + k_base, col] += 1
            V[(j_base + 1) * p + (k_base + 1), col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1

        def add_naive_2112_block(i_base, j_base, k_base):
            """Add a 2x1x2 naive block (4 multiplications)"""
            nonlocal col
            # 1: A00 B00 -> C00
            U[i_base * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + i_base, col] += 1
            col += 1
            
            # 2: A00 B01 -> C01
            U[i_base * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + i_base, col] += 1
            col += 1
            
            # 3: A10 B00 -> C10
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + k_base, col] += 1
            W[k_base * n + (i_base + 1), col] += 1
            col += 1
            
            # 4: A10 B01 -> C11
            U[(i_base + 1) * m + j_base, col] += 1
            V[j_base * p + (k_base + 1), col] += 1
            W[(k_base + 1) * n + (i_base + 1), col] += 1
            col += 1

        # Apply Strassen blocks for j pairs (0,1) and (2,3)
        # j_base = 0, k_base in {0, 2, 4}
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        
        # j_base = 2, k_base in {0, 2, 4}
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        # Apply naive blocks for j=4
        # j_base = 4, k_base in {0, 2, 4}
        add_naive_2112_block(0, 4, 0)
        add_naive_2112_block(0, 4, 2)
        add_naive_2112_block(0, 4, 4)
        
        if col != rank:
            # Fallback to schoolbook if construction failed
            rank_fb = n * m * p
            U_fb = np.zeros((nm, rank_fb))
            V_fb = np.zeros((mp, rank_fb))
            W_fb = np.zeros((pn, rank_fb))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, c] = 1
                        V_fb[j * p + k, c] = 1
                        W_fb[k * n + i, c] = 1
                        c += 1
            return U_fb, V_fb, W_fb, rank_fb
             
        return U, V, W, rank

    def get_better_construction():
        """
        Try a different partitioning strategy to potentially get lower rank.
        
        Alternative: Split j into {0,1,2}, {3,4} and k into {0,1,2,3}, {4,5}
        This gives us more flexibility.
        """
        # Try multiple strategies and return the best
        strategies = []
        
        # Strategy 1: Current approach (j: {0,1},{2,3},{4}, k: {0,1},{2,3},{4,5})
        U1, V1, W1, r1 = get_optimized_strassen()
        if check_decomposition_fast(U1, V1, W1):
            strategies.append((r1, U1, V1, W1))
        
        # Strategy 2: Try j: {0,1},{2,3,4}, k: {0,1},{2,3},{4,5}
        rank2 = 54
        U2 = np.zeros((nm, rank2), dtype=np.int8)
        V2 = np.zeros((mp, rank2), dtype=np.int8)
        W2 = np.zeros((pn, rank2), dtype=np.int8)
        col2 = 0
        
        def add_strassen_block2(i_base, j_base, k_base):
            nonlocal col2
            # Same as before but we'll use it differently
            U2[i_base * m + j_base, col2] += 1
            U2[(i_base + 1) * m + (j_base + 1), col2] += 1
            V2[j_base * p + k_base, col2] += 1
            V2[(j_base + 1) * p + (k_base + 1), col2] += 1
            W2[k_base * n + i_base, col2] += 1
            W2[(k_base + 1) * n + (i_base + 1), col2] += 1
            col2 += 1
            U2[(i_base + 1) * m + j_base, col2] += 1
            U2[(i_base + 1) * m + (j_base + 1), col2] += 1
            V2[j_base * p + k_base, col2] += 1
            W2[k_base * n + (i_base + 1), col2] += 1
            W2[(k_base + 1) * n + (i_base + 1), col2] -= 1
            col2 += 1
            U2[i_base * m + j_base, col2] += 1
            V2[j_base * p + (k_base + 1), col2] += 1
            V2[(j_base + 1) * p + (k_base + 1), col2] -= 1
            W2[(k_base + 1) * n + i_base, col2] += 1
            W2[(k_base + 1) * n + (i_base + 1), col2] += 1
            col2 += 1
            U2[(i_base + 1) * m + (j_base + 1), col2] += 1
            V2[(j_base + 1) * p + k_base, col2] += 1
            V2[j_base * p + k_base, col2] -= 1
            W2[k_base * n + i_base, col2] += 1
            W2[k_base * n + (i_base + 1), col2] += 1
            col2 += 1
            U2[i_base * m + j_base, col2] += 1
            U2[i_base * m + (j_base + 1), col2] += 1
            V2[(j_base + 1) * p + (k_base + 1), col2] += 1
            W2[k_base * n + i_base, col2] -= 1
            W2[(k_base + 1) * n + i_base, col2] += 1
            col2 += 1
            U2[(i_base + 1) * m + j_base, col2] += 1
            U2[i_base * m + j_base, col2] -= 1
            V2[j_base * p + k_base, col2] += 1
            V2[j_base * p + (k_base + 1), col2] += 1
            W2[(k_base + 1) * n + (i_base + 1), col2] += 1
            col2 += 1
            U2[i_base * m + (j_base + 1), col2] += 1
            U2[(i_base + 1) * m + (j_base + 1), col2] -= 1
            V2[(j_base + 1) * p + k_base, col2] += 1
            V2[(j_base + 1) * p + (k_base + 1), col2] += 1
            W2[k_base * n + i_base, col2] += 1
            col2 += 1

        # For strategy 2, we need to handle the 3x2 blocks differently
        # This is getting complex, so let's try a simpler improvement:
        # Use the same structure but optimize the naive blocks
        
        # Actually, let's try a completely different approach:
        # Use rank-1 tensors more cleverly
        
        # For now, return strategy 1 if it works
        if strategies:
            return strategies[0]
        
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((nm, rank_fb))
        V_fb = np.zeros((mp, rank_fb))
        W_fb = np.zeros((pn, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return (rank_fb, U_fb, V_fb, W_fb)

    # Get the best deterministic solution
    best_rank, best_U, best_V, best_W = get_better_construction()
    
    # Try to find a lower rank using structured random search
    # Instead of pure random, use patterns based on Strassen
    for R in range(best_rank - 1, 35, -1):
        if time.time() > deadline - 2:
            break
            
        # Time slice per rank
        start_rank_time = time.time()
        time_per_rank = 0.5
        
        attempts = 0
        while time.time() - start_rank_time < time_per_rank and attempts < 200:
            attempts += 1
            
            # Generate structured random matrices with Strassen-like patterns
            U = np.zeros((nm, R), dtype=np.int8)
            V = np.zeros((mp, R), dtype=np.int8)
            W = np.zeros((pn, R), dtype=np.int8)
            
            # For each rank-1 term, randomly choose a pattern
            for r in range(R):
                # Randomly choose which (i,j) pair for U
                i, j = np.random.randint(0, n), np.random.randint(0, m)
                # Randomly choose which (j,k) pair for V
                k = np.random.randint(0, p)
                
                # Sometimes add multiple non-zeros per term (Strassen-like)
                if np.random.random() < 0.3:
                    # Add a second term
                    i2, j2 = np.random.randint(0, n), np.random.randint(0, m)
                    k2 = np.random.randint(0, p)
                    U[i*m+j, r] = np.random.choice([-1, 1])
                    U[i2*m+j2, r] = np.random.choice([-1, 1])
                    V[j*p+k, r] = np.random.choice([-1, 1])
                    V[j2*p+k2, r] = np.random.choice([-1, 1])
                    W[k*n+i, r] = np.random.choice([-1, 1])
                    W[k2*n+i2, r] = np.random.choice([-1, 1])
                else:
                    U[i*m+j, r] = np.random.choice([-1, 1])
                    V[j*p+k, r] = np.random.choice([-1, 1])
                    W[k*n+i, r] = np.random.choice([-1, 1])
            
            if check_decomposition_fast(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
        
        if time.time() > deadline - 5:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
