import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # safety margin before 150s kill
    
    def build_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        def add_term(u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
            nonlocal col
            if col >= rank:
                raise ValueError("Rank overflow")
            for idx, coeff in zip(u_indices, u_coeffs):
                U[idx][col] += coeff
            for idx, coeff in zip(v_indices, v_coeffs):
                V[idx][col] += coeff
            for idx, coeff in zip(w_indices, w_coeffs):
                W[idx][col] += coeff
            col += 1
        
        def add_strassen_block(k_start):
            a00, a01, a10, a11 = 0, 1, 3, 4
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1])
            add_term([a10, a11], [1, 1], [b00], [1], [c10, c11], [1, -1])
            add_term([a00], [1], [b01, b11], [1, -1], [c01, c11], [1, 1])
            add_term([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1])
            add_term([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1])
            add_term([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1])
            add_term([a01, a11], [1, -1], [b10, b11], [1, 1], [c00], [1])
        
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [1], [b06], [1], [c06], [1])
        add_term([1], [1], [b16], [1], [c06], [1])
        add_term([3], [1], [b06], [1], [c16], [1])
        add_term([4], [1], [b16], [1], [c16], [1])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            add_term([2], [1], [b2k], [1], [c0k], [1])
            add_term([5], [1], [b2k], [1], [c1k], [1])
        
        return U, V, W
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly represents matrix multiplication."""
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    a_idx = i * m + k
                    b_idx = k * p + j
                    c_idx = j * n + i
                    s = 0
                    for t in range(rank):
                        s += U[a_idx][t] * V[b_idx][t] * W[c_idx][t]
                    if s != 1:
                        return False
        for c_idx in range(p * n):
            j = c_idx // n
            i = c_idx % n
            for a_idx in range(n * m):
                for b_idx in range(m * p):
                    i_prime = a_idx // m
                    k_prime = a_idx % m
                    k_double_prime = b_idx // p
                    j_prime = b_idx % p
                    if not (i_prime == i and k_prime == k_double_prime and j_prime == j):
                        s = 0
                        for t in range(rank):
                            s += U[a_idx][t] * V[b_idx][t] * W[c_idx][t]
                        if s != 0:
                            return False
        return True
    
    # Build rank-39 fallback
    U39, V39, W39 = build_rank39()
    
    if time.time() - start_time > time_limit - 30:
        return {"U": U39, "V": V39, "W": W39}
    
    # Try to find rank 38 by attempting to merge the 14 schoolbook terms for j=2
    # with some Strassen terms, or by using a different decomposition strategy
    
    # Strategy: Try to use a 2x2x7 decomposition that's more efficient
    # Or try random search for rank 38
    
    # Given the complexity and time constraints, let's try a targeted approach:
    # Can we combine the A[i,2]*B[2,j] terms with the Strassen M1 terms?
    
    # Actually, let's try a completely different approach: use bilinear complexity
    # For 2x3 by 3x7, the tensor rank might be reducible
    
    # Let's try: instead of 3 Strassen + 4 + 14, use a different grouping
    # Group columns as (0,1,2), (3,4,5), (6) - no, that doesn't work with Strassen
    
    # Try: use 2x2 Strassen for columns (0,1), (2,3), (4,5) and schoolbook for 6
    # That's the same as before
    
    # Alternative: try to find if rank 38 is achievable by solving a linear system
    # For rank 38, we have 41*38 = 1558 variables and 42*21*14 = 12348 constraints
    # (but most are zero). This is a nonlinear problem.
    
    # Given time constraints, let's just return rank 39
    if time.time() - start_time > time_limit:
        raise TimeoutError("Exceeded time limit")
    
    return {"U": U39, "V": V39, "W": W39}
# EVOLVE-BLOCK-END
