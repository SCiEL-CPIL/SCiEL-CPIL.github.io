import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Build the rank-28 decomposition first
    rank28 = 28
    U28 = [[0] * rank28 for _ in range(n * m)]
    V28 = [[0] * rank28 for _ in range(m * p)]
    W28 = [[0] * rank28 for _ in range(p * n)]
    
    r = 0
    # Part 1: A2 B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U28[i * m + 2][r] = 1
            V28[2 * p + k][r] = 1
            W28[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) - 4 terms
    for i in range(n):
        for j in range(2):
            U28[i * m + j][r] = 1
            V28[j * p + 4][r] = 1
            W28[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen - 7 terms
    U28[0 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 0][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    U28[1 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 0][r] = 1
    W28[0 * n + 1][r] = 1
    W28[1 * n + 1][r] = -1
    r += 1
    
    U28[0 * m + 0][r] = 1
    V28[0 * p + 1][r] = 1
    V28[1 * p + 1][r] = -1
    W28[1 * n + 0][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    U28[1 * m + 1][r] = 1
    V28[1 * p + 0][r] = 1
    V28[0 * p + 0][r] = -1
    W28[0 * n + 0][r] = 1
    W28[0 * n + 1][r] = 1
    r += 1
    
    U28[0 * m + 0][r] = 1
    U28[0 * m + 1][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = -1
    W28[1 * n + 0][r] = 1
    r += 1
    
    U28[1 * m + 0][r] = 1
    U28[0 * m + 0][r] = -1
    V28[0 * p + 0][r] = 1
    V28[0 * p + 1][r] = 1
    W28[1 * n + 1][r] = 1
    r += 1
    
    U28[0 * m + 1][r] = 1
    U28[1 * m + 1][r] = -1
    V28[1 * p + 0][r] = 1
    V28[1 * p + 1][r] = 1
    W28[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen - 7 terms
    U28[0 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 2][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    U28[1 * m + 0][r] = 1
    U28[1 * m + 1][r] = 1
    V28[0 * p + 2][r] = 1
    W28[2 * n + 1][r] = 1
    W28[3 * n + 1][r] = -1
    r += 1
    
    U28[0 * m + 0][r] = 1
    V28[0 * p + 3][r] = 1
    V28[1 * p + 3][r] = -1
    W28[3 * n + 0][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    U28[1 * m + 1][r] = 1
    V28[1 * p + 2][r] = 1
    V28[0 * p + 2][r] = -1
    W28[2 * n + 0][r] = 1
    W28[2 * n + 1][r] = 1
    r += 1
    
    U28[0 * m + 0][r] = 1
    U28[0 * m + 1][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = -1
    W28[3 * n + 0][r] = 1
    r += 1
    
    U28[1 * m + 0][r] = 1
    U28[0 * m + 0][r] = -1
    V28[0 * p + 2][r] = 1
    V28[0 * p + 3][r] = 1
    W28[3 * n + 1][r] = 1
    r += 1
    
    U28[0 * m + 1][r] = 1
    U28[1 * m + 1][r] = -1
    V28[1 * p + 2][r] = 1
    V28[1 * p + 3][r] = 1
    W28[2 * n + 0][r] = 1
    r += 1
    
    assert r == rank28, f"Expected 28 terms, got {r}"
    
    # Verify the 28-term decomposition is correct
    U28_arr = np.array(U28, dtype=float)
    V28_arr = np.array(V28, dtype=float)
    W28_arr = np.array(W28, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(rank28):
                    s += U28_arr[i*m+j, rr] * V28_arr[j*p+k, rr] * W28_arr[k*n+i, rr]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    
    if max_err > 1e-9:
        # Fallback to naive rank 30
        rank30 = n * m * p
        U30 = [[0] * rank30 for _ in range(n * m)]
        V30 = [[0] * rank30 for _ in range(m * p)]
        W30 = [[0] * rank30 for _ in range(p * n)]
        r30 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j][r30] = 1
                    V30[j*p+k][r30] = 1
                    W30[k*n+i][r30] = 1
                    r30 += 1
        return {"U": U30, "V": V30, "W": W30}
    
    # Now search for a rank-27 decomposition by finding a linear dependency
    # among the 28 rank-1 tensors. If we find coefficients c_0..c_27 (not all zero)
    # such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0 (as a tensor), then we can
    # eliminate one term.
    
    # The tensor has dimension n*m*p = 30. Each rank-1 term contributes to all 30
    # entries. We need to find if the 28 vectors (flattened rank-1 tensors) are linearly dependent.
    
    # Flatten each rank-1 term into a 30-dim vector
    def flatten_term(U_row, V_row, W_row, n, m, p):
        """Flatten the rank-1 tensor U[i,j]*V[j,k]*W[k,i] into a 30-dim vector."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = U_row[i * m + j] * V_row[j * p + k] * W_row[k * n + i]
        return vec
    
    # Build the 30x28 matrix of flattened terms
    M = np.zeros((n * m * p, rank28))
    for rr in range(rank28):
        M[:, rr] = flatten_term(U28_arr[:, rr], V28_arr[:, rr], W28_arr[:, rr], n, m, p)
    
    # Check if the 28 columns are linearly dependent
    # Compute rank of M
    rank_M = np.linalg.matrix_rank(M, tol=1e-9)
    
    if rank_M < rank28:
        # There's a linear dependency. Find it.
        # Use SVD to find the null vector
        U_svd, S, Vt = np.linalg.svd(M, full_matrices=True)
        # The null space is spanned by the last (rank28 - rank_M) rows of Vt
        null_vec = Vt[-1, :]
        
        # Find the term with the largest absolute coefficient to eliminate
        max_idx = np.argmax(np.abs(null_vec))
        
        if abs(null_vec[max_idx]) > 1e-6:
            # We can eliminate term max_idx by adjusting the others
            # New decomposition: for each remaining term r, add (null_vec[max_idx]/null_vec[r]) * null_vec[r] * term_r
            # Actually, the dependency is: sum_r null_vec[r] * T_r = 0
            # So T_max = -sum_{r != max} (null_vec[r]/null_vec[max]) * T_r
            # This means we can drop T_max and add scaled versions of the others.
            # But we need integer/half-integer entries. Let's check if null_vec is rational.
            
            # Try to find a rational approximation
            # Scale to get small integers
            from fractions import Fraction
            null_frac = [Fraction(x).limit_denominator(1000) for x in null_vec]
            
            # Check if all are simple fractions
            all_simple = all(f.denominator <= 100 for f in null_frac)
            
            if all_simple:
                # Find common denominator
                from math import gcd
                from functools import reduce
                denoms = [f.denominator for f in null_frac]
                lcm = reduce(lambda a, b: a * b // gcd(a, b), denoms)
                
                int_vec = [int(f * lcm) for f in null_frac]
                
                # Find the term to eliminate
                elim_idx = max_idx
                scale = int_vec[elim_idx]
                
                if scale != 0:
                    # Build new rank-27 decomposition
                    rank27 = 27
                    U27 = [[0] * rank27 for _ in range(n * m)]
                    V27 = [[0] * rank27 for _ in range(m * p)]
                    W27 = [[0] * rank27 for _ in range(p * n)]
                    
                    new_r = 0
                    for rr in range(rank28):
                        if rr == elim_idx:
                            continue
                        coeff = int_vec[rr] / scale  # This should be rational
                        # Add coeff * term_rr to the new decomposition
                        # But we need to keep entries as integers/half-integers
                        # Check if coeff * entry is integer or half-integer
                        ok = True
                        for idx in range(n * m):
                            val = coeff * U28_arr[idx, rr]
                            if abs(val - round(val * 2) / 2) > 1e-6:
                                ok = False
                                break
                        if ok:
                            for idx in range(m * p):
                                val = coeff * V28_arr[idx, rr]
                                if abs(val - round(val * 2) / 2) > 1e-6:
                                    ok = False
                                    break
                        if ok:
                            for idx in range(p * n):
                                val = coeff * W28_arr[idx, rr]
                                if abs(val - round(val * 2) / 2) > 1e-6:
                                    ok = False
                                    break
                        
                        if ok:
                            for idx in range(n * m):
                                U27[idx][new_r] = coeff * U28_arr[idx, rr]
                            for idx in range(m * p):
                                V27[idx][new_r] = coeff * V28_arr[idx, rr]
                            for idx in range(p * n):
                                W27[idx][new_r] = coeff * W28_arr[idx, rr]
                            new_r += 1
                        else:
                            # Can't use this term with rational coefficients, break
                            break
                    
                    if new_r == rank27:
                        # Verify
                        U27_arr = np.array(U27, dtype=float)
                        V27_arr = np.array(V27, dtype=float)
                        W27_arr = np.array(W27, dtype=float)
                        
                        max_err27 = 0.0
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    s = 0.0
                                    for rr in range(rank27):
                                        s += U27_arr[i*m+j, rr] * V27_arr[j*p+k, rr] * W27_arr[k*n+i, rr]
                                    if abs(s - 1.0) > max_err27:
                                        max_err27 = abs(s - 1.0)
                        
                        if max_err27 < 1e-9:
                            return {"U": U27, "V": V27, "W": W27}
    
    # No rank-27 found, return rank 28
    return {"U": U28, "V": V28, "W": W28}
