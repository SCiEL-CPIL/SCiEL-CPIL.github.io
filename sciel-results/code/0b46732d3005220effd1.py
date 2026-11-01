import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Strassen parameters for 2x2x2
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    def build_strassen_28():
        num_blocks = p // 2
        rank = num_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for t in range(num_blocks):
            for r in range(7):
                for i in range(n * m):
                    U[i][col] = U_strassen[r][i]
                v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
                for l in range(4):
                    V[v_indices[l]][col] = V_strassen[r][l]
                w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
                for l in range(4):
                    W[w_indices[l]][col] = W_strassen[r][l]
                col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Use several random test matrices for robustness
        for trial in range(3):
            A_rand = np.random.randn(n, m)
            B_rand = np.random.randn(m, p)
            C_expected = A_rand @ B_rand
            
            C_check = np.zeros((n, p))
            A_flat = A_rand.flatten()
            B_flat = B_rand.flatten()
            
            for r in range(rank):
                scalar = np.dot(U_arr[:, r], A_flat) * np.dot(V_arr[:, r], B_flat)
                w_vec = W_arr[:, r]
                for k in range(p):
                    for i in range(n):
                        C_check[i, k] += w_vec[k * n + i] * scalar
            
            if not np.allclose(C_check, C_expected, atol=1e-10):
                return False
        return True
    
    best = build_strassen_28()
    best_rank = 28
    
    # Verify the rank-28 solution
    if not verify_decomposition(best["U"], best["V"], best["W"], best_rank, n, m, p):
        # Fallback to schoolbook if Strassen fails
        rank_school = n * m * p
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = rank_school
    
    # Try to find rank 27 using exact rational arithmetic
    # The matmul tensor T has entries T[a,b,c] where:
    #   a = i*m + j (index into A, i in [0,n), j in [0,m))
    #   b = j'*p + k (index into B, j' in [0,m), k in [0,p))
    #   c = k'*n + i' (index into C in trace convention, k' in [0,p), i' in [0,n))
    # T[a,b,c] = 1 if i=i' and j=j' and k=k', else 0
    # We need: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c
    
    # For rank 27, we have 27 unknowns per matrix (4*27 + 16*27 + 16*27 = 1080 unknowns)
    # and 4*16*16 = 1024 equations
    # This is an overdetermined system of bilinear equations
    
    # Strategy: Fix U and V to the Strassen structure but allow W to be adjusted
    # to see if we can eliminate one rank term
    
    # Alternative: Try a structured search where we modify the last block
    # to share a term with the first block
    
    if time.time() < deadline - 30.0:
        # Local search: check if any non-trivial integer linear combination of the 28
        # Strassen basis tensors yields zero (which would allow rank reduction)
        
        U28 = np.array(best["U"], dtype=Fraction)
        V28 = np.array(best["V"], dtype=Fraction)
        W28 = np.array(best["W"], dtype=Fraction)
        
        # Build the 28x1024 matrix of flattened rank-1 tensor terms
        # Each row t is the vectorization of U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        # Index: a * (16*16) + b * 16 + c for a in [0,4), b in [0,16), c in [0,16)
        
        # For efficiency, build this matrix using numpy with Fraction objects
        # But 28*1024 = 28672 Fraction operations is manageable
        
        # Actually, let's be smarter: we can check linear independence by
        # computing the rank of the 28x1024 matrix over Q
        
        # Build matrix M where M[t, idx] = U28[a,t] * V28[b,t] * W28[c,t]
        # with idx = a*256 + b*16 + c
        
        # For speed, use a sparse approach: most entries are 0 since Strassen
        # matrices are sparse
        
        # Let's build the matrix and do rational Gaussian elimination
        # But 28x1024 with Fractions might be slow. Let's try a different approach.
        
        # Alternative: check if the 28 vectors are linearly independent by
        # looking at their projections onto carefully chosen subspaces
        
        # Key insight: The 28 terms come in 4 groups of 7 (one per 2x2x2 block)
        # Within each block, the 7 Strassen terms are linearly independent
        # (since 2x2x2 has rank exactly 7)
        # Between blocks, the terms act on disjoint sets of B and C indices
        # (different column blocks of B and C), so they are automatically
        # linearly independent across blocks
        
        # Therefore, the 28 terms ARE linearly independent, and rank 27 is
        # NOT achievable with this block-Strassen structure
        
        # However, let's verify this computationally to be sure
        
        if time.time() < deadline - 20.0:
            # Build the 28x1024 matrix of tensor terms
            # Use a dictionary-based approach for sparse storage
            
            # For each term t, record which (a,b,c) indices are nonzero
            term_supports = []
            for t in range(28):
                support = {}
                for a in range(4):
                    if U28[a, t] == 0:
                        continue
                    for b in range(16):
                        if V28[b, t] == 0:
                            continue
                        for c in range(16):
                            if W28[c, t] == 0:
                                continue
                            val = U28[a, t] * V28[b, t] * W28[c, t]
                            idx = a * 256 + b * 16 + c
                            support[idx] = val
                term_supports.append(support)
            
            # Check linear independence by trying to find coefficients
            # c_0, ..., c_27 (not all zero) such that sum_t c_t * v_t = 0
            
            # This is equivalent to: for each index idx, sum_t c_t * v_t[idx] = 0
            # We can set up a linear system
            
            # Collect all indices that appear in any term
            all_indices = set()
            for support in term_supports:
                all_indices.update(support.keys())
            all_indices = sorted(all_indices)
            
            # Build the coefficient matrix: rows = indices, cols = terms
            # M[idx_i, t] = term_supports[t].get(idx_i, 0)
            
            # This is a (num_indices) x 28 matrix
            # We need to find the null space
            
            # Use Gaussian elimination over Q
            num_idx = len(all_indices)
            if num_idx > 0:
                # Build the matrix
                M = [[Fraction(0) for _ in range(28)] for _ in range(num_idx)]
                for i, idx in enumerate(all_indices):
                    for t in range(28):
                        if idx in term_supports[t]:
                            M[i][t] = term_supports[t][idx]
                
                # Gaussian elimination to find the rank
                rank = 0
                pivot_cols = []
                for col in range(28):
                    # Find pivot row
                    pivot_row = -1
                    for row in range(rank, num_idx):
                        if M[row][col] != 0:
                            pivot_row = row
                            break
                    if pivot_row == -1:
                        continue
                    # Swap
                    M[rank], M[pivot_row] = M[pivot_row], M[rank]
                    # Eliminate
                    pivot_val = M[rank][col]
                    for row in range(num_idx):
                        if row != rank and M[row][col] != 0:
                            factor = M[row][col] / pivot_val
                            for col2 in range(col, 28):
                                M[row][col2] -= factor * M[rank][col2]
                    pivot_cols.append(col)
                    rank += 1
                    if rank == 28:
                        break
                
                if rank < 28:
                    # There's a linear dependency!
                    # Find the null space vector
                    # The free variables correspond to columns not in pivot_cols
                    pivot_col_set = set(pivot_cols)
                    free_cols = [c for c in range(28) if c not in pivot_col_set]
                    
                    # For each free column, find a null space vector
                    for free_col in free_cols:
                        null_vec = [Fraction(0) for _ in range(28)]
                        null_vec[free_col] = Fraction(1)
                        
                        # Solve for pivot columns
                        for i in range(len(pivot_cols)):
                            col = pivot_cols[i]
                            # M[i][col] * null_vec[col] = -sum_{j>col} M[i][j] * null_vec[j]
                            sum_val = Fraction(0)
                            for j in range(col + 1, 28):
                                sum_val += M[i][j] * null_vec[j]
                            null_vec[col] = -sum_val / M[i][col]
                        
                        # Verify: sum_t null_vec[t] * v_t should be 0
                        # Check a few indices
                        valid = True
                        for idx in all_indices[:50]:  # Check first 50 indices
                            check_val = Fraction(0)
                            for t in range(28):
                                if idx in term_supports[t]:
                                    check_val += null_vec[t] * term_supports[t][idx]
                            if check_val != 0:
                                valid = False
                                break
                        
                        if valid:
                            # We found a linear dependency!
                            # This means we can potentially reduce the rank
                            # But we need to verify that the resulting decomposition
                            # still works for the actual tensor T
                            
                            # For now, just note that a dependency exists
                            # A full rank-27 search would be needed to exploit this
                            pass
            
            # The computational check is expensive. Based on the structure:
            # - Each 2x2x2 block has rank exactly 7 (no dependency within block)
            # - Different blocks act on disjoint index sets (no dependency across blocks)
            # Therefore, the 28 terms are linearly independent, and rank 27 is
            # not achievable with the block-Strassen approach
    
    return best
