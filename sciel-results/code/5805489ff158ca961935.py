import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank_school = n * m * p  # 40

    start_time = time.time()
    deadline = start_time + 110.0

    # Build schoolbook decomposition as fallback
    U_school = [[0] * rank_school for _ in range(nm)]
    V_school = [[0] * rank_school for _ in range(mp)]
    W_school = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify_decomposition(U_mat, V_mat, W_mat):
        rank = U_mat.shape[1]
        # Check all 1-entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_mat[i * m + j]
                    v_val = V_mat[j * p + k]
                    w_val = W_mat[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check all 0-entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue
                    s = np.sum(U_mat[a] * V_mat[b] * W_mat[c])
                    if abs(s) > 1e-9:
                        return False
        return True

    def to_half_int_list(mat):
        return [[int(round(mat[i, r] * 2)) / 2 for r in range(mat.shape[1])] for i in range(mat.shape[0])]

    # Strassen 2x2x2 patterns
    strassen_patterns = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    # 2x2x1 patterns (rank 4)
    patterns_221 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]),
        ([0, 1, 0, 0], [0, 1], [1, 0]),
        ([0, 0, 1, 0], [1, 0], [0, 1]),
        ([0, 0, 0, 1], [0, 1], [0, 1]),
    ]

    # Build rank-36 solution (verified working)
    total_rank_36 = 36
    U36 = np.zeros((nm, total_rank_36), dtype=np.float64)
    V36 = np.zeros((mp, total_rank_36), dtype=np.float64)
    W36 = np.zeros((pn, total_rank_36), dtype=np.float64)

    col = 0
    for k_block in range(2):  # k=0,1 and k=2,3
        for j_block in range(2):  # j=0,1 and j=2,3
            for u_pat, v_pat, w_pat in strassen_patterns:
                for idx, val in enumerate(u_pat):
                    if val != 0:
                        i_local = idx // 2
                        j_local = idx % 2
                        U36[(i_local) * m + j_block * 2 + j_local, col] = val
                for idx, val in enumerate(v_pat):
                    if val != 0:
                        j_local = idx // 2
                        k_local = idx % 2
                        V36[(j_block * 2 + j_local) * p + k_block * 2 + k_local, col] = val
                for idx, val in enumerate(w_pat):
                    if val != 0:
                        i_local = idx // 2
                        k_local = idx % 2
                        W36[(k_block * 2 + k_local) * n + i_local, col] = val
                col += 1
    # Add 2x2x1 for k=4
    for j_block in range(2):
        for u_pat, v_pat, w_pat in patterns_221:
            for idx, val in enumerate(u_pat):
                if val != 0:
                    i_local = idx // 2
                    j_local = idx % 2
                    U36[i_local * m + j_block * 2 + j_local, col] = val
            for idx, val in enumerate(v_pat):
                if val != 0:
                    j_local = idx
                    V36[(j_block * 2 + j_local) * p + 4, col] = val
            for idx, val in enumerate(w_pat):
                if val != 0:
                    i_local = idx
                    W36[4 * n + i_local, col] = val
            col += 1

    assert col == total_rank_36, f"Expected 36, got {col}"

    if verify_decomposition(U36, V36, W36):
        best_result = {"U": to_half_int_list(U36), "V": to_half_int_list(V36), "W": to_half_int_list(W36)}
        best_rank = 36

    if time.time() > deadline - 20:
        return best_result

    # Try to find a rank-34 solution using a different blocking strategy
    # Use 2x2x3 (rank 10) + 2x2x2 (rank 7) per j-block
    # The 2x2x3 decomposition: 7 Strassen terms for k=0,1 + 3 terms for k=2
    # Each term is (u_vec, v_vec, w_vec) where u is 2-dim (i), v is 2-dim (j_local), w is 3-dim (k)
    
    def build_2x2x3_terms():
        """Build the 10 terms for 2x2x3 decomposition."""
        terms = []
        # 7 Strassen terms for k=0,1 (w = [w0, w1, 0])
        for u_pat, v_pat, w_pat in strassen_patterns:
            # u_pat: [u00, u01, u10, u11] -> u for (i,j) pairs
            # v_pat: [v00, v01, v10, v11] -> v for (j,k) pairs  
            # w_pat: [w00, w01, w10, w11] -> w for (k,i) pairs
            # In the 2x2x2 tensor T[i,j,k], each term contributes:
            #   sum over (i,j,k): u[i,j] * v[j,k] * w[k,i]
            # But for our purposes, we need to extract u_vec (2-dim for i), 
            # v_vec (2-dim for j), w_vec (2-dim for k)
            # Actually, the patterns are already in the right form for the 2x2x2 tensor.
            # u_pat[i*2+j] is the coefficient for A[i,j]
            # v_pat[j*2+k] is the coefficient for B[j,k]
            # w_pat[k*2+i] is the coefficient for C[k,i]
            
            # For the 2x2x3 tensor, we need:
            # U[i] * V[j] * W[k] for each term
            # But the Strassen patterns have 4 elements each, not 2.
            # This means they're not simple rank-1 in the (i,j,k) sense.
            
            # Let me reconsider. The 2x2x2 multiplication tensor is:
            # T[i,j,k] = 1 for all i,j,k in {0,1}
            # Strassen gives 7 rank-1 decompositions where each term is:
            #   M_r = sum_{i,j} A[i,j] * u_r[i,j] * v_r[j] * ... 
            # Actually, the standard form is:
            #   C = A*B, and M_r = (sum_{i,j} A[i,j] * U_r[i,j]) * (sum_{j,k} B[j,k] * V_r[j,k])
            #   C[i,k] += W_r[i,k] * M_r
            # So each term is U_r ⊗ V_r ⊗ W_r where U_r is 2x2, V_r is 2x2, W_r is 2x2
            # But that's not a rank-1 tensor in the (i,j,k) space.
            
            # For the tensor T[i,j,k], a rank-1 decomposition is:
            #   T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
            # where u_r is 2-dim, v_r is 2-dim, w_r is 2-dim.
            
            # The Strassen patterns as given are NOT in this form. They're in the
            # "matrix multiplication" form where U_r, V_r, W_r are 2x2 matrices.
            
            # I need to convert. For the 2x2x2 tensor T[i,j,k] = 1:
            # Each Strassen term r contributes:
            #   For each (i,j,k): U_r[i,j] * V_r[j,k] * W_r[k,i]
            # But this doesn't factor as u_r[i] * v_r[j] * w_r[k] in general.
            
            # So the Strassen patterns as stored are NOT directly usable as rank-1
            # terms in the (i,j,k) tensor space.
            
            # I need a different approach. Let me just solve for the 2x2x3 rank-10
            # decomposition directly.
            pass
        
        # Let me try a direct approach: solve for 10 rank-1 terms.
        # T[i,j,k] = 1 for all i,j,k in {0,1} x {0,1} x {0,1,2}
        # T = sum_{r=1}^{10} u_r[i] * v_r[j] * w_r[k]
        
        # This is a system of 12 equations in 70 unknowns.
        # Let me try a specific construction.
        
        # Approach: Use the fact that T = T_2x2x2 + T_2x2x1(k=2)
        # T_2x2x2 has rank 7, T_2x2x1 has rank 4, but combined rank is 10.
        
        # Let me try: 7 terms that work for k=0,1 and 3 terms that work for k=2,
        # with the 3 terms also contributing to k=0,1 in a way that the total is correct.
        
        # Actually, let me just try the simple construction:
        # 7 Strassen-like terms for k=0,1 (with w_r[2]=0)
        # 3 terms for k=2 (with w_r[0]=w_r[1]=0)
        # Total 10 terms.
        
        # For the 7 terms, I need rank-1 decompositions of the 2x2x2 tensor.
        # The 2x2x2 tensor T2[i,j,k] = 1 for all i,j,k has rank 7.
        # One rank-7 decomposition:
        
        # Let me use a known rank-7 decomposition of the 2x2x2 all-ones tensor.
        # Actually, the 2x2x2 multiplication tensor is NOT the all-ones tensor.
        # The multiplication tensor M[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
        # For 2x2x2: M[i,j,k] = 1 for all i,j,k in {0,1}.
        # So it IS the all-ones tensor in the (i,j,k) space!
        
        # The all-ones 2x2x2 tensor has rank 7 (by Strassen).
        # A rank-7 decomposition:
        # M = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
        
        # Let me construct one. The all-ones 2x2x2 tensor can be written as:
        # M = (e0+e1) ⊗ (e0+e1) ⊗ (e0+e1) + ... 
        # Actually, the rank of the all-ones dxdxd tensor is 2d-1 for d=2, so rank 3? No.
        # The all-ones 2x2x2 tensor: T[i,j,k] = 1 for all i,j,k.
        # T = (1,1) ⊗ (1,1) ⊗ (1,1) gives T[i,j,k] = 1 for all. Rank 1!
        # Wait, that's not right either. (1,1)⊗(1,1)⊗(1,1) has all entries 1.
        # So the all-ones 2x2x2 tensor has rank 1!
        
        # But the 2x2x2 matrix multiplication tensor is NOT the all-ones tensor.
        # The multiplication tensor M has M[i,j,k] = 1 if the (i,j) entry of A times
        # the (j,k) entry of B contributes to the (i,k) entry of C.
        # For 2x2x2: M[i,j,k] = 1 for all i,j,k in {0,1}.
        # So it IS the all-ones tensor.
        
        # I'm confused. Let me re-read the problem.
        # The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
        # For matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k].
        # So T[i,j,k] = 1 for all valid (i,j,k), and 0 otherwise.
        # For 2x4x5: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}.
        # That's 2*4*5 = 40 ones.
        
        # For 2x2x2: T[i,j,k] = 1 for all i,j,k in {0,1}. That's 8 ones.
        # The all-ones 2x2x2 tensor has rank 1 (it's (1,1)⊗(1,1)⊗(1,1)).
        # But Strassen says the 2x2x2 matrix multiplication has rank 7.
        
        # The resolution: the tensor T[i,j,k] is NOT a simple all-ones tensor in the
        # (i,j,k) space. It's a tensor where the index j is "contracted" - it appears
        # in both A and B. The correct tensor is:
        # T[i,j,k] = 1 for all i,j,k, but the rank is computed in the space where
        # i, j, k are independent indices.
        
        # Actually, I think I've been overthinking this. The tensor for 2x2x2 matrix
        # multiplication is:
        # T = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
        # where E_{a,b} is the matrix with 1 at position (a,b) and 0 elsewhere.
        # This is a tensor in the space (R^2⊗R^2) ⊗ (R^2⊗R^2) ⊗ (R^2⊗R^2).
        # The rank is 7 by Strassen.
        
        # For the 2x2x3 case, the tensor is in (R^2⊗R^2) ⊗ (R^2⊗R^3) ⊗ (R^2⊗R^3).
        # The rank is 10.
        
        # Let me just try to find a rank-10 decomposition numerically.
        
        # For now, let me just return the rank-36 solution.
        return None
    
    # Given the complexity, let's just return the verified rank-36 solution.
    return best_result
# EVOLVE-BLOCK-END
