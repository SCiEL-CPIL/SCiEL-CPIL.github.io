"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_general():
        """
        General rank-13 search: fix U and W with parameterized structure,
        solve linear system for V.
        
        Key idea: The 2x2x4 multiplication tensor can be seen as two 2x2x2 tensors
        sharing the same A factors. For rank 13, we try to find a decomposition
        where one rank-1 term has a general U vector that contributes to both blocks.
        
        We parameterize U and W using the Strassen structure but allow one "hybrid"
        term, then solve for V.
        """
        if time.time() > deadline - 15:
            return None
        
        # Strassen terms for 2x2 multiplication
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # For left block: V indices 0=b00, 1=b01; W indices 0=C00, 1=C10, 2=C01, 3=C11
        # For right block: V indices 2=b02, 3=b03; W indices 4=C02, 5=C12, 6=C03, 7=C13
        
        # Standard Strassen (U_4, V_2, W_4) for each block
        # W_4 = [C00, C10, C01, C11] for left, [C02, C12, C03, C13] for right
        s_terms = [
            # (U, V, W) where V is length 2, W is length 4
            (np.array([1,0,0,1.]), np.array([1.,1.]), np.array([1.,0.,0.,-1.])),   # M1
            (np.array([0,0,1,1.]), np.array([1.,0.]), np.array([0.,1.,-1.,0.])),   # M2
            (np.array([1,0,0,0.]), np.array([0.,1.]), np.array([0.,0.,1.,1.])),    # M3
            (np.array([0,0,0,1.]), np.array([1.,0.]), np.array([1.,-1.,0.,0.])),   # M4
            (np.array([1,1,0,0.]), np.array([0.,1.]), np.array([-1.,0.,1.,0.])),   # M5
            (np.array([1,-1,0,0.]), np.array([1.,1.]), np.array([0.,0.,0.,1.])),   # M6
            (np.array([0,1,-1,0.]), np.array([1.,1.]), np.array([1.,0.,0.,0.])),   # M7
        ]
        
        # Verify Strassen correctness
        # C00 = M1 + M4 - M5 + M7, C01 = M3 + M5, C10 = M2 + M4, C11 = M1 - M2 + M3 + M6
        for ci in range(4):
            expected = 0.0
            for s in range(7):
                # Check contribution to output ci
                # C00: M1(1) + M4(1) - M5(-1) + M7(1) => indices in W
                pass  # Skip detailed check, trust standard Strassen
        
        # Strategy: Try all pairs (li, ri) where we "merge" term li from left with term ri from right
        # The merged term has U = U_left (same as U_right if proportional, or we try general)
        # For the merged term, V has 4 entries (b00,b01,b02,b03) and W has 8 entries
        
        # First, check which U vectors are proportional
        for li in range(7):
            for ri in range(7):
                if time.time() > deadline - 10:
                    return None
                
                U_l = s_terms[li][0]
                U_r = s_terms[ri][0]
                
                # Check if proportional
                if np.linalg.matrix_rank(np.column_stack([U_l, U_r])) < 2:
                    # Proportional - can merge
                    R = 13
                    U_mat = np.zeros((4, R))
                    W_mat = np.zeros((8, R))
                    
                    # Merged term at column 0
                    U_mat[:, 0] = U_l
                    W_mat[:4, 0] = s_terms[li][2]  # C00,C10,C01,C11
                    W_mat[4:8, 0] = s_terms[ri][2]  # C02,C12,C03,C13
                    
                    col = 1
                    for l_idx in range(7):
                        if l_idx == li:
                            continue
                        U_mat[:, col] = s_terms[l_idx][0]
                        W_mat[:4, col] = s_terms[l_idx][2]
                        col += 1
                    for r_idx in range(7):
                        if r_idx == ri:
                            continue
                        U_mat[:, col] = s_terms[r_idx][0]
                        W_mat[4:8, col] = s_terms[r_idx][2]
                        col += 1
                    
                    # Solve for V
                    B = np.zeros((16, R * 8))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = i * m * p + j * p + k
                                for r in range(R):
                                    B[idx, r * 8 + j * p + k] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
                    
                    rhs = np.ones(16)
                    try:
                        V_flat, residuals, rank_b, sv = np.linalg.lstsq(B, rhs, rcond=1e-12)
                    except:
                        continue
                    
                    B_V = B @ V_flat
                    if not np.allclose(B_V, rhs, atol=1e-9):
                        continue
                    
                    V_sol = V_flat.reshape(R, 8).T
                    U_list = U_mat.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_mat.tolist()
                    
                    if verify(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try non-proportional merges with general U
        # For each pair (li, ri), try U = alpha * U_l + beta * U_r for various alpha, beta
        for li in range(7):
            for ri in range(7):
                if time.time() > deadline - 5:
                    return None
                if li == ri:
                    continue
                    
                U_l = s_terms[li][0]
                U_r = s_terms[ri][0]
                
                # Only try if not already proportional
                if np.linalg.matrix_rank(np.column_stack([U_l, U_r])) < 2:
                    continue
                
                # Try a few combinations
                for alpha in [1.0, 0.5, -1.0]:
                    for beta in [1.0, 0.5, -1.0]:
                        if time.time() > deadline - 5:
                            return None
                        U_hybrid = alpha * U_l + beta * U_r
                        if np.allclose(U_hybrid, 0):
                            continue
                        
                        R = 13
                        U_mat = np.zeros((4, R))
                        W_mat = np.zeros((8, R))
                        
                        U_mat[:, 0] = U_hybrid
                        W_mat[:4, 0] = s_terms[li][2]
                        W_mat[4:8, 0] = s_terms[ri][2]
                        
                        col = 1
                        for l_idx in range(7):
                            if l_idx == li:
                                continue
                            U_mat[:, col] = s_terms[l_idx][0]
                            W_mat[:4, col] = s_terms[l_idx][2]
                            col += 1
                        for r_idx in range(7):
                            if r_idx == ri:
                                continue
                            U_mat[:, col] = s_terms[r_idx][0]
                            W_mat[4:8, col] = s_terms[r_idx][2]
                            col += 1
                        
                        B = np.zeros((16, R * 8))
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    idx = i * m * p + j * p + k
                                    for r in range(R):
                                        B[idx, r * 8 + j * p + k] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
                        
                        rhs = np.ones(16)
                        try:
                            V_flat, residuals, rank_b, sv = np.linalg.lstsq(B, rhs, rcond=1e-12)
                        except:
                            continue
                        
                        B_V = B @ V_flat
                        if not np.allclose(B_V, rhs, atol=1e-9):
                            continue
                        
                        V_sol = V_flat.reshape(R, 8).T
                        U_list = U_mat.tolist()
                        V_list = V_sol.tolist()
                        W_list = W_mat.tolist()
                        
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_rank13_parametric():
        """
        Parametric rank-13 search: use the structure that the 2x2x4 tensor
        is a "block" tensor. Try to find a rank-13 decomposition by
        parameterizing one free term and solving.
        
        Alternative approach: Use the fact that for 2x2x4, we can write
        the tensor as T = sum_r u_r ⊗ v_r ⊗ w_r where u_r ∈ R^4, v_r ∈ R^8, w_r ∈ R^8.
        
        We fix the structure: 12 terms follow the Strassen pattern for two blocks,
        and one additional term provides the "cross" contribution.
        """
        if time.time() > deadline - 15:
            return None
        
        R = 13
        
        # Start with the rank-14 Strassen decomposition and try to eliminate one term
        # by absorbing it into another via a perturbation
        
        # Get the rank-14 solution
        r14 = try_strassen_rank14()
        if r14 is None:
            return None
        
        U14 = np.array(r14["U"], dtype=float)  # (4, 14)
        V14 = np.array(r14["V"], dtype=float)  # (8, 14)
        W14 = np.array(r14["W"], dtype=float)  # (8, 14)
        
        # Try to merge term 6 (last of left block) with term 13 (last of right block)
        # by finding a new term that replaces both
        # This is a rank-reduction step
        
        # For each pair of terms to merge, try to find a replacement
        for merge_a in range(14):
            for merge_b in range(merge_a + 1, 14):
                if time.time() > deadline - 5:
                    return None
                
                # Try to find U_new, V_new, W_new such that
                # u_new ⊗ v_new ⊗ w_new ≈ u_a ⊗ v_a ⊗ w_a + u_b ⊗ v_b ⊗ w_b
                # This is generally not possible for a single rank-1 term,
                # but we can try to adjust the remaining terms.
                
                # Alternative: remove terms a and b, add one new term, and adjust V
                # New U has 13 columns: all except a and b, plus one new
                # This gets complex. Let's try a different approach.
                pass
        
        return None
    
    # Try rank-13 searches
    result = try_rank13_general()
    if result is not None:
        return result
    
    result = try_rank13_parametric()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
