"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

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
    
    def strassen_2x2_terms():
        """Return the 7 Strassen terms as (U, V, W) for 2x2x2 multiplication."""
        # U: (4,) = [a00, a01, a10, a11]
        # V: (2,) = [b00, b01]
        # W: (4,) = [C00, C10, C01, C11]
        return [
            (np.array([1.,0.,0.,1.]), np.array([1.,1.]), np.array([1.,0.,0.,-1.])),
            (np.array([0.,0.,1.,1.]), np.array([1.,0.]), np.array([0.,1.,-1.,0.])),
            (np.array([1.,0.,0.,0.]), np.array([0.,1.]), np.array([0.,0.,1.,1.])),
            (np.array([0.,0.,0.,1.]), np.array([1.,0.]), np.array([1.,-1.,0.,0.])),
            (np.array([1.,1.,0.,0.]), np.array([0.,1.]), np.array([-1.,0.,1.,0.])),
            (np.array([1.,-1.,0.,0.]), np.array([1.,1.]), np.array([0.,0.,0.,1.])),
            (np.array([0.,1.,-1.,0.]), np.array([1.,1.]), np.array([1.,0.,0.,0.])),
        ]
    
    def try_strassen_rank14():
        """Rank-14: two independent Strassen 2x2 blocks."""
        R = 14
        U = [[0.]*R for _ in range(4)]
        V = [[0.]*R for _ in range(8)]
        W = [[0.]*R for _ in range(8)]
        terms = strassen_2x2_terms()
        
        for t in range(7):
            r = t
            for i in range(4):
                U[i][r] = terms[t][0][i]
            for j in range(2):
                V[j][r] = terms[t][1][j]
            for k in range(4):
                W[k][r] = terms[t][2][k]
        
        for t in range(7):
            r = 7 + t
            for i in range(4):
                U[i][r] = terms[t][0][i]
            for j in range(2):
                V[2+j][r] = terms[t][1][j]
            for k in range(4):
                W[4+k][r] = terms[t][2][k]
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def solve_for_V(U_mat, W_mat):
        """Given U (4xR) and W (8xR), solve linear system for V (8xR)."""
        R = U_mat.shape[1]
        A = np.zeros((16, 8 * R))
        b = np.ones(16)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(R):
                        A[eq_idx, r * mp + j * p + k] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
        
        try:
            V_flat, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=1e-12)
        except:
            return None
        
        A_V = A @ V_flat
        if not np.allclose(A_V, b, atol=1e-9):
            return None
        
        V_sol = V_flat.reshape(R, mp).T
        return V_sol
    
    def try_rank13_shared_v():
        """
        Rank-13: merge one left Strassen term with one right Strassen term.
        The merged term has U from left term, W spanning both blocks.
        V is solved from the linear system.
        """
        if time.time() > deadline - 15:
            return None
        
        terms = strassen_2x2_terms()
        R = 13
        
        for li in range(7):
            for ri in range(7):
                if time.time() > deadline - 5:
                    return None
                
                U_mat = np.zeros((4, R))
                W_mat = np.zeros((8, R))
                
                # Merged term at column 0
                U_mat[:, 0] = terms[li][0]
                W_mat[:4, 0] = terms[li][2]
                W_mat[4:8, 0] = terms[ri][2]
                
                col = 1
                for l_idx in range(7):
                    if l_idx == li:
                        continue
                    U_mat[:, col] = terms[l_idx][0]
                    W_mat[:4, col] = terms[l_idx][2]
                    col += 1
                for r_idx in range(7):
                    if r_idx == ri:
                        continue
                    U_mat[:, col] = terms[r_idx][0]
                    W_mat[4:8, col] = terms[r_idx][2]
                    col += 1
                
                V_sol = solve_for_V(U_mat, W_mat)
                if V_sol is None:
                    continue
                
                U_list = U_mat.tolist()
                V_list = V_sol.tolist()
                W_list = W_mat.tolist()
                
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_rank13_general_u():
        """
        Rank-13: hybrid U = alpha*U_li + beta*U_ri for the merged term.
        """
        if time.time() > deadline - 15:
            return None
        
        terms = strassen_2x2_terms()
        R = 13
        
        for li in range(7):
            for ri in range(7):
                if time.time() > deadline - 5:
                    return None
                
                U_l = terms[li][0]
                U_r = terms[ri][0]
                
                if np.linalg.matrix_rank(np.column_stack([U_l, U_r])) < 2:
                    continue
                
                for alpha in [1.0, 0.5, -1.0, 2.0, -0.5]:
                    for beta in [1.0, 0.5, -1.0, 2.0, -0.5]:
                        if time.time() > deadline - 5:
                            return None
                        
                        U_hybrid = alpha * U_l + beta * U_r
                        if np.allclose(U_hybrid, 0):
                            continue
                        
                        U_mat = np.zeros((4, R))
                        W_mat = np.zeros((8, R))
                        
                        U_mat[:, 0] = U_hybrid
                        W_mat[:4, 0] = terms[li][2]
                        W_mat[4:8, 0] = terms[ri][2]
                        
                        col = 1
                        for l_idx in range(7):
                            if l_idx == li:
                                continue
                            U_mat[:, col] = terms[l_idx][0]
                            W_mat[:4, col] = terms[l_idx][2]
                            col += 1
                        for r_idx in range(7):
                            if r_idx == ri:
                                continue
                            U_mat[:, col] = terms[r_idx][0]
                            W_mat[4:8, col] = terms[r_idx][2]
                            col += 1
                        
                        V_sol = solve_for_V(U_mat, W_mat)
                        if V_sol is None:
                            continue
                        
                        U_list = U_mat.tolist()
                        V_list = V_sol.tolist()
                        W_list = W_mat.tolist()
                        
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    def try_rank13_general_w():
        """
        Rank-13: hybrid W with scaled components for the merged term.
        W_left = gamma * W_li, W_right = delta * W_ri
        """
        if time.time() > deadline - 15:
            return None
        
        terms = strassen_2x2_terms()
        R = 13
        
        for li in range(7):
            for ri in range(7):
                if time.time() > deadline - 5:
                    return None
                
                for gamma in [1.0, 0.5, -1.0, 2.0]:
                    for delta in [1.0, 0.5, -1.0, 2.0]:
                        if time.time() > deadline - 5:
                            return None
                        
                        U_mat = np.zeros((4, R))
                        W_mat = np.zeros((8, R))
                        
                        U_mat[:, 0] = terms[li][0]
                        W_mat[:4, 0] = gamma * terms[li][2]
                        W_mat[4:8, 0] = delta * terms[ri][2]
                        
                        col = 1
                        for l_idx in range(7):
                            if l_idx == li:
                                continue
                            U_mat[:, col] = terms[l_idx][0]
                            W_mat[:4, col] = terms[l_idx][2]
                            col += 1
                        for r_idx in range(7):
                            if r_idx == ri:
                                continue
                            U_mat[:, col] = terms[r_idx][0]
                            W_mat[4:8, col] = terms[r_idx][2]
                            col += 1
                        
                        V_sol = solve_for_V(U_mat, W_mat)
                        if V_sol is None:
                            continue
                        
                        U_list = U_mat.tolist()
                        V_list = V_sol.tolist()
                        W_list = W_mat.tolist()
                        
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try rank-13 searches
    result = try_rank13_shared_v()
    if result is not None:
        return result
    
    result = try_rank13_general_u()
    if result is not None:
        return result
    
    result = try_rank13_general_w()
    if result is not None:
        return result
    
    # Fallback to rank-14 Strassen
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
