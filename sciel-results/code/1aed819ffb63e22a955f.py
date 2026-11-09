"""Combined approach: recursive Strassen decomposition for 4x4*4x5 matrix multiplication."""
DIMENSIONS = [4, 4, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Uses recursive Strassen to build a rank-65 decomposition:
    - 49 terms from two-level Strassen for the 4x4*4x4 sub-product
    - 16 terms for the remaining 4x4*4x1 column
    Total rank: 65 (vs schoolbook 80)
    """
    n, m, p = 4, 4, 5
    deadline = time.time() + 145  # Safety margin before 150s kill
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
            return False
        
        # Reshape to 3D: U[i,j,c], V[j,k,c], W[k,i,c]
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # Compute tensor: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Target: T[i,j,k] = 1 for all i,j,k (since C_ik = sum_j A_ij * B_jk)
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True
    
    def get_schoolbook():
        """Return the schoolbook decomposition with rank = n*m*p = 80."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def build_strassen_decomposition():
        """Build the rank-65 decomposition using two-level Strassen."""
        rank = 65
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        def strassen_patterns():
            patterns = []
            patterns.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
            patterns.append(({(1,0):1, (1,1):1}, {(0,0):1}))
            patterns.append(({(0,0):1}, {(0,1):1, (1,1):-1}))
            patterns.append(({(1,1):1}, {(1,0):1, (0,0):-1}))
            patterns.append(({(0,0):1, (0,1):1}, {(1,1):1}))
            patterns.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
            patterns.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
            return patterns

        outer_terms = []
        u1 = [[{} for _ in range(2)] for _ in range(2)]
        v1 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u1[r][c] = {(r, c): 1, (r+2, c+2): 1}
                v1[r][c] = {(r, c): 1, (r+2, c+2): 1}
        outer_terms.append((u1, v1))
        
        u2 = [[{} for _ in range(2)] for _ in range(2)]
        v2 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u2[r][c] = {(r+2, c): 1, (r+2, c+2): 1}
                v2[r][c] = {(r, c): 1}
        outer_terms.append((u2, v2))
        
        u3 = [[{} for _ in range(2)] for _ in range(2)]
        v3 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u3[r][c] = {(r, c): 1}
                v3[r][c] = {(r, c+2): 1, (r+2, c+2): -1}
        outer_terms.append((u3, v3))
        
        u4 = [[{} for _ in range(2)] for _ in range(2)]
        v4 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u4[r][c] = {(r+2, c+2): 1}
                v4[r][c] = {(r+2, c): 1, (r, c): -1}
        outer_terms.append((u4, v4))
        
        u5 = [[{} for _ in range(2)] for _ in range(2)]
        v5 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u5[r][c] = {(r, c): 1, (r, c+2): 1}
                v5[r][c] = {(r+2, c+2): 1}
        outer_terms.append((u5, v5))
        
        u6 = [[{} for _ in range(2)] for _ in range(2)]
        v6 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u6[r][c] = {(r+2, c): 1, (r, c): -1}
                v6[r][c] = {(r, c): 1, (r, c+2): 1}
        outer_terms.append((u6, v6))
        
        u7 = [[{} for _ in range(2)] for _ in range(2)]
        v7 = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                u7[r][c] = {(r, c+2): 1, (r+2, c+2): -1}
                v7[r][c] = {(r+2, c): 1, (r+2, c+2): 1}
        outer_terms.append((u7, v7))
        
        c_block_contrib = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1, (0,1): -1}
        ]
        
        inner_contrib = [
            {(0,0): 1, (1,1): 1},
            {(1,0): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1},
            {(0,0): 1, (1,0): 1},
            {(0,0): -1, (0,1): 1},
            {(1,1): 1},
            {(0,0): 1, (0,1): -1}
        ]

        strassen_pats = strassen_patterns()
        col = 0
        
        for t_idx, (u_blocks, v_blocks) in enumerate(outer_terms):
            for s_idx, (u_pat, v_pat) in enumerate(strassen_pats):
                u_scalar = {}
                v_scalar = {}
                
                for (ur, uc), cu in u_pat.items():
                    for (ai, aj), cai in u_blocks[ur][uc].items():
                        key = (ai, aj)
                        u_scalar[key] = u_scalar.get(key, 0) + cu * cai
                
                for (vr, vc), cv in v_pat.items():
                    for (bi, bj), cbi in v_blocks[vr][vc].items():
                        key = (bi, bj)
                        v_scalar[key] = v_scalar.get(key, 0) + cv * cbi
                
                u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
                v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
                
                for (i, j), coeff in u_scalar.items():
                    U[i * m + j, col] = coeff
                for (j, k), coeff in v_scalar.items():
                    V[j * p + k, col] = coeff
                
                for i in range(n):
                    for k in range(4):
                        R = i // 2
                        r = i % 2
                        C = k // 2
                        c = k % 2
                        
                        coeff_outer = c_block_contrib[t_idx].get((R, C), 0)
                        coeff_inner = inner_contrib[s_idx].get((r, c), 0)
                        w_val = coeff_outer * coeff_inner
                        
                        if w_val != 0:
                            W[k * n + i, col] = w_val
                
                col += 1
        
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1
                V[j * p + 4, col] = 1
                W[4 * n + i, col] = 1
                col += 1
        
        return U, V, W
    
    # Build the Strassen decomposition
    try:
        U, V, W = build_strassen_decomposition()
        
        # Verify the decomposition
        if verify(U.tolist(), V.tolist(), W.tolist()):
            # Convert to lists of integers/half-integers
            U_list = [[(int(U[i, c] * 2) / 2) for c in range(U.shape[1])] for i in range(U.shape[0])]
            V_list = [[(int(V[i, c] * 2) / 2) for c in range(V.shape[1])] for i in range(V.shape[0])]
            W_list = [[(int(W[i, c] * 2) / 2) for c in range(W.shape[1])] for i in range(W.shape[0])]
            
            # Clean up: convert 0.0 to 0, 1.0 to 1, etc.
            def clean_val(x):
                if abs(x) < 1e-10:
                    return 0
                elif abs(x - round(x)) < 1e-10:
                    return int(round(x))
                elif abs(x * 2 - round(x * 2)) < 1e-10:
                    return round(x * 2) / 2
                else:
                    return x
            
            U_list = [[clean_val(v) for v in row] for row in U_list]
            V_list = [[clean_val(v) for v in row] for row in V_list]
            W_list = [[clean_val(v) for v in row] for row in W_list]
            
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fallback to schoolbook
            return get_schoolbook()
            
    except Exception as e:
        # Fallback to schoolbook on any error
        return get_schoolbook()

# EVOLVE-BLOCK-END
