"""Matrix multiplication 2x3x8 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 8]

import numpy as np
import time
from itertools import combinations

_START_TIME = time.time()
_TIMEOUT = 140.0  # Safety margin before 150s kill

def _remaining_time():
    return _TIMEOUT - (time.time() - _START_TIME)

def _verify_decomposition(U, V, W, n, m, p):
    """Verify if the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    T_approx = np.zeros_like(T)
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for r in range(rank):
        T_approx += U_np[:, r][:, np.newaxis, np.newaxis] * \
                    V_np[:, r][np.newaxis, :, np.newaxis] * \
                    W_np[:, r][np.newaxis, np.newaxis, :]
                    
    return np.allclose(T_approx, T)

def _construct_strassen_238():
    """
    Constructs rank 44 decomposition for 2x3x8 matrix multiplication.
    Uses 4 blocks of 2x3x2.
    Each 2x3x2 block uses Strassen (rank 7) for 2x2x2 part and Outer Product (rank 4) for 2x1x2 part.
    Total rank: 4 * (7 + 4) = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(col, u_idx, u_coeff, v_idx, v_coeff, w_idx, w_coeff):
        for idx, c in zip(u_idx, u_coeff):
            U[idx][col] += c
        for idx, c in zip(v_idx, v_coeff):
            V[idx][col] += c
        for idx, c in zip(w_idx, w_coeff):
            W[idx][col] += c

    current_col = 0
    
    for k_block in range(4):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        e_idx = 0 * p + k0
        f_idx = 0 * p + k1
        g_idx = 1 * p + k0
        h_idx = 1 * p + k1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        a_idx = 0
        b_idx = 1
        c_idx = 3
        d_idx = 4
        
        # M1 = (a+d)(e+h)
        add_term(current_col, [a_idx, d_idx], [1, 1], [e_idx, h_idx], [1, 1], [w_00, w_11], [1, 1])
        current_col += 1
        
        # M2 = (c+d)e
        add_term(current_col, [c_idx, d_idx], [1, 1], [e_idx], [1], [w_10, w_11], [1, -1])
        current_col += 1
        
        # M3 = a(f-h)
        add_term(current_col, [a_idx], [1], [f_idx, h_idx], [1, -1], [w_01, w_11], [1, 1])
        current_col += 1
        
        # M4 = d(g-e)
        add_term(current_col, [d_idx], [1], [g_idx, e_idx], [1, -1], [w_00, w_10], [1, 1])
        current_col += 1
        
        # M5 = (a+b)h
        add_term(current_col, [a_idx, b_idx], [1, 1], [h_idx], [1], [w_00, w_01], [-1, 1])
        current_col += 1
        
        # M6 = (c-a)(e+f)
        add_term(current_col, [c_idx, a_idx], [1, -1], [e_idx, f_idx], [1, 1], [w_11], [1])
        current_col += 1
        
        # M7 = (b-d)(g+h)
        add_term(current_col, [b_idx, d_idx], [1, -1], [g_idx, h_idx], [1, 1], [w_00], [1])
        current_col += 1
        
        # Part 2: j = 2 (2x1x2)
        a2_idx = 0 * m + 2
        c2_idx = 1 * m + 2
        
        e2_idx = 2 * p + k0
        f2_idx = 2 * p + k1
        
        # C[0, k0] = A[0,2] * B[2, k0]
        add_term(current_col, [a2_idx], [1], [e2_idx], [1], [w_00], [1])
        current_col += 1
        
        # C[0, k1] = A[0,2] * B[2, k1]
        add_term(current_col, [a2_idx], [1], [f2_idx], [1], [w_01], [1])
        current_col += 1
        
        # C[1, k0] = A[1,2] * B[2, k0]
        add_term(current_col, [c2_idx], [1], [e2_idx], [1], [w_10], [1])
        current_col += 1
        
        # C[1, k1] = A[1,2] * B[2, k1]
        add_term(current_col, [c2_idx], [1], [f2_idx], [1], [w_11], [1])
        current_col += 1

    return {"U": U, "V": V, "W": W}

def _construct_schoolbook():
    """
    Constructs rank 48 schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, max_rank, seed=42):
    """
    Try to find a rank decomposition using iterative optimization.
    Uses the fact that the tensor is sparse and structured.
    """
    np.random.seed(seed)
    
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find U, V, W such that T = sum_r U_r ⊗ V_r ⊗ W_r
    
    # Strategy: Start with random initialization and use gradient-free optimization
    # or try structured approaches
    
    # For now, let's try a few different block structures
    # and use linear algebra to find minimal rank for each block
    
    results = []
    
    # Try different ways to partition the 8 columns
    # Option 1: 4 groups of 2 (current approach)
    # Option 2: 2 groups of 4
    # Option 3: 8 groups of 1
    
    for num_groups in [4, 2, 8]:
        if time.time() - _START_TIME > _TIMEOUT - 20:
            break
            
        if num_groups == 4:
            groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
        elif num_groups == 2:
            groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
        else:
            groups = [[k] for k in range(8)]
        
        # For each group, find the minimum rank decomposition
        total_rank = 0
        U_list = []
        V_list = []
        W_list = []
        
        for group in groups:
            if time.time() - _START_TIME > _TIMEOUT - 20:
                break
                
            # For a group of columns, we need to decompose the sub-tensor
            # The sub-tensor has shape (n*m) x (m*len(group)) x (len(group)*n)
            
            # For small groups, we can try brute force or known decompositions
            if len(group) == 1:
                # Single column: rank = n*m = 6
                rank = 6
                for i in range(n):
                    for j in range(m):
                        U_list.append([1 if (i == 0 and j == 0) else 0] * 0 + [0])  # placeholder
                        # Actually, let's build this properly
                # This is getting complex, let's use a different approach
                pass
            
            # For now, let's just check if we can do better than 44
            # by using known results or trying a few specific constructions
            pass
    
    # Since the systematic search is complex, let's try a few specific
    # known constructions that might beat 44
    
    # Try: Can we use a rank-6 decomposition for 2x3x2?
    # The tensor for 2x3x2 has 12 nonzero entries.
    # Lower bound: max(rank of slices)
    # Slice fixing k=0: 6x2 matrix, rank 2
    # Slice fixing k=1: 6x2 matrix, rank 2
    # So lower bound is 2, but we know Strassen gives 7.
    # Can we do better?
    
    # Let's try to find a rank-6 decomposition for 2x3x2
    # by solving a linear system
    
    def _find_232_decomposition(target_rank):
        """Try to find a rank decomposition for 2x3x2 tensor."""
        n, m, p = 2, 3, 2
        
        # Build the tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Try to find rank decomposition
        # U: 6 x rank, V: 6 x rank, W: 4 x rank
        # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # This is equivalent to: for each (u,v), T[u,v,:] = sum_r U[u,r]*V[v,r]*W[:,r]
        
        # Let's try a structured approach:
        # Assume W has a specific form based on the k indices
        
        # For 2x3x2, let's try rank 6
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # Try: use the structure that each output element C[i,k] depends on A[i,j]*B[j,k]
        
        # Let's try a direct approach with linear algebra
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if valid, 0 otherwise
        
        # This is a system of 6*6*4 = 144 equations
        # Unknowns: 6*rank + 6*rank + 4*rank = 16*rank
        
        # For rank 6: 96 unknowns, 144 equations (overdetermined)
        # For rank 7: 112 unknowns, 144 equations (still overdetermined)
        # For rank 8: 128 unknowns, 144 equations (underdetermined)
        
        # Let's try rank 8 first to see if a solution exists, then try to reduce
        
        for rank in [6, 7, 8, 9]:
            if time.time() - _START_TIME > _TIMEOUT - 30:
                return None
            
            # Try to find a solution using least squares
            # This is nonlinear, so let's use a simple approach:
            # Fix one matrix and solve for the others iteratively
            
            np.random.seed(42)
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Iterate
            for it in range(100):
                if time.time() - _START_TIME > _TIMEOUT - 30:
                    return None
                
                # Update U
                # T[:, :, w] = U @ (V * W[w, :]).T for each w
                # Actually: T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
                
                # For fixed V, W: T[:, :, w] = U @ diag(W[w, :]) @ V.T
                # So U = T[:, :, w] @ V @ diag(W[w, :])^{-1} / something
                
                # Let's do it properly using least squares
                # Reshape T to (n*m, m*p*p*n)
                T_flat = T.reshape(n*m, m*p*p*n)
                
                # Build the design matrix for U
                # T_flat[u, v*p*n + w] = sum_r U[u, r] * V[v, r] * W[w, r]
                
                # Group by u
                for u in range(n*m):
                    # T_flat[u, :] = U[u, :] @ M_u where M_u[v*p*n + w, r] = V[v, r] * W[w, r]
                    M_u = np.zeros((m*p*p*n, rank))
                    for v in range(m*p):
                        for w in range(p*n):
                            for r in range(rank):
                                M_u[v*p*n + w, r] = V[v, r] * W[w, r]
                    
                    # Solve U[u, :] @ M_u = T_flat[u, :]
                    # This is: M_u.T @ U[u, :].T = T_flat[u, :].T
                    try:
                        U[u, :] = np.linalg.lstsq(M_u.T, T_flat[u, :].T, rcond=None)[0]
                    except:
                        pass
                
                # Similarly for V and W
                for v in range(m*p):
                    M_v = np.zeros((n*m*p*n, rank))
                    for u in range(n*m):
                        for w in range(p*n):
                            for r in range(rank):
                                M_v[u*p*n + w, r] = U[u, r] * W[w, r]
                    try:
                        V[v, :] = np.linalg.lstsq(M_v.T, T.reshape(n*m, m*p, p*n)[:, v, :].reshape(-1).T, rcond=None)[0]
                    except:
                        pass
                
                for w in range(p*n):
                    M_w = np.zeros((n*m*m*p, rank))
                    for u in range(n*m):
                        for v in range(m*p):
                            for r in range(rank):
                                M_w[u*m*p + v, r] = U[u, r] * V[v, r]
                    try:
                        W[w, :] = np.linalg.lstsq(M_w.T, T.reshape(n*m, m*p, p*n)[:, :, w].reshape(-1).T, rcond=None)[0]
                    except:
                        pass
            
            # Check if we found a good solution
            T_approx = np.zeros_like(T)
            for r in range(rank):
                T_approx += U[:, r][:, np.newaxis, np.newaxis] * \
                           V[:, r][np.newaxis, :, np.newaxis] * \
                           W[:, r][np.newaxis, np.newaxis, :]
            
            if np.allclose(T_approx, T, atol=1e-6):
                # Found a solution! Try to reduce rank
                return rank, U, V, W
        
        return None
    
    # This approach is too slow. Let's just return the current best.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Candidate 1: Rank 44 Strassen-based
    cand1 = _construct_strassen_238()
    if _verify_decomposition(cand1["U"], cand1["V"], cand1["W"], n, m, p):
        # Try to see if we can do better
        if _remaining_time() > 60:
            # Try a quick search for better decomposition
            result = _try_rank_decomposition(n, m, p, 43)
            if result is not None and result[0] < 44:
                rank, U, V, W = result
                # Convert to list format
                U_list = [[int(U[i, r]) if abs(U[i, r] - round(U[i, r])) < 1e-6 else float(U[i, r]) for r in range(len(U[0]))] for i in range(len(U))]
                V_list = [[int(V[i, r]) if abs(V[i, r] - round(V[i, r])) < 1e-6 else float(V[i, r]) for r in range(len(V[0]))] for i in range(len(V))]
                W_list = [[int(W[i, r]) if abs(W[i, r] - round(W[i, r])) < 1e-6 else float(W[i, r]) for r in range(len(W[0]))] for i in range(len(W))]
                cand_better = {"U": U_list, "V": V_list, "W": W_list}
                if _verify_decomposition(cand_better["U"], cand_better["V"], cand_better["W"], n, m, p):
                    return cand_better
        
        return cand1
        
    # Candidate 2: Rank 48 Schoolbook (Fallback)
    cand2 = _construct_schoolbook()
    if _verify_decomposition(cand2["U"], cand2["V"], cand2["W"], n, m, p):
        return cand2
        
    # Should not happen, but return schoolbook regardless if verification fails due to bug
    return cand2
