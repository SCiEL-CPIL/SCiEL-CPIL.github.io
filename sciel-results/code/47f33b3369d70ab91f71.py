import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Baseline rank 42 solution
    rank_base = 42
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    U_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    V_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    W_coeffs_222 = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(2): # A block
        for b in range(3): # B col block
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs_222[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U_base[i * m + j][col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs_222[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V_base[j * p + k][col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs_222[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W_base[k * n + i][col] += coeff
                col += 1
                
    # Attempt to find a rank 40 solution by optimizing two 2x2x6 decompositions
    # If successful, return rank 40. Otherwise, return rank 42.
    
    # We will try to optimize one 2x2x6 decomposition to rank 20.
    # If we find one, we reuse it for both blocks.
    
    # Construct the 2x2x6 tensor
    n_sub, m_sub, p_sub = 2, 2, 6
    T = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub))
    for i in range(n_sub):
        for j in range(m_sub):
            for k in range(p_sub):
                a_idx = i * m_sub + j
                b_idx = j * p_sub + k
                c_idx = k * n_sub + i
                T[a_idx, b_idx, c_idx] = 1.0
                
    # Random initialization for rank 20
    rank_sub = 20
    # Use random initialization scaled appropriately
    np.random.seed(42)
    U_sub = np.random.randn(n_sub * m_sub, rank_sub) / np.sqrt(rank_sub)
    V_sub = np.random.randn(m_sub * p_sub, rank_sub) / np.sqrt(rank_sub)
    W_sub = np.random.randn(p_sub * n_sub, rank_sub) / np.sqrt(rank_sub)
    
    # Simple gradient descent / ALS optimization
    # We use Alternating Least Squares (ALS) which is more stable for tensor decomposition.
    # Objective: min || T - U V W ||_F^2
    
    best_error = np.inf
    best_U, best_V, best_W = U_sub, V_sub, W_sub
    
    for it in range(1000): # Max iterations
        if time.time() > deadline - 20:
            break
            
        # Update U
        # T is (I, J, K). We want min || T - U V W ||^2
        # For fixed V, W, U = T (V^T W)^-1? No.
        # Let's reshape T to (I, J*K).
        # T_ik = sum_s U_is V_js W_ks.
        # For each s, we have a vector.
        # Actually, ALS updates one factor at a time.
        # Fix V, W. Update U.
        # T = sum_s u_s v_s^T w_s^T? No, outer product.
        # T_{ijk} = sum_s U_{is} V_{js} W_{ks}.
        
        # We can write T as a matricization.
        # T_{i, jk} = sum_s U_{is} (V_{js} W_{ks})
        # Let M_{s, jk} = V_{js} W_{ks}. This is not quite right because s is summed.
        # T_{i, jk} = sum_s U_{is} M_{s, jk}
        # So T_{i, jk} = (U M^T)_{i, jk}
        # Thus U = T_{i, jk} (M^T)^+
        # Where M is (S, J*K). M_{s, jk} = V_{js} W_{ks}.
        # M = V^T \otimes W? No.
        # M_{s, jk} = V_{js} W_{ks}.
        # This is the element-wise product of V and W transposed?
        # V is (J, S), W is (K, S).
        # V^T is (S, J), W^T is (S, K).
        # We want M (S, J*K).
        # M_{s, jk} = V^T_{s, j} W^T_{s, k}.
        # This is the Khatri-Rao product? No.
        # It's just a matrix where row s is the outer product of V^T[s] and W^T[s].
        # So M = np.einsum('sj,sk->sjk', V.T, W.T).reshape(S, J*K)
        
        M = np.einsum('sj,sk->sjk', V_sub.T, W_sub.T).reshape(rank_sub, m_sub * p_sub)
        # T_reshaped is (I, J*K)
        T_reshaped = T.reshape(n_sub * m_sub, m_sub * p_sub)
        
        # Solve U M^T = T_reshaped
        # U = T_reshaped (M^T)^+
        # Using least squares: U M^T = T_reshaped => M U^T = T_reshaped^T
        # np.linalg.lstsq(M, T_reshaped.T)
        
        U_new, residuals, rank, s = np.linalg.lstsq(M, T_reshaped.T, rcond=None)
        U_new = U_new.T
        
        # Update V
        # T_{j, ik} = sum_s V_{js} (U_{is} W_{ks})
        # M2_{s, ik} = U_{is} W_{ks}
        M2 = np.einsum('si,sk->sik', U_new.T, W_sub.T).reshape(rank_sub, n_sub * p_sub)
        T_reshaped2 = T.transpose(1, 0, 2).reshape(m_sub * p_sub, n_sub * p_sub)
        V_new, residuals, rank, s = np.linalg.lstsq(M2, T_reshaped2.T, rcond=None)
        V_new = V_new.T
        
        # Update W
        # T_{k, ij} = sum_s W_{ks} (U_{is} V_{js})
        # M3_{s, ij} = U_{is} V_{js}
        M3 = np.einsum('si,sj->sij', U_new.T, V_new.T).reshape(rank_sub, n_sub * m_sub)
        T_reshaped3 = T.transpose(2, 0, 1).reshape(p_sub * n_sub, n_sub * m_sub)
        W_new, residuals, rank, s = np.linalg.lstsq(M3, T_reshaped3.T, rcond=None)
        W_new = W_new.T
        
        U_sub, V_sub, W_sub = U_new, V_new, W_new
        
        # Calculate error
        # T_approx_{ijk} = sum_s U_is V_js W_ks
        T_approx = np.einsum('is,js,ks->ijk', U_sub, V_sub, W_sub)
        error = np.linalg.norm(T - T_approx)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
            
        if best_error < 1e-10:
            break

    # Check if we found a good solution
    if best_error < 1e-8:
        # We have a rank 20 decomposition for 2x2x6
        # Construct the full rank 40 decomposition
        rank = 40
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Map the sub-decomposition to the full matrices
        # Sub U is (4, 20) corresponding to A (2x2)
        # Sub V is (12, 20) corresponding to B (2x6)
        # Sub W is (12, 20) corresponding to C (2x2, 6 cols -> 12 entries)
        
        # We need to place these into the full matrices.
        # A is 2x4. We have two blocks of 2x2.
        # The sub-decomposition was for 2x2x6.
        # This corresponds to A_L (2x2) and B (2x6).
        # But B in the full problem is 4x6.
        # The sub-decomposition assumes A is 2x2 and B is 2x6.
        # This matches A_L and B_1 (top half of B).
        # So we can use this for the a=0 block.
        # For a=1 block, we can use the same decomposition because the structure is identical (A_R and B_2).
        
        # So we place the sub_U into U for a=0 block.
        # Sub_U indices: i (0..1), j_loc (0..1).
        # Full U indices: i, j = a*2 + j_loc.
        # Sub_U is (4, 20). Row index r = i*2 + j_loc.
        # So for r in 0..3, i = r//2, j_loc = r%2.
        # For a=0, j = j_loc.
        # For a=1, j = 2 + j_loc.
        
        for r in range(4):
            i = r // 2
            j_loc = r % 2
            for s in range(20):
                val = best_U[r, s]
                # a=0
                U[i * m + j_loc][s] += val
                # a=1
                U[i * m + 2 + j_loc][s + 20] += val
        
        # Sub V is (12, 20). Row index r = j_loc*6 + k.
        # j_loc is 0..1, k is 0..5.
        # Full V indices: j, k.
        # For a=0, j = j_loc.
        # For a=1, j = 2 + j_loc.
        
        for r in range(12):
            j_loc = r // 6
            k = r % 6
            for s in range(20):
                val = best_V[r, s]
                # a=0
                V[j_loc * p + k][s] += val
                # a=1
                V[(2 + j_loc) * p + k][s + 20] += val
        
        # Sub W is (12, 20). Row index r = k*2 + i.
        # k is 0..5, i is 0..1.
        # Full W indices: k, i.
        # w_idx = k * n + i.
        
        for r in range(12):
            k = r // 2
            i = r % 2
            for s in range(20):
                val = best_W[r, s]
                # a=0
                W[k * n + i][s] += val
                # a=1
                W[k * n + i][s + 20] += val
        
        return {"U": U, "V": V, "W": W}
    else:
        # Return baseline
        return {"U": U_base, "V": V_base, "W": W_base}
