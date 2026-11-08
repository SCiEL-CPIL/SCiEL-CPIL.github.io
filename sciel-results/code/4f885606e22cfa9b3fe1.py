"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strategy:
    # 1. Solve for M_{2,2,6} decomposition with minimal rank R_sub.
    # 2. Construct M_{2,4,6} decomposition with rank 2 * R_sub.
    # 3. If optimization fails, fall back to rank 42.
    
    def solve_sub_problem(n_sub, m_sub, p_sub, max_rank, min_rank):
        """
        Find a rank-minimal decomposition for M_{n_sub, m_sub, p_sub}.
        Returns (U, V, W, rank) or None.
        """
        # Build target tensor T_sub
        T_sub = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=np.float64)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    T_sub[i * m_sub + j, j * p_sub + k, k * n_sub + i] = 1.0
        
        T_flat = T_sub.flatten()
        
        def tensor_product(U, V, W, R):
            result = np.zeros((n_sub * m_sub, m_sub * p_sub, p_sub * n_sub), dtype=np.float64)
            for r in range(R):
                result += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            return result
        
        def objective_and_grads(U, V, W, R):
            prod = tensor_product(U, V, W, R)
            diff = prod - T_sub
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(R):
                grad_U[:, r] = np.einsum('ijk,j,k->i', diff, V[:, r], W[:, r])
                grad_V[:, r] = np.einsum('ijk,i,k->j', diff, U[:, r], W[:, r])
                grad_W[:, r] = np.einsum('ijk,i,j->k', diff, U[:, r], V[:, r])
            
            obj = np.dot(diff.flatten(), diff.flatten())
            return obj, grad_U, grad_V, grad_W
        
        def verify_decomp(U, V, W):
            prod = tensor_product(U, V, W, U.shape[1])
            return np.max(np.abs(prod - T_sub))
        
        np.random.seed(42)
        
        def snap_to_half_int(M):
            return np.round(M * 2.0) / 2.0

        # Try ranks from min to max
        for R in range(min_rank, max_rank + 1):
            if time.time() > deadline - 30:
                break
            
            found = False
            max_attempts = 10
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 25:
                    break
                
                # Random initialization
                U_try = np.random.randn(n_sub * m_sub, R) * 0.5
                V_try = np.random.randn(m_sub * p_sub, R) * 0.5
                W_try = np.random.randn(p_sub * n_sub, R) * 0.5
                
                lr = 0.05
                obj_prev = float('inf')
                stall_count = 0
                
                max_steps = 300
                for step in range(max_steps):
                    if time.time() > deadline - 20:
                        break
                    
                    obj, dU, dV, dW = objective_and_grads(U_try, V_try, W_try, R)
                    
                    if obj < 1e-10:
                        found = True
                        break
                    
                    if obj < obj_prev:
                        stall_count = 0
                    else:
                        stall_count += 1
                    
                    if stall_count > 20:
                        lr *= 0.5
                        if lr < 1e-6:
                            break
                    
                    U_try -= lr * dU
                    V_try -= lr * dV
                    W_try -= lr * dW
                    
                    obj_prev = obj
                
                # Snap to half-integers
                if obj < 1e-4:
                    U_snap = snap_to_half_int(U_try)
                    V_snap = snap_to_half_int(V_try)
                    W_snap = snap_to_half_int(W_try)
                    
                    max_err_snap = verify_decomp(U_snap, V_snap, W_snap)
                    if max_err_snap < 1e-9:
                        found = True
                        U_try, V_try, W_try = U_snap, V_snap, W_snap
                        break
                
                if found:
                    break
            
            if found:
                return U_try, V_try, W_try, R
        
        return None

    # Solve for M_{2,2,6}
    # n_sub=2, m_sub=2, p_sub=6
    # Try ranks 12 to 16
    sub_result = solve_sub_problem(2, 2, 6, max_rank=16, min_rank=12)
    
    if sub_result is not None:
        U_sub, V_sub, W_sub, R_sub = sub_result
        R_full = 2 * R_sub
        
        # Construct full matrices
        U_full = np.zeros((n * m, R_full), dtype=np.float64)
        V_full = np.zeros((m * p, R_full), dtype=np.float64)
        W_full = np.zeros((p * n, R_full), dtype=np.float64)
        
        # Mapping for U
        # A_L (j=0,1) -> indices 0,1,4,5 in full. Local indices 0,1,2,3.
        # A_R (j=2,3) -> indices 2,3,6,7 in full. Local indices 0,1,2,3.
        
        # U_full[0:4, 0:R_sub] corresponds to A_L?
        # No, U_full has 8 rows.
        # Rows 0,1,4,5 are A_L. Rows 2,3,6,7 are A_R.
        
        # Let's map carefully.
        # Full index a = i * 4 + j.
        # Sub index a_sub = i * 2 + j_loc.
        
        # For A_L (j in 0,1):
        # a=0 (i=0, j=0) -> a_sub=0
        # a=1 (i=0, j=1) -> a_sub=1
        # a=4 (i=1, j=0) -> a_sub=2
        # a=5 (i=1, j=1) -> a_sub=3
        
        # For A_R (j in 2,3):
        # a=2 (i=0, j=2) -> a_sub=0 (j_loc=0)
        # a=3 (i=0, j=3) -> a_sub=1 (j_loc=1)
        # a=6 (i=1, j=2) -> a_sub=2 (j_loc=0)
        # a=7 (i=1, j=3) -> a_sub=3 (j_loc=1)
        
        # Fill U_full
        # First R_sub columns from U_sub for A_L
        U_full[0, 0:R_sub] = U_sub[0]
        U_full[1, 0:R_sub] = U_sub[1]
        U_full[4, 0:R_sub] = U_sub[2]
        U_full[5, 0:R_sub] = U_sub[3]
        
        # Next R_sub columns from U_sub for A_R
        U_full[2, R_sub:2*R_sub] = U_sub[0]
        U_full[3, R_sub:2*R_sub] = U_sub[1]
        U_full[6, R_sub:2*R_sub] = U_sub[2]
        U_full[7, R_sub:2*R_sub] = U_sub[3]
        
        # Mapping for V
        # Full index b = j' * 6 + k.
        # Sub index b_sub = j'_sub * 6 + k.
        
        # For B_T (j' in 0,1):
        # b=0..5 (j'=0) -> b_sub=0..5
        # b=6..11 (j'=1) -> b_sub=6..11
        
        # For B_B (j' in 2,3):
        # b=12..17 (j'=2) -> b_sub=0..5 (j'_sub=0)
        # b=18..23 (j'=3) -> b_sub=6..11 (j'_sub=1)
        
        # Fill V_full
        # First R_sub columns from V_sub for B_T
        V_full[0:6, 0:R_sub] = V_sub[0:6]
        V_full[6:12, 0:R_sub] = V_sub[6:12]
        
        # Next R_sub columns from V_sub for B_B
        V_full[12:18, R_sub:2*R_sub] = V_sub[0:6]
        V_full[18:24, R_sub:2*R_sub] = V_sub[6:12]
        
        # Mapping for W
        # W is the same for both, just duplicated in columns.
        W_full[:, 0:R_sub] = W_sub
        W_full[:, R_sub:2*R_sub] = W_sub
        
        # Verify full decomposition
        def verify_full(U, V, W):
            # Reconstruct T
            T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1.0
            prod = np.zeros_like(T)
            for r in range(U.shape[1]):
                prod += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            return np.max(np.abs(prod - T))
        
        if verify_full(U_full, V_full, W_full) < 1e-9:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
        
    # Fallback to rank 42
    # (Re-use the baseline construction from the original code)
    rank_42 = 42
    U_base = np.zeros((n * m, rank_42), dtype=np.float64)
    V_base = np.zeros((m * p, rank_42), dtype=np.float64)
    W_base = np.zeros((p * n, rank_42), dtype=np.float64)
    
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1}
    ]
    
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1}
    ]
    
    W_contrib = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1}
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U_base[i * m + j, col] += coeff
                
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V_base[j * p + k, col] += coeff
                
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W_base[k * n + i, col] += coeff
                
                col += 1
    
    return {"U": U_base.tolist(), "V": V_base.tolist(), "W": W_base.tolist()}

# EVOLVE-BLOCK-END
