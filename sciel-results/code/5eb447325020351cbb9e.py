"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k.
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _round_to_half_ints(U, V, W):
    """Round entries to nearest half-integer."""
    Ur = [[round(U[i][c] * 2) / 2.0 for c in range(len(U[0]))] for i in range(len(U))]
    Vr = [[round(V[i][c] * 2) / 2.0 for c in range(len(V[0]))] for i in range(len(V))]
    Wr = [[round(W[i][c] * 2) / 2.0 for c in range(len(W[0]))] for i in range(len(W))]
    return Ur, Vr, Wr

def _strassen_4x4x4():
    """
    Returns a rank-49 decomposition for 4x4x4 matrix multiplication using Strassen.
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Strassen coefficients for 2x2 blocks
    # A_coeffs[m][block] where block in 0..3 (00, 01, 10, 11)
    A_coeffs = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00+A01
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    B_coeffs = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01-B11
        [0, 0, 1, -1],# M4: B10-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    C_coeffs = [
        [1, -1, 0, 1, 1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for m_outer in range(7):
        for m_inner in range(7):
            col = m_outer * 7 + m_inner
            
            # U[4*i+j]
            for i in range(4):
                for j in range(4):
                    r = i // 2
                    c = j // 2
                    u_i = i % 2
                    u_j = j % 2
                    b = r * 2 + c
                    d = u_i * 2 + u_j
                    U[4*i+j][col] = A_coeffs[m_outer][b] * A_coeffs[m_inner][d]
            
            # V[4*j+k]
            for j in range(4):
                for k in range(4):
                    c = j // 2
                    d = k // 2
                    v_j = j % 2
                    v_k = k % 2
                    b_B = c * 2 + d
                    d_B = v_j * 2 + v_k
                    V[4*j+k][col] = B_coeffs[m_outer][b_B] * B_coeffs[m_inner][d_B]
            
            # W[4*k+i]
            for k in range(4):
                for i in range(4):
                    r = i // 2
                    d = k // 2
                    w_i = i % 2
                    w_k = k % 2
                    b_C = r * 2 + d
                    d_C = w_i * 2 + w_k
                    W[4*k+i][col] = C_coeffs[m_outer][b_C] * C_coeffs[m_inner][d_C]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _optimize_rank(n, m, p, target_rank, start_time, time_limit, initial_guess=None):
    """
    Attempt to find a valid decomposition of the given rank using L-BFGS-B.
    Returns (U, V, W) as lists of lists if successful, else None.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None

    r = target_rank
    size_u = n * m * r
    size_v = m * p * r
    size_w = p * n * r
    total_size = size_u + size_v + size_w

    if initial_guess is not None:
        x0 = initial_guess
    else:
        # Random initialization
        x0 = np.random.randn(total_size) * 0.5

    def objective(x):
        if time.time() - start_time > time_limit:
            return 1e10
        u_flat = x[:size_u]
        v_flat = x[size_u:size_u + size_v]
        w_flat = x[size_u + size_v:]
        
        U = u_flat.reshape(n, m, r)
        V = v_flat.reshape(m, p, r)
        W = w_flat.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        error = np.sum((T - 1.0) ** 2)
        return error

    # We need gradients for L-BFGS-B to be fast.
    # Gradient of E = sum (T-1)^2 w.r.t U, V, W
    # T_ijk = sum_c U_ijc V_jkc W_kic
    # dE/dU_ijc = 2 * sum_k (T_ijk - 1) * V_jkc * W_kic
    # dE/dV_jkc = 2 * sum_i (T_ijk - 1) * U_ijc * W_kic
    # dE/dW_kic = 2 * sum_j (T_ijk - 1) * U_ijc * V_jkc
    
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros(total_size)
        
        u_flat = x[:size_u]
        v_flat = x[size_u:size_u + size_v]
        w_flat = x[size_u + size_v:]
        
        U = u_flat.reshape(n, m, r)
        V = v_flat.reshape(m, p, r)
        W = w_flat.reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err_tensor = T - 1.0
        
        # Gradients
        # dU_ijc = 2 * sum_k err_ijk V_jkc W_kic
        # Note: V is (m,p,r), W is (p,n,r)
        # We need sum over k.
        # dU = 2 * einsum('ijk,jkc,kic->ijc', err_tensor, V, W)
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', err_tensor, V, W)
        
        # dV_jkc = 2 * sum_i err_ijk U_ijc W_kic
        # dV = 2 * einsum('ijk,ijc,kic->jkc', err_tensor, U, W)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', err_tensor, U, W)
        
        # dW_kic = 2 * sum_j err_ijk U_ijc V_jkc
        # dW = 2 * einsum('ijk,ijc,jkc->kic', err_tensor, U, V)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', err_tensor, U, V)
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        error = np.sum(err_tensor ** 2)
        return error, grad

    try:
        # Use L-BFGS-B with gradient
        res = minimize(
            objective_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 1000, 'maxfun': 10000, 'maxls': 50, 'maxcor': 10}
        )
        
        if res.fun < 1e-6:
            x_opt = res.x
            U_opt = x_opt[:size_u].reshape(n*m, r).tolist()
            V_opt = x_opt[size_u:size_u+size_v].reshape(m*p, r).tolist()
            W_opt = x_opt[size_u+size_v:].reshape(p*n, r).tolist()
            
            # Round to half-integers
            Ur, Vr, Wr = _round_to_half_ints(U_opt, V_opt, W_opt)
            
            # Verify
            if _verify_decomp(n, m, p, Ur, Vr, Wr):
                return Ur, Vr, Wr
            else:
                # If rounding failed, check if unrounded is close enough? 
                # The task requires integer/half-integer.
                # Sometimes L-BFGS lands exactly on integers.
                # Let's try rounding to integers too if half-integers failed?
                # No, half-integers cover integers.
                pass
    except Exception as e:
        pass
        
    return None

def construct():
    deadline = time.time() + 140 # Safety margin for 150s limit
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to improve using Strassen-based initialization
    # Start with rank 53 (49 + 4)
    
    # We will try to optimize from a Strassen-like start.
    # First, let's get the Strassen 4x4x4 decomposition.
    base_444 = _strassen_4x4x4()
    
    # Extend to 4x4x5 by adding 4 columns for the 5th column of B.
    # This gives a valid rank 53 decomposition.
    # We can use this as an initial guess for optimization at rank 53 or lower.
    
    rank_53_U = [row[:] + [0]*4 for row in base_444["U"]]
    rank_53_V = [row[:] + [0]*4 for row in base_444["V"]]
    rank_53_W = [row[:] + [0]*4 for row in base_444["W"]]
    
    # Add the 4 terms for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j] B[j,4]
    # Terms j=0..3
    for j in range(4):
        c = 49 + j
        for i in range(4):
            rank_53_U[4*i+j][c] = 1
            rank_53_V[4*j+4][c] = 1
            rank_53_W[16+i][c] = 1 if i == j else 0
            
    # Verify rank 53
    if _verify_decomp(n, m, p, rank_53_U, rank_53_V, rank_53_W):
        best_result = {"U": rank_53_U, "V": rank_53_V, "W": rank_53_W}
        best_rank = 53
        
    # Now try to optimize for lower ranks.
    # We can use the rank 53 solution as a warm start for rank 52, 51, etc.
    # However, the dimensions change. We can't directly use a 53-vector for a 52-vector.
    # We can truncate? Or just restart with random?
    # Truncation might work if the last few columns are small, but they are not necessarily.
    # Let's try random restarts for lower ranks, and also try truncating the Strassen-based guess.
    
    current_U = rank_53_U
    current_V = rank_53_V
    current_W = rank_53_W
    
    for r in range(52, 48, -1): # Try 52, 51, 50, 49
        if time.time() > deadline - 10:
            break
            
        # Attempt 1: Truncate the previous best (if it was higher rank)
        # This is a heuristic. If the previous solution had small entries in the last columns, this might work.
        # We only have the rank 53 solution.
        # Let's flatten it and take the first r columns?
        # This is risky but worth a try.
        
        U_flat_prev = np.array(current_U).flatten()
        V_flat_prev = np.array(current_V).flatten()
        W_flat_prev = np.array(current_W).flatten()
        
        # We can't just slice the flat array because the structure is (rows, cols).
        # We need to take the first r columns of the matrices.
        U_guess = np.array(current_U)[:, :r]
        V_guess = np.array(current_V)[:, :r]
        W_guess = np.array(current_W)[:, :r]
        
        x0_guess = np.concatenate([U_guess.flatten(), V_guess.flatten(), W_guess.flatten()])
        
        result = _optimize_rank(n, m, p, r, time.time(), deadline, initial_guess=x0_guess)
        if result:
            Ur, Vr, Wr = result
            best_result = {"U": Ur, "V": Vr, "W": Wr}
            best_rank = r
            current_U, current_V, current_W = Ur, Vr, Wr
            # If we find a valid rank, we can try to go lower immediately
            # But let's just continue the loop to try r-1 next.
            
        # Attempt 2: Random restarts if truncation failed
        if best_rank > r:
            for _ in range(3): # 3 random restarts
                if time.time() > deadline - 5:
                    break
                result = _optimize_rank(n, m, p, r, time.time(), deadline, initial_guess=None)
                if result:
                    Ur, Vr, Wr = result
                    best_result = {"U": Ur, "V": Vr, "W": Wr}
                    best_rank = r
                    current_U, current_V, current_W = Ur, Vr, Wr
                    break # Found one, move to next lower rank
    
    return best_result
# EVOLVE-BLOCK-END
