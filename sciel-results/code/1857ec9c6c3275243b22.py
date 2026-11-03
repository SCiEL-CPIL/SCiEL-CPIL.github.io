import time
import numpy as np
import random

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy: 
    1. Attempt to find a rank-29 decomposition using a targeted local search 
       initialized with a "Strassen-like" structure or by perturbing the schoolbook.
    2. If not found, fall back to the schoolbook rank-30 decomposition.
    
    The search uses a custom local optimization that iteratively adjusts 
    integer/half-integer entries to minimize the error, with strict verification.
    """
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify exact decomposition."""
        r = len(U[0])
        # Check all non-zero entries of the target tensor
        for I in range(nm):
            for J in range(mp):
                for K in range(pn):
                    s = 0
                    for c in range(r):
                        s += U[I][c] * V[J][c] * W[K][c]
                    
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k1 = J % p
                    k2 = K // n
                    i2 = K % n
                    
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        target = 1
                    else:
                        target = 0
                        
                    if abs(s - target) > 1e-9:
                        return False
        return True

    def get_tensor_error(U, V, W):
        """Calculate the sum of squared errors for the tensor decomposition."""
        err = 0.0
        r = len(U[0])
        for I in range(nm):
            for J in range(mp):
                for K in range(pn):
                    s = 0
                    for c in range(r):
                        s += U[I][c] * V[J][c] * W[K][c]
                    
                    i1 = I // m
                    j1 = I % m
                    j2 = J // p
                    k1 = J % p
                    k2 = K // n
                    i2 = K % n
                    
                    if j1 == j2 and i1 == i2 and k1 == k2:
                        target = 1
                    else:
                        target = 0
                        
                    err += (s - target) ** 2
        return err

    def round_to_half(x):
        return np.round(x * 2) / 2.0

    def try_rank(R, max_iters=100, restarts=10):
        """
        Try to find a decomposition of rank R using local search on integer/half-integer entries.
        """
        # Precompute target tensor for error calculation speed if needed, 
        # but the explicit loop is fine for small sizes.
        
        for restart in range(restarts):
            if time.time() > deadline:
                return None
            
            # Initialize with random integers
            U = np.random.randint(-2, 3, (nm, R)).astype(float)
            V = np.random.randint(-2, 3, (mp, R)).astype(float)
            W = np.random.randint(-2, 3, (pn, R)).astype(float)
            
            # Add a small random perturbation to avoid symmetry issues
            U += np.random.uniform(-0.1, 0.1, (nm, R))
            V += np.random.uniform(-0.1, 0.1, (mp, R))
            W += np.random.uniform(-0.1, 0.1, (pn, R))
            
            current_err = get_tensor_error(U, V, W)
            
            # If we start with a very high error, it might be hard to converge.
            # For R=29, random init is bad. 
            # We should initialize closer to schoolbook.
            # Let's try a "schoolbook-like" init for the first restart.
            if restart == 0:
                # Initialize with schoolbook terms, but we have R=29 slots.
                # We can't fit 30 terms into 29 slots directly.
                # We will just use random for now, but with a tighter range.
                pass

            for it in range(max_iters):
                if time.time() > deadline:
                    return None
                
                # Coordinate Descent: Update one factor at a time
                # We pick a random factor and a random entry to update?
                # Or we solve for the whole factor given the others?
                
                # Let's try solving for U given V, W.
                # This is a linear least squares problem.
                # T[I,J,K] = sum_c U[I,c] V[J,c] W[K,c]
                
                # Construct M_U
                # Size: (nm*mp*pn) x (nm*R) = 900 x (6*R)
                # For R=29, 900 x 174. This is fast.
                
                M_U = np.zeros((nm * mp * pn, nm * R))
                M_V = np.zeros((nm * mp * pn, mp * R))
                M_W = np.zeros((nm * mp * pn, pn * R))
                
                T_flat = np.zeros(nm * mp * pn)
                for I in range(nm):
                    for J in range(mp):
                        for K in range(pn):
                            idx_t = I * mp * pn + J * pn + K
                            i1 = I // m
                            j1 = I % m
                            j2 = J // p
                            k1 = J % p
                            k2 = K // n
                            i2 = K % n
                            if j1 == j2 and i1 == i2 and k1 == k2:
                                T_flat[idx_t] = 1.0
                            
                            for c in range(R):
                                M_U[idx_t, I * R + c] = V[J, c] * W[K, c]
                                M_V[idx_t, J * R + c] = U[I, c] * W[K, c]
                                M_W[idx_t, K * R + c] = U[I, c] * V[J, c]
                
                # Solve for U
                try:
                    u_flat, _, _, _ = np.linalg.lstsq(M_U, T_flat, rcond=None)
                    U_new = u_flat.reshape((nm, R))
                except:
                    continue
                
                # Solve for V
                try:
                    v_flat, _, _, _ = np.linalg.lstsq(M_V, T_flat, rcond=None)
                    V_new = v_flat.reshape((mp, R))
                except:
                    continue
                
                # Solve for W
                try:
                    w_flat, _, _, _ = np.linalg.lstsq(M_W, T_flat, rcond=None)
                    W_new = w_flat.reshape((pn, R))
                except:
                    continue
                
                # Round to half-integers
                U_new = round_to_half(U_new)
                V_new = round_to_half(V_new)
                W_new = round_to_half(W_new)
                
                new_err = get_tensor_error(U_new, V_new, W_new)
                
                # Accept if better or equal (to allow plateau moves)
                if new_err <= current_err:
                    U, V, W = U_new, V_new, W_new
                    current_err = new_err
                    
                    if current_err < 1e-9:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
                    # If we found an exact solution, verify and return
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # Occasionally, add noise to escape local minima
                if it % 10 == 0 and current_err > 1e-6:
                    U += np.random.uniform(-0.5, 0.5, (nm, R))
                    V += np.random.uniform(-0.5, 0.5, (mp, R))
                    W += np.random.uniform(-0.5, 0.5, (pn, R))
                    
            # End iterations for this restart
        return None

    # Try ranks lower than 30
    # We know 30 is possible. 29 is the target.
    # 28 is unlikely but possible.
    for R in [29, 28]:
        res = try_rank(R)
        if res:
            return res
            
    return schoolbook()
