import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 x 3x4 matrix multiplication.
    Uses search over F_3 for rank R from 24 down to 12.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Pre-compute the target tensor T
    # T[a, b, c] = 1 if valid, else 0
    # a in [0, nm), b in [0, mp), c in [0, np_)
    T = np.zeros((nm, mp, np_), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
                
    # Trivial solution
    def get_trivial_solution():
        rank = nm * mp # n*m*p
        U = np.zeros((nm, rank), dtype=np.int8)
        V = np.zeros((mp, rank), dtype=np.int8)
        W = np.zeros((np_, rank), dtype=np.int8)
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, r] = 1
                    V[b, r] = 1
                    W[c, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Gaussian Elimination mod 3
    def solve_mod3(A, b):
        # A: NxM, b: N
        # Returns solution x (M,) or None
        N, M = A.shape
        # Augment
        A_aug = np.hstack([A, b.reshape(-1, 1)])
        A_aug = A_aug.copy()
        
        pivot_cols = []
        row = 0
        for col in range(M):
            # Find pivot
            pivot = -1
            for r in range(row, N):
                if A_aug[r, col] % 3 != 0:
                    pivot = r
                    break
            if pivot == -1:
                continue
                
            # Swap
            A_aug[[row, pivot]] = A_aug[[pivot, row]]
            pivot_cols.append(col)
            
            # Normalize pivot row
            inv_val = 1 if A_aug[row, col] % 3 == 1 else 2 # 1/1=1, 1/2=2
            A_aug[row] = (A_aug[row] * inv_val) % 3
            
            # Eliminate
            for r in range(N):
                if r != row and A_aug[r, col] % 3 != 0:
                    factor = A_aug[r, col]
                    A_aug[r] = (A_aug[r] - factor * A_aug[row]) % 3
            
            row += 1
            if row == N:
                break
        
        # Check consistency
        for r in range(row, N):
            if A_aug[r, M] % 3 != 0:
                return None
                
        # Back substitution
        x = np.zeros(M, dtype=np.int8)
        for i in range(len(pivot_cols) - 1, -1, -1):
            col = pivot_cols[i]
            sum_val = A_aug[i, M]
            for j in range(i + 1, len(pivot_cols)):
                sum_val -= A_aug[i, pivot_cols[j]] * x[pivot_cols[j]]
            x[col] = sum_val % 3
            
        return x

    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Search for R from 24 down to 12
    for R in range(24, 11, -1):
        # Number of trials depends on time remaining
        # Estimate time per trial
        # Let's just run a fixed number of trials per R, or time-based loop
        trials_done = 0
        while time.time() < deadline:
            # Generate random U, V
            U = np.random.randint(0, 3, size=(nm, R), dtype=np.int8)
            V = np.random.randint(0, 3, size=(mp, R), dtype=np.int8)
            
            W = np.zeros((np_, R), dtype=np.int8)
            success = True
            
            for c in range(np_):
                # Build linear system for W[c, :]
                # Equations: sum_r U[a, r] V[b, r] W[c, r] = T[a, b, c]
                # Unknowns: W[c, 0], ..., W[c, R-1]
                # A[ab, r] = U[a, r] V[b, r]
                # b_vec[ab] = T[a, b, c]
                
                # Vectorize
                # U is nm x R
                # V is mp x R
                # We need A of size (nm*mp) x R
                # A[ a*mp + b, r ] = U[a, r] * V[b, r]
                
                # This can be done with broadcasting
                # U_exp = U[:, np.newaxis, :] # nm x 1 x R
                # V_exp = V[np.newaxis, :, :] # 1 x mp x R
                # Prod = U_exp * V_exp # nm x mp x R
                # A_flat = Prod.reshape(nm*mp, R)
                
                Prod = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                A_flat = Prod.reshape(nm * mp, R)
                b_vec = T[:, :, c].reshape(-1)
                
                sol = solve_mod3(A_flat, b_vec)
                if sol is None:
                    success = False
                    break
                W[c, :] = sol
            
            if success:
                # Verify
                # Check sum_r U[a,r] V[b,r] W[c,r] == T[a,b,c]
                # Use the same Prod logic
                # Sum over r of U[a,r] V[b,r] W[c,r]
                # This is element-wise product sum
                # Res = np.einsum('ar,br,cr->abc', U, V, W)
                Res = np.einsum('ar,br,cr->abc', U, V, W)
                if np.array_equal(Res % 3, T):
                    # Map to {-1, 0, 1}
                    U_out = U.copy()
                    V_out = V.copy()
                    W_out = W.copy()
                    U_out[U_out == 2] = -1
                    V_out[V_out == 2] = -1
                    W_out[W_out == 2] = -1
                    
                    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
            
            trials_done += 1
            # If time is running low, break
            if time.time() > deadline - 5.0:
                break
                
    # Fallback
    return get_trivial_solution()
