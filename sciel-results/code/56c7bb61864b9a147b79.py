import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    np_ = n * p
    
    # Target tensor values
    # T[i, j, jp, k, ip, kp]
    # We only need to check non-zero entries for the objective?
    # No, we need all entries to be 0 where T is 0.
    # But we can just compute the full tensor product and compare.
    
    def residuals(x):
        # x is flattened U, V, W
        size_U = nm * len(x) // (nm + mp + np_) # This is wrong, we need to know rank
        # We pass rank as a closure or argument
        pass

    # We need to define the function inside a loop over ranks.
    
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((np_, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1

    # Try to find better rank
    for rank in range(best_rank - 1, 30, -1):
        if time.time() > deadline:
            break
        
        def objective(x):
            U = x[:nm*rank].reshape(nm, rank)
            V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
            W = x[nm*rank+mp*rank:].reshape(np_, rank)
            
            # Compute sum_c U_c V_c W_c
            # U_c is (nm,), V_c is (mp,), W_c is (np_)
            # We need to form the tensor T_calc of shape (n, m, m, p, n, p)
            # T_calc[i, j, jp, k, ip, kp] = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[ip*n+kp, c]
            
            # Efficient computation:
            # Reshape U to (n, m, rank)
            # Reshape V to (m, p, rank)
            # Reshape W to (n, p, rank)
            
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(n, p, rank)
            
            # Sum over c
            # We can use einsum
            # T_calc[i, j, jp, k, ip, kp] = sum_c U[i, j, c] V[jp, k, c] W[ip, kp, c]
            # Note: V index is (jp, k). In V_reshaped, it is (m, p, rank).
            # So V[jp, k, c] is correct.
            
            T_calc = np.einsum('ijc,jkc,ikc->ijj kik', U_reshaped, V_reshaped, W_reshaped)
            # Wait, the output indices are i, j, jp, k, ip, kp.
            # U is (i, j, c)
            # V is (jp, k, c) -- wait, V dimension is m*p. Index is j'*p+k.
            # So V_reshaped is (m, p, rank). Index (jp, k) corresponds to V_reshaped[jp, k, c].
            # W is (ip, kp, c).
            
            # The target T is non-zero only when i==ip and j==jp.
            # We can just return the difference for all indices.
            
            # Flatten T_calc
            return T_calc.flatten() - T_target.flatten()

        # Prepare T_target
        T_target = np.zeros((n, m, m, p, n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i, j, j, k, i, k] = 1.0

        # Initial guess: random
        x0 = np.random.randn(nm*rank + mp*rank + np_*rank) * 0.1
        
        try:
            result = least_squares(objective, x0, method='trf', max_nfev=100)
            if result.success and result.cost < 1e-6:
                # Check if we can round to half-integers
                U_sol = result.x[:nm*rank].reshape(nm, rank)
                V_sol = result.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W_sol = result.x[nm*rank+mp*rank:].reshape(np_, rank)
                
                # Round to nearest 0.5
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_sol * 2) / 2
                
                # Verify
                U_r = U_round.reshape(n, m, rank)
                V_r = V_round.reshape(m, p, rank)
                W_r = W_round.reshape(n, p, rank)
                T_calc = np.einsum('ijc,jkc,ikc->ijj kik', U_r, V_r, W_r)
                
                if np.allclose(T_calc, T_target, atol=1e-9):
                    best_rank = rank
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    # Break if we found a very low rank?
                    # Or continue to find even lower?
                    # Let's break to save time, as lower ranks are harder.
                    break
        except Exception:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
