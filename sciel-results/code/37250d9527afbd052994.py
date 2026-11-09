import time
import numpy as np
import random

def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s kill

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n

    def schoolbook(n, m, p):
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify(U, V, W, n, m, p, tol=1e-10):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def tensor_error(U, V, W, n, m, p):
        R = U.shape[1]
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err += (s - 1.0) ** 2
        return err

    # Search for rank 11 decomposition of 2x3x2
    # n2=2, m2=3, p2=2
    n2, m2, p2 = 2, 3, 2
    R_target = 11
    found_decomp = False
    U_small, V_small, W_small = None, None, None

    # Initialize with schoolbook (rank 12) and remove one term
    # Then optimize
    for attempt in range(100):
        if time.time() > deadline - 20:
            break
            
        # Start with schoolbook 12 terms, zero out one random column
        U_init = np.zeros((n2 * m2, R_target + 1))
        V_init = np.zeros((m2 * p2, R_target + 1))
        W_init = np.zeros((p2 * n2, R_target + 1))
        col = 0
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    U_init[i * m2 + j, col] = 1
                    V_init[j * p2 + k, col] = 1
                    W_init[k * n2 + i, col] = 1
                    col += 1
        
        # Zero out a random column
        rem_idx = random.randint(0, R_target)
        U_init[:, rem_idx] = 0
        V_init[:, rem_idx] = 0
        W_init[:, rem_idx] = 0
        
        # Perturb the rest
        U_curr = U_init.copy()
        V_curr = V_init.copy()
        W_curr = W_init.copy()
        
        # Simple gradient descent on the coefficients
        # We want to minimize sum (sum_r U_ijk,r V_jk,r W_ki,r - 1)^2
        # This is hard to differentiate directly in a simple loop, 
        # but we can use a random perturbation approach (Simulated Annealing or similar)
        
        best_err = tensor_error(U_curr, V_curr, W_curr, n2, m2, p2)
        
        for step in range(500):
            if time.time() > deadline - 10:
                break
            
            # Pick a random variable to perturb
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, n2*m2-1)
            elif mat == 'V':
                row = random.randint(0, m2*p2-1)
            else:
                row = random.randint(0, p2*n2-1)
            
            col_idx = random.randint(0, R_target) # Don't touch the removed column? 
            # Actually, we can touch any column, but the removed one is 0.
            # Let's allow all.
            
            delta = random.uniform(-0.1, 0.1)
            
            if mat == 'U':
                U_curr[row, col_idx] += delta
                err_new = tensor_error(U_curr, V_curr, W_curr, n2, m2, p2)
            elif mat == 'V':
                V_curr[row, col_idx] += delta
                err_new = tensor_error(U_curr, V_curr, W_curr, n2, m2, p2)
            else:
                W_curr[row, col_idx] += delta
                err_new = tensor_error(U_curr, V_curr, W_curr, n2, m2, p2)
            
            if err_new < best_err:
                best_err = err_new
                # Accept
            else:
                # Revert
                if mat == 'U':
                    U_curr[row, col_idx] -= delta
                elif mat == 'V':
                    V_curr[row, col_idx] -= delta
                else:
                    W_curr[row, col_idx] -= delta
        
        # Check if we found a solution
        if best_err < 1e-6:
            # Round to nearest half-integer
            U_rounded = np.round(U_curr * 2) / 2
            V_rounded = np.round(V_curr * 2) / 2
            W_rounded = np.round(W_curr * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded, n2, m2, p2, tol=1e-10):
                U_small = U_rounded
                V_small = V_rounded
                W_small = W_rounded
                found_decomp = True
                break

    if found_decomp:
        # Construct 2x3x8 decomposition
        # Split k (size 8) into 4 groups of 2
        # For each group g (0..3), use the small decomposition
        # U: (2*3, 4*11) = (6, 44)
        # V: (3*8, 4*11) = (24, 44)
        # W: (8*2, 4*11) = (16, 44)
        
        R_total = 4 * R_target
        U_big = np.zeros((nm, R_total))
        V_big = np.zeros((mp, R_total))
        W_big = np.zeros((pn, R_total))
        
        for g in range(4):
            # k indices for this group: 2*g, 2*g+1
            k_start = 2 * g
            
            # U is independent of k, so we repeat the same U for each group?
            # No, U depends on i, j.
            # In the small decomposition, U is (2*3, 11).
            # In the big decomposition, U is (2*3, 44).
            # For group g, the U part is U_small.
            U_big[:, g*R_target:(g+1)*R_target] = U_small
            
            # V depends on j, k.
            # For group g, k is 2*g + k_small (where k_small is 0 or 1).
            # V_small[j, k_small] is at index j*p2 + k_small.
            # V_big[j, 2*g + k_small] should be V_small[j, k_small].
            for j in range(m2):
                for k_small in range(p2):
                    V_big[j*p2 + (2*g + k_small), g*R_target:(g+1)*R_target] = V_small[j, k_small]
            
            # W depends on k, i.
            # W_small[k_small, i] is at index k_small*n2 + i.
            # W_big[2*g + k_small, i] should be W_small[k_small, i].
            for k_small in range(p2):
                for i in range(n2):
                    W_big[(2*g + k_small)*n2 + i, g*R_target:(g+1)*R_target] = W_small[k_small*n2 + i]
        
        # Verify big decomposition
        if verify(U_big, V_big, W_big, n, m, p, tol=1e-10):
            return {"U": U_big.tolist(), "V": V_big.tolist(), "W": W_big.tolist()}
        else:
            # Fallback to schoolbook if verification fails (should not happen if small is valid)
            pass

    # Fallback to schoolbook
    U0, V0, W0 = schoolbook(n, m, p)
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
