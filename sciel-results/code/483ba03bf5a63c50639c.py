"""Matrix multiplication 3,5,7 - rank-minimal tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute index arrays for all (i,j,k) triples
    i_idx = np.array([i for i in range(n) for j in range(m) for k in range(p)], dtype=int)
    j_idx = np.array([j for i in range(n) for j in range(m) for k in range(p)], dtype=int)
    k_idx = np.array([k for i in range(n) for j in range(m) for k in range(p)], dtype=int)
    
    U_idx = (i_idx * m + j_idx).astype(int)  # index into U rows
    V_idx = (j_idx * p + k_idx).astype(int)  # index into V rows
    W_idx = (k_idx * n + i_idx).astype(int)  # index into W rows
    
    num_constraints = len(i_idx)  # n*m*p = 105

    def compute_tensor(U, V, W):
        """Compute the decomposed tensor values for all (i,j,k)."""
        vals = np.zeros(num_constraints)
        for r in range(U.shape[1]):
            vals += U[U_idx, r] * V[V_idx, r] * W[W_idx, r]
        return vals

    def is_valid(U, V, W, tol=1e-8):
        """Check if decomposition is valid."""
        vals = compute_tensor(U, V, W)
        return np.all(np.abs(vals - 1.0) < tol)

    def objective(U, V, W):
        """Sum of squared errors."""
        vals = compute_tensor(U, V, W)
        errors = vals - 1.0
        return np.sum(errors ** 2)

    # Trivial rank-105 decomposition (always valid)
    rank_trivial = nm * p  # = 105
    U = np.zeros((nm, rank_trivial), dtype=float)
    V = np.zeros((mp, rank_trivial), dtype=float)
    W = np.zeros((pn, rank_trivial), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0

    best_U, best_V, best_W = U, V, W
    best_rank = rank_trivial
    best_obj = 0.0

    def try_gradient_descent(r, seed=42, max_iter=5000, lr_init=0.05):
        """Try to find a valid decomposition of rank r using gradient descent with adaptive learning rate."""
        nonlocal best_U, best_V, best_W, best_rank, best_obj
        
        if time.time() > deadline - 20:
            return False
        
        np.random.seed(seed)
        
        # Smart initialization: start from a partial structure
        # Initialize with small random values
        Ur = np.random.randn(nm, r) * 0.3
        Vr = np.random.randn(mp, r) * 0.3
        Wr = np.random.randn(pn, r) * 0.3
        
        # Alternative: initialize some columns to give structure
        # For each term, assign random (i,j,k) and set corresponding entries
        for rcol in range(r):
            i0 = np.random.randint(n)
            j0 = np.random.randint(m)
            k0 = np.random.randint(p)
            Ur[i0 * m + j0, rcol] = np.random.choice([-1.0, 1.0])
            Vr[j0 * p + k0, rcol] = np.random.choice([-1.0, 1.0])
            Wr[k0 * n + i0, rcol] = np.random.choice([-1.0, 1.0])
        
        obj = objective(Ur, Vr, Wr)
        
        lr = lr_init
        patience = 0
        best_local_obj = obj
        best_local_U, best_local_V, best_local_W = Ur.copy(), Vr.copy(), Wr.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Compute current tensor values
            vals = np.zeros(num_constraints)
            for rcol in range(r):
                vals += Ur[U_idx, rcol] * Vr[V_idx, rcol] * Wr[W_idx, rcol]
            
            errors = vals - 1.0
            obj = np.sum(errors ** 2)
            
            if obj < best_local_obj:
                best_local_obj = obj
                best_local_U, best_local_V, best_local_W = Ur.copy(), Vr.copy(), Wr.copy()
                patience = 0
            else:
                patience += 1
            
            if patience > 500:
                # Reset to best local and try smaller lr
                Ur, Vr, Wr = best_local_U.copy(), best_local_V.copy(), best_local_W.copy()
                lr *= 0.5
                patience = 0
                if lr < 1e-6:
                    break
            
            # Check convergence
            if obj < 1e-10:
                if is_valid(Ur, Vr, Wr, tol=1e-8):
                    best_U, best_V, best_W = Ur.copy(), Vr.copy(), Wr.copy()
                    best_rank = r
                    best_obj = obj
                    return True
                break
            
            # Compute gradients
            dUr = np.zeros((nm, r))
            dVr = np.zeros((mp, r))
            dWr = np.zeros((pn, r))
            
            for rcol in range(r):
                u_col = Ur[:, rcol]
                v_col = Vr[:, rcol]
                w_col = Wr[:, rcol]
                
                # Gradient w.r.t. U: for each row index, sum over constraints where U_idx[c] = row
                # dUr[row, rcol] = sum_{c: U_idx[c]=row} errors[c] * Vr[V_idx[c], rcol] * Wr[W_idx[c], rcol]
                # Vectorized: for each constraint c, contribution to dUr[U_idx[c], rcol] is
                # errors[c] * Vr[V_idx[c], rcol] * Wr[W_idx[c], rcol]
                
                contrib_U = errors * v_col[V_idx] * w_col[W_idx]
                np.add.at(dUr, (U_idx, rcol), contrib_U)
                
                contrib_V = errors * u_col[U_idx] * w_col[W_idx]
                np.add.at(dVr, (V_idx, rcol), contrib_V)
                
                contrib_W = errors * u_col[U_idx] * v_col[V_idx]
                np.add.at(dWr, (W_idx, rcol), contrib_W)
            
            # Update
            Ur -= lr * dUr
            Vr -= lr * dVr
            Wr -= lr * dWr
        
        # Final check with best local
        if best_local_obj < 1e-6 and is_valid(best_local_U, best_local_V, best_local_W, tol=1e-6):
            best_U, best_V, best_W = best_local_U.copy(), best_local_V.copy(), best_local_W.copy()
            best_rank = r
            best_obj = best_local_obj
            return True
        
        # Update global best if this was better
        if best_local_obj < best_obj:
            best_U, best_V, best_W = best_local_U.copy(), best_local_V.copy(), best_local_W.copy()
            best_obj = best_local_obj
        
        return False

    # Try structured decomposition: exploit the structure of matrix multiplication
    # Key insight: The tensor T[i,j,k] = 1 has a special structure.
    # For fixed i: sum_j,k terms where W[k*n+i, r] couples j and k through V.
    # 
    # Alternative approach: Use the fact that for matrix multiplication,
    # we can use a "block" structure. For each i, the subproblem involves
    # all j,k pairs. 
    #
    # Let's try a different strategy: parameterize using the structure.
    # T[i,j,k] = sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r = 1
    #
    # One known approach: use rank = n*p + m*p + n*m - something (Strassen-like)
    # For 3x5x7, let's try various ranks.

    # Try ranks from high to low
    ranks_to_try = [100, 90, 80, 70, 60, 55, 50, 45, 40, 38, 35]
    
    for r_target in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in [42, 123, 7, 999, 314, 271, 555, 888]:
            if time.time() > deadline - 25:
                break
            
            success = try_gradient_descent(r_target, seed=seed, max_iter=3000, lr_init=0.05)
            if success:
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try lower
            continue
        else:
            # Didn't find at this rank, try next (lower) rank
            pass

    # If we haven't improved, try a more structured approach
    if best_rank == rank_trivial:
        # Try a specific structured decomposition
        # For matrix multiplication, one can use the "outer product" structure
        # Let's try rank = n*m + m*p + n*p - n - m - p + 1 type constructions
        
        # Actually, let's try a different parameterization:
        # Set U[i*m+j, r] = a_{i,j,r}, V[j*p+k, r] = b_{j,k,r}, W[k*n+i, r] = c_{k,i,r}
        # We need sum_r a_{i,j,r} b_{j,k,r} c_{k,i,r} = 1 for all i,j,k
        
        # Try rank = m + n*p or similar
        for r_target in [35, 30, 25, 21, 20, 15, 12]:
            if time.time() > deadline - 20:
                break
            
            for seed in [42, 123, 7, 999, 314]:
                if time.time() > deadline - 15:
                    break
                
                success = try_gradient_descent(r_target, seed=seed, max_iter=5000, lr_init=0.1)
                if success:
                    break

    # Final polish: if we have a near-valid solution, try to refine it
    if best_obj < 1e-4 and best_obj > 0:
        # Try to polish with more iterations
        Ur, Vr, Wr = best_U.copy(), best_V.copy(), best_W.copy()
        lr = 0.01
        for it in range(5000):
            if time.time() > deadline - 5:
                break
            
            vals = np.zeros(num_constraints)
            for rcol in range(Ur.shape[1]):
                vals += Ur[U_idx, rcol] * Vr[V_idx, rcol] * Wr[W_idx, rcol]
            
            errors = vals - 1.0
            obj = np.sum(errors ** 2)
            
            if obj < 1e-12:
                break
            
            dUr = np.zeros_like(Ur)
            dVr = np.zeros_like(Vr)
            dWr = np.zeros_like(Wr)
            
            for rcol in range(Ur.shape[1]):
                u_col = Ur[:, rcol]
                v_col = Vr[:, rcol]
                w_col = Wr[:, rcol]
                
                contrib_U = errors * v_col[V_idx] * w_col[W_idx]
                np.add.at(dUr, (U_idx, rcol), contrib_U)
                
                contrib_V = errors * u_col[U_idx] * w_col[W_idx]
                np.add.at(dVr, (V_idx, rcol), contrib_V)
                
                contrib_W = errors * u_col[U_idx] * v_col[V_idx]
                np.add.at(dWr, (W_idx, rcol), contrib_W)
            
            Ur -= lr * dUr
            Vr -= lr * dVr
            Wr -= lr * dWr
        
        if is_valid(Ur, Vr, Wr, tol=1e-8):
            best_U, best_V, best_W = Ur, Vr, Wr
            best_obj = 0.0

    # Return the best solution found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
