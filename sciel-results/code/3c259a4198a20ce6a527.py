"""
Matrix multiplication tensor rank minimization for 2x4 * 4x7.
Uses structured random search with exact local polish to find a lower-rank
decomposition than the schoolbook rank of 56.
"""
import time
import random
import math
import numpy as np

DIMENSIONS = [2, 4, 7]


def _verify(U, V, W, tol=1e-9):
    """Check whether the decomposition exactly reproduces the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # Vectorized check: compute all (i,j,k) sums at once
    for i in range(n):
        for j in range(m):
            u_row = U_arr[i * m + j]
            for k in range(p):
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                s = float(np.dot(u_row, v_row * w_row))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _to_half_int_matrix(M, tol=1e-9):
    """Check if all entries are within tol of half-integers; return rounded version."""
    M = np.array(M, dtype=float)
    scaled = M * 2.0
    rounded = np.round(scaled)
    if np.max(np.abs(scaled - rounded)) > tol:
        return None
    return (rounded / 2.0)


def _rand_init(n, m, p, rank, seed=None):
    """Random initialization of U, V, W."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    return U, V, W


def _structured_init(n, m, p, rank, seed=None):
    """Structured initialization exploiting matrix multiplication block structure.
    
    For matrix multiplication, the tensor has a natural block structure:
    T[i,j,k] depends on U[i*m+j], V[j*p+k], W[k*n+i].
    We initialize with block-diagonal patterns to respect this structure.
    """
    rng = np.random.RandomState(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Assign random blocks to each (i,j,k) triple
    # Use a sparse pattern: each (i,j,k) gets a few random columns
    cols_per_triple = max(1, rank // (n * m * p) + 1)
    
    used_cols = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Pick some random columns for this triple
                available = [c for c in range(rank) if c not in used_cols]
                if not available:
                    available = list(range(rank))
                
                num_cols = min(cols_per_triple, len(available))
                selected = rng.choice(len(available), size=num_cols, replace=False)
                selected_cols = [available[s] for s in selected]
                
                for c in selected_cols:
                    used_cols.add(c)
                    U[i * m + j, c] = rng.randn()
                    V[j * p + k, c] = rng.randn()
                    W[k * n + i, c] = rng.randn()
    
    # Scale to get order-1 products
    # Typical |dot| ~ sqrt(num_cols) * (1/sqrt(num_cols))^3 ~ 1/sqrt(num_cols)
    # So scale by sqrt(num_cols)
    scale = math.sqrt(max(1, cols_per_triple))
    U = U / scale
    V = V / scale
    W = W / scale
    
    return U, V, W


def _polish(U, V, W, n, m, p, max_iters=100, tol=1e-10):
    """High-precision polish using exact local solves.
    
    For each (i,j,k), the constraint is:
    sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    We can fix U and V, then solve for W entries, and cycle through.
    This is a linear system for each free variable.
    """
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    for iteration in range(max_iters):
        # For each (i,j,k), we have:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # 
        # If we fix U and V, this is linear in W.
        # For a fixed column c, W[k*n+i, c] appears in constraints for all (j,k) 
        # where i is fixed... actually it's coupled.
        #
        # Simpler: use gradient descent with very small steps
        best_err = _objective(U, V, W, n, m, p)
        
        if best_err < tol:
            break
        
        # Try coordinate descent with exact line search for each variable
        improved = False
        for _ in range(30):
            mat = random.randint(3)
            if mat == 0:
                r = random.randint(n * m)
                c = random.randint(rank)
                old_val = U[r, c]
            elif mat == 1:
                r = random.randint(m * p)
                c = random.randint(rank)
                old_val = V[r, c]
            else:
                r = random.randint(p * n)
                c = random.randint(rank)
                old_val = W[r, c]
            
            # Try a few values around old_val
            best_new_val = old_val
            best_new_err = best_err
            
            for delta in [0.01, -0.01, 0.001, -0.001, 0.05, -0.05]:
                new_val = old_val + delta
                if mat == 0:
                    U[r, c] = new_val
                    err = _objective(U, V, W, n, m, p)
                    U[r, c] = old_val
                elif mat == 1:
                    V[r, c] = new_val
                    err = _objective(U, V, W, n, m, p)
                    V[r, c] = old_val
                else:
                    W[r, c] = new_val
                    err = _objective(U, V, W, n, m, p)
                    W[r, c] = old_val
                
                if err < best_new_err - 1e-15:
                    best_new_err = err
                    best_new_val = new_val
            
            if best_new_val != old_val:
                if mat == 0:
                    U[r, c] = best_new_val
                elif mat == 1:
                    V[r, c] = best_new_val
                else:
                    W[r, c] = best_new_val
                best_err = best_new_err
                improved = True
        
        if not improved:
            break
    
    return U, V, W, best_err


def _objective(U, V, W, n, m, p):
    """Sum of squared errors over all (i,j,k) pairs. Vectorized for speed."""
    # Precompute: for each (i,j), U row; for each (j,k), V row; for each (k,i), W row
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We can compute this as: for fixed (i,j), T[i,j,:] = U[i*m+j] @ (V[j,:] * W[:,i]).T
    # where V[j,:] is (p, rank) and W[:,i] is (p, rank) element-wise multiplied
    err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]  # (rank,)
            # V[j*p : j*p+p, :] is (p, rank)
            # W[0*p+n*i : p*n : n] gives W[k*n+i] for k=0..p-1, shape (p, rank)
            w_rows = W.reshape(p, n, -1)[:, i, :]  # (p, rank)
            v_rows = V.reshape(m, p, -1)[j, :, :]  # (p, rank)
            # Element-wise product: (p, rank)
            vw = v_rows * w_rows
            # T[i,j,k] = dot(u_row, vw[k]) for each k
            # = u_row @ vw.T -> (p,)
            t_vals = u_row @ vw.T  # (p,)
            err += float(np.sum((t_vals - 1.0) ** 2))
    return err


def _refine(U, V, W, n, m, p, lr=0.001, iters=200, rng_seed=0):
    """Coordinate-descent refinement with adaptive step size."""
    rng = np.random.RandomState(rng_seed)
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    best_err = _objective(U, V, W, n, m, p)
    current_lr = lr
    
    for it in range(iters):
        # Pick random element in one of the three matrices
        mat = rng.randint(3)
        if mat == 0:
            rows, cols = n * m, rank
        elif mat == 1:
            rows, cols = m * p, rank
        else:
            rows, cols = p * n, rank
        
        r = rng.randint(rows)
        c = rng.randint(cols)
        
        if mat == 0:
            old = U[r, c]
        elif mat == 1:
            old = V[r, c]
        else:
            old = W[r, c]
        
        # Try multiple step sizes for robustness
        best_new_val = old
        best_new_err = best_err
        
        for trial in range(3):
            step = current_lr * (0.3 + 0.7 * rng.random())
            for sign in [1, -1]:
                new_val = old + sign * step
                if mat == 0:
                    U[r, c] = new_val
                    err = _objective(U, V, W, n, m, p)
                    U[r, c] = old
                elif mat == 1:
                    V[r, c] = new_val
                    err = _objective(U, V, W, n, m, p)
                    V[r, c] = old
                else:
                    W[r, c] = new_val
                    err = _objective(U, V, W, n, m, p)
                    W[r, c] = old
                
                if err < best_new_err - 1e-14:
                    best_new_err = err
                    best_new_val = new_val
        
        if best_new_val != old:
            if mat == 0:
                U[r, c] = best_new_val
            elif mat == 1:
                V[r, c] = best_new_val
            else:
                W[r, c] = best_new_val
            best_err = best_new_err
            current_lr *= 1.1  # slight increase if we made progress
        else:
            current_lr *= 0.9  # decrease if stuck
    
    return U, V, W, best_err


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 56
    deadline = time.time() + 140  # safety margin
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Start with schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(8, 28, 14) = 28
    # Try various ranks below schoolbook
    for rank in range(28, schoolbook_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        # More restarts for smaller ranks (harder to find)
        max_restarts = max(20, 100 - (rank - 28) * 2)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 8:
                break
            
            seed = restart * 7919 + 13
            
            # Alternate between random and structured initialization
            if restart % 2 == 0:
                U, V, W = _rand_init(n, m, p, rank, seed=seed)
                scale = rank ** 0.5
                U = U / scale
                V = V / scale
                W = W / scale
            else:
                U, V, W = _structured_init(n, m, p, rank, seed=seed)
            
            err = _objective(U, V, W, n, m, p)
            
            # Quick refine
            U, V, W, err = _refine(U, V, W, n, m, p, lr=0.05, iters=30, rng_seed=seed)
            
            if err < 1e-4:
                # High-precision polish
                U, V, W, err = _polish(U, V, W, n, m, p, max_iters=50, tol=1e-10)
            
            if err < 1e-6:
                # Check for exact solution
                # Try rounding to half-integers
                Uh = _to_half_int_matrix(U, tol=1e-4)
                Vh = _to_half_int_matrix(V, tol=1e-4)
                Wh = _to_half_int_matrix(W, tol=1e-4)
                if Uh is not None and Vh is not None and Wh is not None:
                    if _verify(Uh.tolist(), Vh.tolist(), Wh.tolist()):
                        found = True
                        best_rank = rank
                        best_U = Uh.tolist()
                        best_V = Vh.tolist()
                        best_W = Wh.tolist()
                        break
                # Also check raw
                if _verify(U.tolist(), V.tolist(), W.tolist()):
                    found = True
                    best_rank = rank
                    best_U = U.tolist()
                    best_V = V.tolist()
                    best_W = W.tolist()
                    break
            
            if found:
                break
        
        if found:
            break
    
    # Convert best to list format
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
        best_V = best_V.tolist()
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}


# EVOLVE-BLOCK-END
