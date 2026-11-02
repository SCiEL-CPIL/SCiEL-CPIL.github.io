"""Matrix multiplication 3x5x7 - rank minimization via ALS and lattice snapping."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Vectorized check
    # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # This is a 3-way contraction.
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # Sum over rank
    # We want an array of shape (n, m, p) where entry [i,j,k] is the sum.
    # np.einsum is perfect here.
    # 'ijc,jkc,kic->ijc'? No, output should be indexed by i,j,k.
    # U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Sum over c.
    # Result[i,j,k]
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    except Exception:
        return False
        
    # Check if all entries are close to 1
    if not np.allclose(result, 1.0, atol=1e-10):
        return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using fractions."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target = Fraction(1)
                val = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(U[i*m+j][c])
                    v_val = Fraction(V[j*p+k][c])
                    w_val = Fraction(W[k*n+i][c])
                    val += u_val * v_val * w_val
                if val != target:
                    return False
    return True

def _snap_to_half_integer(arr):
    """Snap an array to the nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _calculate_error(U, V, W, n, m, p):
    """Calculate the sum of squared errors."""
    U_t = np.array(U, dtype=np.float64).reshape(n, m, -1)
    V_t = np.array(V, dtype=np.float64).reshape(m, p, -1)
    W_t = np.array(W, dtype=np.float64).reshape(p, n, -1)
    rank = U_t.shape[2]
    
    # Compute the reconstructed tensor
    # result[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    error = np.sum((result - 1.0)**2)
    return error

def construct():
    """Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n  # 15, 35, 21
    
    # Schoolbook baseline: rank 105
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p

    def als_decompose(r, max_iter=50, tol=1e-8):
        """
        Perform ALS to find a real-valued rank-r decomposition.
        Returns U, V, W as numpy arrays of shape (nm, r), (mp, r), (pn, r).
        """
        # Initialize with random values
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Reshape for easier manipulation
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        
        prev_error = float('inf')
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            # For each i, j, we want to minimize sum_k (T[i,j,k] - sum_c U[i,j,c] V[j,k,c] W[k,i,c])^2
            # This is a linear system for U[i,j,:].
            # Let A[i,j] be a matrix where A[i,j]_{c, k} = V[j,k,c] * W[k,i,c]
            # Then U[i,j,:] should be the solution to A[i,j] @ U[i,j,:] = 1 (since T=1)
            
            # We can vectorize this.
            # V_t: (m, p, r) -> transpose to (m, r, p)?
            # W_t: (p, n, r) -> transpose to (n, p, r)?
            
            V_t = V.reshape(m, p, r)
            W_t = W.reshape(p, n, r)
            
            # We need for each i, j:
            # A_{c, k} = V_t[j, k, c] * W_t[k, i, c]
            # Note: V_t[j,k,c] is V[j*p+k, c]. W_t[k,i,c] is W[k*n+i, c].
            
            # Let's build A for all i, j at once?
            # A[i, j, c, k] = V_t[j, k, c] * W_t[k, i, c]
            # We want to solve A[i, j] @ U[i, j] = 1
            
            # Using einsum to build A
            # V_t is (m, p, r). W_t is (p, n, r).
            # We want A[i, j, c, k] where i in 0..n-1, j in 0..m-1, c in 0..r-1, k in 0..p-1.
            # V_t[j, k, c] -> 'jk c'
            # W_t[k, i, c] -> 'k i c'
            # Product: 'jk c, k i c -> i j c k'
            
            try:
                A = np.einsum('jkc,kic->ijck', V_t, W_t)
            except:
                return U, V, W
                
            # A shape: (n, m, r, p)
            # We want to solve A[i, j] @ U[i, j] = 1
            # A[i, j] is (r, p). U[i, j] is (r,).
            # Wait, the sum is over c.
            # sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c])
            # This is U[i, j, :] dot (V_t[j, :, :] * W_t[:, i, :]).T ?
            # Let's check indices.
            # Term for fixed i, j, k:
            # sum_c U[i, j, c] V[j, k, c] W[k, i, c]
            # Let M[i, j, k] = sum_c V[j, k, c] W[k, i, c]
            # Then we want sum_c U[i, j, c] M[i, j, k, c]? No.
            # The sum is over c.
            # So for fixed i, j, the equation for each k is:
            # sum_c U[i, j, c] * (V[j, k, c] * W[k, i, c]) = 1
            # Let B[i, j, k, c] = V[j, k, c] * W[k, i, c]
            # Then sum_c U[i, j, c] B[i, j, k, c] = 1
            # This is a linear system: B[i, j, k, :] @ U[i, j, :] = 1
            # B[i, j] is a (p, r) matrix.
            # U[i, j] is a (r,) vector.
            # We want to solve B[i, j] @ U[i, j] = 1 (vector of ones of length p)
            
            # B[i, j, k, c] = V_t[j, k, c] * W_t[k, i, c]
            # einsum: 'jkc, kic -> i j k c'
            try:
                B = np.einsum('jkc,kic->ijkc', V_t, W_t)
            except:
                return U, V, W
                
            # B shape: (n, m, p, r)
            # We want to solve B[i, j] @ U[i, j] = 1
            # B[i, j] is (p, r). U[i, j] is (r,).
            # Use np.linalg.lstsq
            U_new = np.zeros((n, m, r))
            for i in range(n):
                for j in range(m):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(B[i, j], np.ones(p), rcond=None)
                        U_new[i, j] = sol
                    except:
                        U_new[i, j] = U[i*m+j, :]
                        
            U = U_new.reshape(nm, r)
            
            # Update V
            # Similarly, for fixed j, k:
            # sum_c V[j, k, c] * (U[i, j, c] * W[k, i, c]) = 1
            # Let C[j, k, i, c] = U[i, j, c] * W[k, i, c]
            # sum_c V[j, k, c] C[j, k, i, c] = 1
            # C[j, k] is (n, r). V[j, k] is (r,).
            # Solve C[j, k] @ V[j, k] = 1 (vector of ones of length n)
            
            U_t = U.reshape(n, m, r)
            W_t = W.reshape(p, n, r)
            try:
                C = np.einsum('ijc,kic->jkic', U_t, W_t)
            except:
                return U, V, W
            # C shape: (m, p, n, r)
            
            V_new = np.zeros((m, p, r))
            for j in range(m):
                for k in range(p):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(C[j, k], np.ones(n), rcond=None)
                        V_new[j, k] = sol
                    except:
                        V_new[j, k] = V[j*p+k, :]
                        
            V = V_new.reshape(mp, r)
            
            # Update W
            # For fixed k, i:
            # sum_c W[k, i, c] * (U[i, j, c] * V[j, k, c]) = 1
            # Let D[k, i, j, c] = U[i, j, c] * V[j, k, c]
            # sum_c W[k, i, c] D[k, i, j, c] = 1
            # D[k, i] is (m, r). W[k, i] is (r,).
            # Solve D[k, i] @ W[k, i] = 1 (vector of ones of length m)
            
            U_t = U.reshape(n, m, r)
            V_t = V.reshape(m, p, r)
            try:
                D = np.einsum('ijc,jkc->kijc', U_t, V_t)
            except:
                return U, V, W
            # D shape: (p, n, m, r)
            
            W_new = np.zeros((p, n, r))
            for k in range(p):
                for i in range(n):
                    try:
                        sol, _, _, _ = np.linalg.lstsq(D[k, i], np.ones(m), rcond=None)
                        W_new[k, i] = sol
                    except:
                        W_new[k, i] = W[k*n+i, :]
                        
            W = W_new.reshape(pn, r)
            
            # Check convergence
            error = _calculate_error(U, V, W, n, m, p)
            if abs(prev_error - error) < tol:
                break
            prev_error = error
            
        return U, V, W

    def try_rank_als(r, num_trials=5):
        """Try to find a decomposition of rank r using ALS + snapping + local search."""
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        random.seed(42 + r)
        np.random.seed(42 + r)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return False
                
            # 1. Real-valued ALS
            U_real, V_real, W_real = als_decompose(r, max_iter=20)
            
            # 2. Snap to half-integers
            U_snap = _snap_to_half_integer(U_real)
            V_snap = _snap_to_half_integer(V_real)
            W_snap = _snap_to_half_integer(W_real)
            
            # 3. Check error
            err = _calculate_error(U_snap, V_snap, W_snap, n, m, p)
            
            # 4. Local Search (Hill Climbing)
            # If error is small, try to fix it by flipping individual entries
            if err < 1.0: # Only worth trying if it's close
                # Max 1000 steps of local search
                for step in range(1000):
                    if time.time() > deadline:
                        return False
                    if err == 0:
                        break
                        
                    # Pick a random matrix and entry
                    mat_id = random.choice([0, 1, 2]) # 0:U, 1:V, 2:W
                    if mat_id == 0:
                        mat = U_snap
                        size = nm
                    elif mat_id == 1:
                        mat = V_snap
                        size = mp
                    else:
                        mat = W_snap
                        size = pn
                        
                    row = random.randint(0, size - 1)
                    col = random.randint(0, r - 1)
                    
                    # Try changing mat[row, col] to nearest other half-integer
                    current_val = mat[row, col]
                    # Neighbors: current_val - 0.5, current_val + 0.5
                    # Also 0 if not 0?
                    neighbors = [current_val - 0.5, current_val + 0.5]
                    # Ensure we don't try the same value
                    best_neighbor_val = None
                    best_neighbor_err = err
                    
                    for nv in neighbors:
                        mat[row, col] = nv
                        new_err = _calculate_error(U_snap, V_snap, W_snap, n, m, p)
                        if new_err < best_neighbor_err:
                            best_neighbor_err = new_err
                            best_neighbor_val = nv
                            
                    if best_neighbor_val is not None:
                        mat[row, col] = best_neighbor_val
                        err = best_neighbor_err
                    else:
                        mat[row, col] = current_val
                        # If no improvement, break or continue?
                        # For speed, break if stuck
                        break
                        
            # 5. Final Verification
            if err == 0:
                # Convert to lists for return
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                if _verify_exact(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    return True
                    
        return False

    # Try ranks from lower to higher
    # Start from a reasonable lower bound
    # Schoolbook is 105.
    # Let's try from 50 up to 104.
    # If we find one, we stop.
    
    # To save time, we can skip ranks that are unlikely to work based on parity or other constraints?
    # For now, just iterate.
    
    # We can also try to use the schoolbook as a starting point for local search?
    # No, ALS is better.
    
    for r in range(50, 105):
        if time.time() > deadline:
            break
        if try_rank_als(r, num_trials=3):
            break
            
    # If we found something, try to go even lower?
    # The loop above breaks on first success.
    # If we found rank 80, we should try 79, 78...
    # But the loop goes 50, 51... so if it fails at 50..79, it tries 80.
    # If it succeeds at 80, it breaks.
    # This is correct for finding the *minimum* in the range [50, 104].
    
    return best_result
