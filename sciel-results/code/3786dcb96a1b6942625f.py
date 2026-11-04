"""Matrix multiplication 2x4x8 - rank minimization via structured decomposition."""
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x8 matrix.
    """
    n, m, p = 2, 4, 8
    
    # ============================================================
    # Build the baseline rank-56 Strassen decomposition
    # ============================================================
    def build_strassen_rank56():
        rank = 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                # M1 = (A00 + A11)(B00 + B11)
                U[0 * m + j0, col] += 1.0
                U[1 * m + (j0 + 1), col] += 1.0
                V[j0 * p + k0, col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += 1.0
                W[k0 * n + 0, col] += 1.0
                W[(k0 + 1) * n + 1, col] += 1.0
                col += 1
                
                # M2 = (A10 + A11)B00
                U[1 * m + j0, col] += 1.0
                U[1 * m + (j0 + 1), col] += 1.0
                V[j0 * p + k0, col] += 1.0
                W[k0 * n + 1, col] += 1.0
                W[(k0 + 1) * n + 1, col] += -1.0
                col += 1
                
                # M3 = A00(B01 - B11)
                U[0 * m + j0, col] += 1.0
                V[j0 * p + (k0 + 1), col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += -1.0
                W[(k0 + 1) * n + 0, col] += 1.0
                W[(k0 + 1) * n + 1, col] += 1.0
                col += 1
                
                # M4 = A11(B10 - B00)
                U[1 * m + (j0 + 1), col] += 1.0
                V[(j0 + 1) * p + k0, col] += 1.0
                V[j0 * p + k0, col] += -1.0
                W[k0 * n + 0, col] += 1.0
                W[k0 * n + 1, col] += 1.0
                col += 1
                
                # M5 = (A00 + A01)B11
                U[0 * m + j0, col] += 1.0
                U[0 * m + (j0 + 1), col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += 1.0
                W[k0 * n + 0, col] += -1.0
                W[(k0 + 1) * n + 0, col] += 1.0
                col += 1
                
                # M6 = (A10 - A00)(B00 + B01)
                U[1 * m + j0, col] += 1.0
                U[0 * m + j0, col] += -1.0
                V[j0 * p + k0, col] += 1.0
                V[j0 * p + (k0 + 1), col] += 1.0
                W[(k0 + 1) * n + 1, col] += 1.0
                col += 1
                
                # M7 = (A01 - A11)(B10 + B11)
                U[0 * m + (j0 + 1), col] += 1.0
                U[1 * m + (j0 + 1), col] += -1.0
                V[(j0 + 1) * p + k0, col] += 1.0
                V[(j0 + 1) * p + (k0 + 1), col] += 1.0
                W[k0 * n + 0, col] += 1.0
                col += 1
        
        return U, V, W, rank
    
    U, V, W, rank = build_strassen_rank56()
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # ============================================================
    # Verify decomposition correctness
    # ============================================================
    def verify(U_mat, V_mat, W_mat, trials=5):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rng = np.random.RandomState(42)
        r_max = U_mat.shape[1]
        
        for _ in range(trials):
            A = rng.randint(-10, 10, size=(n, m)).astype(float)
            B = rng.randint(-10, 10, size=(m, p)).astype(float)
            
            # Correct multiplication
            C_correct = A @ B
            
            # Decomposition: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j,r]) * (sum_j B[j,k] * V[j*p+k,r]) * W[k*n+i,r]
            C_decomp = np.zeros((n, p))
            for r in range(r_max):
                u_vals = np.array([np.dot(A[i, :], U_mat[i*m:(i+1)*m, r]) for i in range(n)])
                v_vals = np.array([np.dot(B[:, k], V_mat[k::p, r]) for k in range(p)])
                
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += u_vals[i] * v_vals[k] * W_mat[k*n+i, r]
            
            if np.max(np.abs(C_correct - C_decomp)) > 1e-6:
                return False
        return True
    
    # Verify baseline
    if not verify(U, V, W):
        # Should not happen, but just in case
        pass
    
    # ============================================================
    # Try to reduce rank by finding linear dependencies
    # ============================================================
    def try_reduce_rank(U_mat, V_mat, W_mat, max_iter=100000):
        """Try to reduce rank by finding and merging proportional terms."""
        U_cur = U_mat.copy()
        V_cur = V_mat.copy()
        W_cur = W_mat.copy()
        r_cur = U_cur.shape[1]
        
        for iteration in range(max_iter):
            if _time_left() < 10:
                break
            
            # Look for pairs of columns that are proportional in U and V
            found_merge = False
            for r1 in range(r_cur):
                if _time_left() < 10:
                    break
                for r2 in range(r1 + 1, r_cur):
                    # Check if column r1 is proportional to column r2 in U
                    u1 = U_cur[:, r1]
                    u2 = U_cur[:, r2]
                    
                    # Find ratio from first non-zero element
                    ratio = None
                    for idx in range(len(u1)):
                        if abs(u1[idx]) > 1e-10:
                            if abs(u2[idx]) < 1e-10:
                                ratio = None
                                break
                            ratio = u1[idx] / u2[idx]
                            break
                        elif abs(u2[idx]) > 1e-10:
                            ratio = None
                            break
                    
                    if ratio is None or abs(ratio) < 1e-10:
                        continue
                    
                    # Verify U proportionality
                    if not np.allclose(u1, ratio * u2, atol=1e-9, rtol=1e-9):
                        continue
                    
                    # Check V proportionality
                    v1 = V_cur[:, r1]
                    v2 = V_cur[:, r2]
                    if not np.allclose(v1, ratio * v2, atol=1e-9, rtol=1e-9):
                        continue
                    
                    # Merge: add W[r1] to W[r2], zero out column r1
                    W_cur[:, r2] += W_cur[:, r1]
                    U_cur[:, r1] = 0
                    V_cur[:, r1] = 0
                    W_cur[:, r1] = 0
                    r_cur -= 1
                    found_merge = True
                    break
                if found_merge:
                    break
            
            if not found_merge:
                break
        
        return U_cur, V_cur, W_cur, r_cur
    
    # Try rank reduction
    if _time_left() > 20:
        U_red, V_red, W_red, rank_red = try_reduce_rank(U, V, W, max_iter=50000)
        if verify(U_red, V_red, W_red) and rank_red < best_rank:
            best_rank = rank_red
            best_U, best_V, best_W = U_red, V_red, W_red
    
    # ============================================================
    # New structural approach: Try a different block structure
    # ============================================================
    # Instead of the standard Strassen 2x2 blocks, try a different
    # partitioning that might allow lower rank.
    
    def try_alternative_decomposition():
        """Try alternative structured decompositions."""
        if _time_left() < 30:
            return None
        
        # Try a 1x1x1 base with different grouping
        # For 2x4x8, we can think of it as 2x(4x8) or (2x4)x8
        # Let's try to find a decomposition using fewer terms
        
        # Approach: Use the fact that we have n=2 (small)
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We can try to express this with fewer multiplications
        
        # For each pair (i,k), we need to compute sum_j A[i,j] B[j,k]
        # Total terms needed: 2*4*8 = 64 (schoolbook)
        # Strassen gives 56
        # Can we do better?
        
        # Try a direct search for rank < 56
        # This is a hard problem, so let's try a heuristic
        
        rng = np.random.RandomState(999)
        
        for target_rank in range(48, 56):
            if _time_left() < 15:
                break
            
            # Try to find a valid decomposition of this rank
            # by solving a system of equations
            
            # The tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
            # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is a polynomial system. Let's try to solve it iteratively.
            
            # Initialize with random values
            U_try = rng.randn(n * m, target_rank)
            V_try = rng.randn(m * p, target_rank)
            W_try = rng.randn(p * n, target_rank)
            
            # Use gradient descent to minimize the error
            # Error = sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
            
            lr = 0.01
            for step in range(500):
                if _time_left() < 10:
                    break
                
                # Compute gradient
                U_grad = np.zeros_like(U_try)
                V_grad = np.zeros_like(V_try)
                W_grad = np.zeros_like(W_try)
                
                for r in range(target_rank):
                    for i in range(n):
                        for j in range(m):
                            # d/dU[i*m+j,r] of sum_k (sum_j' A[i,j']B[j',k] - sum_r' ...)^2
                            # Simplified: for each (i,k), the term involving U[i*m+j,r] is
                            # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                            for k in range(p):
                                # Error contribution
                                val = 0.0
                                for r2 in range(target_rank):
                                    val += U_try[i*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+i, r2]
                                err = val - 1.0
                                U_grad[i*m+j, r] += 2 * err * V_try[j*p+k, r] * W_try[k*n+i, r]
                    
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r2 in range(target_rank):
                                val += U_try[0*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+0, r2]
                            err = val - 1.0
                            V_grad[j*p+k, r] += 2 * err * U_try[0*m+j, r] * W_try[k*n+0, r]
                            
                            val = 0.0
                            for r2 in range(target_rank):
                                val += U_try[1*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+1, r2]
                            err = val - 1.0
                            V_grad[j*p+k, r] += 2 * err * U_try[1*m+j, r] * W_try[k*n+1, r]
                    
                    for k in range(p):
                        for i in range(n):
                            val = 0.0
                            for j in range(m):
                                for r2 in range(target_rank):
                                    val += U_try[i*m+j, r2] * V_try[j*p+k, r2] * W_try[k*n+i, r2]
                            err = val - 1.0
                            W_grad[k*n+i, r] += 2 * err * U_try[i*m+j, r] * V_try[j*p+k, r]
                
                # Update
                U_try -= lr * U_grad
                V_try -= lr * V_grad
                W_try -= lr * W_grad
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U_try[i*m+j, r] * V_try[j*p+k, r] * W_try[k*n+i, r]
                            err += (val - 1.0)**2
                
                if err < 1e-8:
                    # Found a solution!
                    if verify(U_try, V_try, W_try):
                        return U_try, V_try, W_try, target_rank
        
        return None
    
    # Try alternative decomposition
    if _time_left() > 40:
        result = try_alternative_decomposition()
        if result is not None:
            U_alt, V_alt, W_alt, rank_alt = result
            if rank_alt < best_rank:
                best_rank = rank_alt
                best_U, best_V, best_W = U_alt, V_alt, W_alt
    
    # ============================================================
    # Final cleanup and return
    # ============================================================
    if _time_left() > 5:
        # Convert to integers if possible
        def to_int_if_close(arr):
            arr_rounded = np.round(arr, 6)
            if np.allclose(arr, arr_rounded, atol=1e-6):
                return arr_rounded.astype(int)
            return arr
        
        U_out = to_int_if_close(best_U)
        V_out = to_int_if_close(best_V)
        W_out = to_int_if_close(best_W)
        
        # Convert numpy arrays to lists for the return format
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist()
        }
    
    # Fallback
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
