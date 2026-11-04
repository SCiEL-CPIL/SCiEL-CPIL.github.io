"""Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_and_clean(U, V, W, n, m, p, rank):
    """Verify and clean up numerical noise in the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Clean small values
    U[np.abs(U) < 1e-10] = 0.0
    V[np.abs(V) < 1e-10] = 0.0
    W[np.abs(W) < 1e-10] = 0.0
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p, rank, tol=1e-4):
        return None
    
    # Try to snap to half-integers
    def snap(arr):
        snapped = np.round(arr * 2) / 2
        # Check if snapping is close
        if np.max(np.abs(arr - snapped)) < 1e-5:
            return snapped
        # Try integers
        snapped_int = np.round(arr)
        if np.max(np.abs(arr - snapped_int)) < 1e-5:
            return snapped_int
        return arr
    
    U_s = snap(U)
    V_s = snap(V)
    W_s = snap(W)
    
    # Verify snapped version
    if _verify_decomposition(U_s, V_s, W_s, n, m, p, rank, tol=1e-4):
        # Convert to lists of ints or halves
        def to_list(arr):
            result = []
            for row in arr:
                r = []
                for v in row:
                    if abs(v - round(v)) < 1e-9:
                        r.append(int(round(v)))
                    elif abs(2*v - round(2*v)) < 1e-9:
                        r.append(int(round(2*v))/2.0)
                    else:
                        r.append(float(v))
                result.append(r)
            return result
        return {"U": to_list(U_s), "V": to_list(V_s), "W": to_list(W_s)}
    
    return None

def _als_decompose(n, m, p, rank, max_iter=300, deadline=None, verbose=False):
    """Alternating least squares for CP decomposition of the matmul tensor."""
    nm, mp, pn = n*m, m*p, p*n
    
    # Random initialization
    np.random.seed(int(time.time() * 1000) % 100000)
    U = np.random.randn(nm, rank)
    V = np.random.randn(mp, rank)
    W = np.random.randn(pn, rank)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W: fix U and V
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        M = np.einsum('ijc,jkc->kic', U3, V3)  # shape (p, n, rank)
        
        for k in range(p):
            for i in range(n):
                vec = M[k, i, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    W[k*n+i, :] = vec / norm_sq
                else:
                    W[k*n+i, :] = 0.0
        
        # Update U: fix V and W
        W3 = W.reshape(p, n, rank)
        N = np.einsum('jkc,kic->ijc', V3, W3)  # shape (n, m, rank)
        
        for i in range(n):
            for j in range(m):
                vec = N[i, j, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    U[i*m+j, :] = vec / norm_sq
                else:
                    U[i*m+j, :] = 0.0
        
        # Update V: fix U and W
        U3 = U.reshape(n, m, rank)
        P = np.einsum('ijc,kic->jkc', U3, W3)  # shape (m, p, rank)
        
        for j in range(m):
            for k in range(p):
                vec = P[j, k, :]
                norm_sq = np.dot(vec, vec)
                if norm_sq > 1e-12:
                    V[j*p+k, :] = vec / norm_sq
                else:
                    V[j*p+k, :] = 0.0
        
        # Check convergence
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        err = np.max(np.abs(recon - 1.0))
        
        if verbose and it % 50 == 0:
            print(f"  Iter {it}: err={err:.2e}")
        
        if err < 1e-8:
            return U, V, W, True
    
    return U, V, W, False

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try ranks from low to high
    # Known lower bounds: R(2,3,7) >= max(2*3, 3*7, 2*7) = 21
    # But tensor rank can be higher than these trivial bounds
    # Try from 21 upward
    
    random.seed(42)
    np.random.seed(42)
    
    for rank in range(21, best_rank):
        if time.time() > deadline - 30:
            break
        
        found = False
        # Multiple random restarts for each rank
        for attempt in range(5):
            if time.time() > deadline - 25:
                break
            
            U, V, W, converged = _als_decompose(n, m, p, rank, max_iter=200, deadline=deadline - 20)
            
            if converged:
                result = _verify_and_clean(U, V, W, n, m, p, rank)
                if result is not None:
                    best = result
                    best_rank = rank
                    found = True
                    break
        
        if found:
            continue
    
    # Final verification
    if best_rank < rank_sb:
        U_final = np.array(best["U"])
        V_final = np.array(best["V"])
        W_final = np.array(best["W"])
        if _verify_decomposition(U_final, V_final, W_final, n, m, p, best_rank, tol=1e-4):
            return best
    
    # Return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
