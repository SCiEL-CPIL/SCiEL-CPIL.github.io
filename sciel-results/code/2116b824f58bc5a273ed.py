"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from itertools import product
from fractions import Fraction
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Build the matrix multiplication tensor as a 3D array T[i, j, k]
    # T[i,j,k] = 1 if this is the coefficient for A[i,j]*B[j,k]
    # The tensor has shape (n*m, m*p, p*n)
    # T[u, v, w] where u = i*m+j, v = j*p+k, w = k*n+i
    # Actually, the tensor is defined as T[u, v, w] = 1 if there exist i,j,k 
    # such that u=i*m+j, v=j*p+k, w=k*n+i
    # But that's only if the indices are consistent: same j for u,v and same 
    # (i,k) for u,w and same j for v,w
    
    # Let's think about this more carefully.
    # We want to decompose the matrix multiplication tensor.
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has indices (i, j, k) and T[i,j,k] = 1 for all valid i,j,k
    # Wait, that's not right either. The tensor is defined such that:
    # (AB)_{i,k} = sum_r U_{i*m+j?, ...}
    
    # Let me use the standard tensor formulation.
    # The matrix multiplication tensor M has entries M[i, j, k] = 1 for i in [0,n), j in [0,m), k in [0,p)
    # where these indices correspond to (row of A, column of A = row of B, column of B)
    # The flattening: for the U matrix (size n*m x rank), the row index is i*m + j
    # For V (size m*p x rank), the row index is j*p + k  
    # For W (size p*n x rank), the row index is k*n + i
    # The decomposition says: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Since M[i,j,k] = 1 for all valid (i,j,k), we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a 3-tensor with "multi-linear" structure. Let's think of it as:
    # For each (i,j,k), the trilinear form U[i*m+j,:] · V[j*p+k,:] · W[k*n+i,:] = 1
    
    # Actually, the standard way: the matrix multiplication tensor has entries
    # M_{(i,j),(j,k),(k,i)} = 1, which means when we flatten:
    # The tensor T of shape (n*m, m*p, p*n) has T[u,v,w] = 1 iff
    # u = i*m+j, v = j*p+k, w = k*n+i for some consistent (i,j,k)
    
    # Let's construct the tensor T as a 3D numpy array
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Now we want to find rank-minimal U, V, W such that
    # T[u,v,w] = sum_r U[u,r] * V[v,r] * W[w,r]
    # This is a 3-mode tensor decomposition (CP decomposition)
    
    # The rank of matrix multiplication for n x m by m x p:
    # Known results: rank(2x2x2) = 7, rank(2x2x3) = ?
    # For 2x5x6, we don't have a known closed form.
    # The schoolbook rank is n*m*p = 60.
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also rank >= (n*m*p) / something... 
    # A known lower bound is rank >= max(n, m, p)^2 for square cases, but that's not tight here.
    
    # Let's try to find a lower rank using randomized search or gradient-based optimization.
    # Given the 150s time limit, we need to be efficient.
    
    # Strategy: Try random initializations and use alternating least squares or 
    # gradient descent to find a low-rank decomposition.
    
    # First, let's establish the schoolbook solution as a fallback
    U_fb = np.zeros((n*m, rank_school))
    V_fb = np.zeros((m*p, rank_school))
    W_fb = np.zeros((p*n, rank_school))
    for r in range(rank_school):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U_fb[i*m + j, r] = 1.0
        V_fb[j*p + k, r] = 1.0
        W_fb[k*n + i, r] = 1.0
    
    # Try to find lower rank decompositions
    best_U = U_fb
    best_V = V_fb
    best_W = W_fb
    best_rank = rank_school
    best_err = 0.0
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Try ranks from lower bound up
    # Lower bound: at least max(n*m, m*p, p*n) = 30
    # Also, rank >= (n*m*p)/min(n,m,p)? Not sure. Let's just try from 30 upward.
    
    # Actually, for matrix multiplication, the rank is known to be at least 
    # max(n*m, m*p, p*n) and at most n*m*p.
    # Let's try a few random inits at various ranks and see if we can find something.
    
    np.random.seed(42)
    
    # Try to see if we can find rank < 60
    # Use a simple iterative approach: for a given rank r, initialize U, V, W randomly
    # and try to minimize ||T - U*V*W||_F using alternating least squares.
    
    def _try_rank(r, max_iters=200, tol=1e-10):
        """Try to find a rank-r decomposition using ALS."""
        if r <= 0:
            return None, float('inf')
        
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        best_err_local = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each u, U[u, :] = argmin_Uu ||T[u, :, :] - Uu * (V * W)||
            # T[u, :, :] is a (m*p, p*n) matrix
            # We need Uu (size r) such that: for all v, w: T[u,v,w] = sum_r Uu[r] * V[v,r] * W[w,r]
            # This is: T[u, :, :] = Uu^T ⊙ (V * W^T) element-wise? No.
            # T[u,v,w] = sum_r Uu[r] * V[v,r] * W[w,r]
            # Let M_uv = V[v, :] * W[:, v]... no.
            # For fixed u, we have a 2D matrix T_u of size (m*p, p*n)
            # T_u[v, w] = sum_r Uu[r] * V[v,r] * W[w,r]
            # This is: T_u = (V * diag(Uu)) * W^T? 
            # (V * diag(Uu))[v, r] = V[v,r] * Uu[r]
            # ((V * diag(Uu)) * W^T)[v, w] = sum_r V[v,r] * Uu[r] * W[w,r] = T_u[v,w] ✓
            # So T_u = V * diag(Uu) * W^T
            # To find Uu given V, W: we can use the vec trick or solve a linear system.
            # vec(T_u) = (W ⊗ V) * Uu
            # So Uu = pinv(W ⊗ V) * vec(T_u)
            
            # But W ⊗ V is (p*n * m*p) x r, which is huge: (12*30=360) x r
            # That's manageable for r up to ~60.
            
            # Actually, let's use a more efficient approach.
            # T_u = V * diag(Uu) * W^T
            # Let's reshape: for fixed u, define A_u = V (m*p x r), B_u = W (p*n x r)
            # T_u[v,w] = sum_r A_u[v,r] * Uu[r] * B_u[w,r]
            # This is a bilinear form. We can solve for Uu using:
            # sum_v,w T_u[v,w]^2 = sum_v,w (sum_r A_u[v,r] * Uu[r] * B_u[w,r])^2
            # This is nonlinear in Uu.
            
            # Alternative: use the fact that T_u = V * diag(Uu) * W^T
            # Multiply on right by W^+ (pseudo-inverse of W^T, i.e., W_pinv):
            # T_u * W_pinv = V * diag(Uu)
            # Where W_pinv is such that W^T * W_pinv ≈ I (if r <= p*n)
            # Then V^+ * T_u * W_pinv = diag(Uu) (approximately)
            # So Uu[r] ≈ (V^+ * T_u * W_pinv)[r, r]
            
            # Let's compute V_pinv = pinv(V) (r x m*p) and W_pinv = pinv(W) (r x p*n)
            V_pinv = np.linalg.pinv(V)  # (r, m*p)
            W_pinv = np.linalg.pinv(W)  # (r, p*n)
            
            # For each u:
            new_U = np.zeros_like(U)
            for u in range(n*m):
                # T_u = T[u, :, :] shape (m*p, p*n)
                # V_pinv @ T_u @ W_pinv.T gives (r, r)
                # The diagonal should approximate Uu
                M = V_pinv @ T[u, :, :] @ W_pinv.T  # (r, r)
                new_U[u, :] = np.diag(M)
            
            U = new_U
            
            # Similarly for V: T_v[u, w] = sum_r U[u,r] * V[v,r] * W[w,r]
            # T_v = U^T * diag(Vv) * W  where U^T is (r, n*m)... 
            # Wait: T_v[u, w] = sum_r U[u,r] * Vv[r] * W[w,r]
            # Let's define: for fixed v, T[:, v, :] is a (n*m, p*n) matrix
            # T_v[u, w] = sum_r U[u,r] * Vv[r] * W[w,r]
            # = (U * diag(Vv)) @ W^T? 
            # (U * diag(Vv))[u, r] = U[u,r] * Vv[r]
            # ((U * diag(Vv)) @ W^T)[u, w] = sum_r U[u,r] * Vv[r] * W[w,r] ✓
            # So T_v = (U * diag(Vv)) @ W^T
            # T_v @ pinv(W^T) = U * diag(Vv)
            # pinv(U) @ T_v @ pinv(W^T) = diag(Vv)
            
            W_pinv = np.linalg.pinv(W)  # (r, p*n), so pinv(W^T) = W_pinv.T
            U_pinv = np.linalg.pinv(U)  # (r, n*m)
            
            new_V = np.zeros_like(V)
            for v in range(m*p):
                T_v = T[:, v, :]  # (n*m, p*n)
                M = U_pinv @ T_v @ W_pinv.T  # (r, r)
                new_V[v, :] = np.diag(M)
            V = new_V
            
            # Similarly for W: for fixed w, T[:, :, w] is (n*m, m*p)
            # T_w[u, v] = sum_r U[u,r] * V[v,r] * Ww[r]
            # T_w = U^T * diag(Ww) * V? 
            # T_w[u, v] = sum_r U[u,r] * Ww[r] * V[v,r]
            # = (U * diag(Ww)) @ V^T
            # (U * diag(Ww))[u,r] = U[u,r]*Ww[r]
            # ((U*diag(Ww)) @ V^T)[u,v] = sum_r U[u,r]*Ww[r]*V[v,r] ✓
            # So T_w = (U * diag(Ww)) @ V^T
            # T_w @ pinv(V^T) = U * diag(Ww)
            # pinv(U) @ T_w @ pinv(V^T) = diag(Ww)
            
            V_pinv = np.linalg.pinv(V)  # (r, m*p)
            U_pinv = np.linalg.pinv(U)  # (r, n*m)
            
            new_W = np.zeros_like(W)
            for w in range(p*n):
                T_w = T[:, :, w]  # (n*m, m*p)
                M = U_pinv @ T_w @ V_pinv.T  # (r, r)
                new_W[w, :] = np.diag(M)
            W = new_W
            
            # Compute error
            err = 0.0
            for u in range(n*m):
                for v in range(m*p):
                    # T[u,v,:] = sum_r U[u,r] * V[v,r] * W[:, r]
                    # = (U[u,:] * V[v,:]) @ W
                    pred = (U[u, :] * V[v, :]) @ W  # (p*n,)
                    err += np.sum((T[u, v, :] - pred) ** 2)
            
            if err < best_err_local:
                best_err_local = err
                if err < tol:
                    break
        return (U, V, W), best_err_local
    
    # Try various ranks
    # Start from a reasonable lower bound
    # Known: rank(2x2x2) = 7, schoolbook = 8
    # For 2x5x6, let's try from 40 upward to see if we can beat 60
    ranks_to_try = list(range(40, 60)) + [60]
    np.random.shuffle(ranks_to_try)
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        
        (U_cand, V_cand, W_cand), err = _try_rank(r, max_iters=50)
        
        if err < 1e-8:
            # Found a valid decomposition
            best_rank = r
            best_U = U_cand
            best_V = V_cand
            best_W = W_cand
            best_err = err
            print(f"Found rank {r} decomposition with error {err}")
    
    # Convert best solution to the required format
    # Need integer or half-integer entries
    def _snap_to_half_integer(x):
        """Snap to nearest half-integer."""
        return np.round(x * 2) / 2
    
    # If we found a better rank, try to clean up
    if best_rank < rank_school:
        best_U = _snap_to_half_integer(best_U)
        best_V = _snap_to_half_integer(best_V)
        best_W = _snap_to_half_integer(best_W)
        
        # Verify
        err = 0.0
        for u in range(n*m):
            for v in range(m*p):
                pred = (best_U[u, :] * best_V[v, :]) @ best_W
                err += np.sum((T[u, v, :] - pred) ** 2)
        
        if err < 1e-6:
            # Good, use this solution
            pass
        else:
            # Fallback to schoolbook
            best_U = U_fb
            best_V = V_fb
            best_W = W_fb
            best_rank = rank_school
    
    # Convert to list of lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
