"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, T, tol=1e-9):
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return bool(np.allclose(result, T, atol=tol))

def _rank17(n, m, p):
    n_a, n_b, n_c = n*m, m*p, p*n
    R = 17
    U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
    
    # Strassen for top-left 2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
    # U index: i*m+j -> 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2)
    # V index: j*p+k -> 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,0) 4:(1,1) 5:(1,2) 6:(2,0) 7:(2,1) 8:(2,2)
    # W index: k*n+i -> 0:(0,0) 1:(0,1) 2:(1,0) 3:(1,1) 4:(2,0) 5:(2,1)
    
    def s(r, ud, vd, wd):
        for a, v in ud.items(): U[a, r] = v
        for b, v in vd.items(): V[b, r] = v
        for c, v in wd.items(): W[c, r] = v
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00,C11
    s(0, {0:1,4:1}, {0:1,4:1}, {0:1,3:1})
    # M2 = (A10+A11)B00 -> C10, -C11
    s(1, {3:1,4:1}, {0:1}, {2:1,3:-1})
    # M3 = A00(B01-B11) -> C01, C11
    s(2, {0:1}, {1:1,4:-1}, {1:1,3:1})
    # M4 = A11(B10-B00) -> C00, C10
    s(3, {4:1}, {3:1,0:-1}, {0:1,2:1})
    # M5 = (A00+A01)B11 -> -C00, C01
    s(4, {0:1,1:1}, {4:1}, {0:-1,1:1})
    # M6 = (A10-A00)(B00+B01) -> C11
    s(5, {3:1,0:-1}, {0:1,1:1}, {3:1})
    # M7 = (A01-A11)(B10+B11) -> C00
    s(6, {1:1,4:-1}, {3:1,4:1}, {0:1})
    
    # Direct: j in {0,1}, k=2
    s(7,  {0:1}, {2:1}, {4:1})   # A00*B02 -> C02
    s(8,  {1:1}, {5:1}, {4:1})   # A01*B12 -> C02
    s(9,  {3:1}, {2:1}, {5:1})   # A10*B02 -> C12
    s(10, {4:1}, {5:1}, {5:1})   # A11*B12 -> C12
    
    # Direct: j=2
    s(11, {2:1}, {6:1}, {0:1})   # A02*B20 -> C00
    s(12, {2:1}, {7:1}, {1:1})   # A02*B21 -> C01
    s(13, {2:1}, {8:1}, {4:1})   # A02*B22 -> C02
    s(14, {5:1}, {6:1}, {2:1})   # A12*B20 -> C10
    s(15, {5:1}, {7:1}, {3:1})   # A12*B21 -> C11
    s(16, {5:1}, {8:1}, {5:1})   # A12*B22 -> C12
    
    return U, V, W

def _try_rank16_structured(T, U17, V17, W17, deadline):
    """
    Structural search for rank 16 by trying to merge pairs of terms from rank 17
    and checking if a linear combination can replace them.
    Also tries a global optimization approach.
    """
    n_a, n_b, n_c = T.shape
    R = 17
    T_flat = T.flatten()
    
    # Approach 1: Try dropping pairs and adding a single replacement term
    # For each pair (i,j), check if there exist scalars alpha, beta such that
    # alpha*term_i + beta*term_j can be replaced by a single rank-1 term
    # that, combined with the remaining R-2 terms, still sums to T.
    
    # More practically: for each pair to drop, solve for a replacement
    for drop1 in range(R):
        if time.time() > deadline:
            return None
        for drop2 in range(drop1 + 1, R):
            if time.time() > deadline:
                return None
            
            keep = [t for t in range(R) if t not in (drop1, drop2)]
            n_keep = len(keep)
            
            # Build matrix of kept terms
            A_keep = np.zeros((324, n_keep))
            for ci, ti in enumerate(keep):
                outer = np.einsum('a,b,c->abc', U17[:, ti], V17[:, ti], W17[:, ti])
                A_keep[:, ci] = outer.flatten()
            
            # The residual we need to cover with one new term
            residual = T_flat - A_keep @ np.ones(n_keep)
            
            # We need u ⊗ v ⊗ w = residual (as a 3D array)
            # This means residual must be rank-1 in the tensor sense
            # Reshape to 2D and check if it's rank 1
            res_3d = residual.reshape(n_a, n_b, n_c)
            
            # Check if res_3d is a rank-1 tensor: res_3d[a,b,c] = u[a]*v[b]*w[c]
            # Find nonzero entries
            nz = np.nonzero(res_3d)
            if len(nz[0]) == 0:
                # Already done with R-2 terms
                Ux = np.zeros((n_a, n_keep)); Vx = np.zeros((n_b, n_keep)); Wx = np.zeros((n_c, n_keep))
                for ci, ti in enumerate(keep):
                    Ux[:, ci] = U17[:, ti]; Vx[:, ci] = V17[:, ti]; Wx[:, ci] = W17[:, ti]
                if _verify(Ux, Vx, Wx, T):
                    return (Ux, Vx, Wx)
                continue
            
            # Try to factor as rank-1
            # Pick a reference index
            a0, b0, c0 = nz[0][0], nz[1][0], nz[2][0]
            val0 = res_3d[a0, b0, c0]
            if abs(val0) < 1e-12:
                continue
            
            # u[a] = res_3d[a, b0, c0] / val0
            # v[b] = res_3d[a0, b, c0] / val0  
            # w[c] = res_3d[a0, b0, c] / val0
            u_vec = res_3d[:, b0, c0] / val0
            v_vec = res_3d[a0, :, c0] / val0
            w_vec = res_3d[a0, b0, :] / val0
            
            # Verify the factorization
            recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
            if np.allclose(recon, res_3d, atol=1e-9):
                # Success! Build the new decomposition
                Ux = np.zeros((n_a, n_keep + 1)); Vx = np.zeros((n_b, n_keep + 1)); Wx = np.zeros((n_c, n_keep + 1))
                for ci, ti in enumerate(keep):
                    Ux[:, ci] = U17[:, ti]; Vx[:, ci] = V17[:, ti]; Wx[:, ci] = W17[:, ti]
                Ux[:, n_keep] = u_vec; Vx[:, n_keep] = v_vec; Wx[:, n_keep] = w_vec
                if _verify(Ux, Vx, Wx, T):
                    return (Ux, Vx, Wx)
    
    # Approach 2: Global optimization - start from rank 17, zero out one term,
    # and optimize the remaining terms to compensate
    if time.time() > deadline:
        return None
    
    np.random.seed(123)
    
    for drop_idx in range(R):
        if time.time() > deadline:
            return None
        
        n_new = R - 1  # 16
        # Initialize with the kept terms
        U_cand = np.zeros((n_a, n_new)); V_cand = np.zeros((n_b, n_new)); W_cand = np.zeros((n_c, n_new))
        keep = [t for t in range(R) if t != drop_idx]
        for ci, ti in enumerate(keep):
            U_cand[:, ci] = U17[:, ti]; V_cand[:, ci] = V17[:, ti]; W_cand[:, ci] = W17[:, ti]
        
        # Add small random perturbation to help escape local minima
        for _ in range(50):
            if time.time() > deadline:
                break
            
            U_pert = U_cand + np.random.randn(n_a, n_new) * 0.01
            V_pert = V_cand + np.random.randn(n_b, n_new) * 0.01
            W_pert = W_cand + np.random.randn(n_c, n_new) * 0.01
            
            # Build linear system: sum_r u_r ⊗ v_r ⊗ w_r = T
            # This is nonlinear, so use iterative approach
            # Flatten each outer product
            A = np.zeros((324, n_new))
            for r in range(n_new):
                outer = np.einsum('a,b,c->abc', U_pert[:, r], V_pert[:, r], W_pert[:, r])
                A[:, r] = outer.flatten()
            
            # Solve A c = T_flat
            try:
                c, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=None)
            except:
                continue
            
            # Apply correction
            U_sol = U_pert.copy()
            for r in range(n_new):
                U_sol[:, r] *= c[r]
            
            if _verify(U_sol, V_pert, W_pert, T):
                return (U_sol, V_pert, W_pert)
    
    return None

def _try_rank16_random(T, deadline):
    """Randomized search for rank 16 with better structure awareness."""
    n_a, n_b, n_c = T.shape
    T_flat = T.flatten()
    np.random.seed(42)
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        R = 16
        U = np.zeros((n_a, R)); V = np.zeros((n_b, R)); W = np.zeros((n_c, R))
        
        for r in range(R):
            # Each term should cover a consistent subset of (i,j,k) indices
            # Randomly pick 1-3 "blocks" to cover
            n_blocks = np.random.randint(1, 4)
            for _ in range(n_blocks):
                i = np.random.randint(0, 2)
                j = np.random.randint(0, 3)
                k = np.random.randint(0, 3)
                a = i*3 + j
                b = j*3 + k
                c = k*2 + i
                U[a, r] += np.random.choice([-1, 1])
                V[b, r] += np.random.choice([-1, 1])
                W[c, r] += np.random.choice([-1, 1])
        
        # Build and solve
        A = np.zeros((324, R))
        for r in range(R):
            outer = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            A[:, r] = outer.flatten()
        
        try:
            c, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=None)
        except:
            continue
        
        U_sol = U.copy()
        for r in range(R):
            U_sol[:, r] *= c[r]
        
        if _verify(U_sol, V, W, T):
            return (U_sol, V, W)
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Baseline rank 17
    U17, V17, W17 = _rank17(n, m, p)
    if _verify(U17, V17, W17, T):
        base = (U17, V17, W17)
    else:
        # Fallback schoolbook
        R = 18
        U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1
                    col += 1
        base = (U, V, W)
    
    # Try to find rank 16
    result = _try_rank16_structured(T, base[0], base[1], base[2], deadline)
    if result is None and time.time() < deadline:
        result = _try_rank16_random(T, deadline)
    
    if result is not None:
        U, V, W = result
    else:
        U, V, W = base
    
    # Final verification
    if not _verify(U, V, W, T):
        U, V, W = base
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
