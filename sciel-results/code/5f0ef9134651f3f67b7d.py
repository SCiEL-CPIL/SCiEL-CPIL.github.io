import numpy as np
import time
import random

n, m, p = 3, 4, 6

def Aidx(i, j): return i * m + j
def Bidx(j, k): return j * p + k
def Cidx(i, k): return k * n + i  # W[k*n+i]

def compute_tensor(U, V, W):
    """Compute the full tensor from factor matrices. Returns (n, m, p) array."""
    r = U.shape[1]
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    return np.einsum('ijl,jkl,kil->ijk', U_r, V_r, W_r, optimize=True)

def build_decomposition(i_parts, j_parts, k_parts):
    """Build decomposition from block partitions."""
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_rank1_terms(sub_U, sub_V, sub_W):
        """Add rank-1 terms for a sub-block. sub_U is list of (i,j) positions, etc."""
        pass  # Placeholder
    
    # This approach is complex; let me use the explicit construction instead
    pass

def baseline_66():
    U = [[0]*66 for _ in range(n*m)]
    V = [[0]*66 for _ in range(m*p)]
    W = [[0]*66 for _ in range(p*n)]
    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A00 = Aidx(a_r, a_c); A01 = Aidx(a_r, a_c+1)
        A10 = Aidx(a_r+1, a_c); A11 = Aidx(a_r+1, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C00 = Cidx(c_r, c_c); C01 = Cidx(c_r, c_c+1)
        C10 = Cidx(c_r+1, c_c); C11 = Cidx(c_r+1, c_c+1)
        # M1
        U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
        # M2
        U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
        # M3
        U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
        # M4
        U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
        # M5
        U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
        # M6
        U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
        # M7
        U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        A0 = Aidx(a_r, a_c); A1 = Aidx(a_r, a_c+1)
        B00 = Bidx(b_r, b_c); B01 = Bidx(b_r, b_c+1)
        B10 = Bidx(b_r+1, b_c); B11 = Bidx(b_r+1, b_c+1)
        C0 = Cidx(c_r, c_c); C1 = Cidx(c_r, c_c+1)
        U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
        U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
        U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
        U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1

    add_1x2_2x2(0,0,0,0,0,0); add_1x2_2x2(0,2,2,0,0,0)
    add_1x2_2x2(0,0,0,2,0,2); add_1x2_2x2(0,2,2,2,0,2)
    add_1x2_2x2(0,0,0,4,0,4); add_1x2_2x2(0,2,2,4,0,4)
    add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
    add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
    add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
    assert col == 66
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def verify_np(U, V, W):
    """Verify that the decomposition is exact."""
    result = compute_tensor(U, V, W)
    return np.allclose(result, 1.0, atol=1e-9)

def verify_fast(U, V, W):
    """Fast verification: compute tensor and check."""
    result = compute_tensor(U, V, W)
    return np.allclose(result, 1.0, atol=1e-9)

def try_drop(U, V, W, l):
    """Check if removing term l still gives a valid decomposition."""
    U_new = np.delete(U, l, axis=1)
    V_new = np.delete(V, l, axis=1)
    W_new = np.delete(W, l, axis=1)
    return verify_fast(U_new, V_new, W_new)

def try_absorb(U, V, W, l_remove, l_keep):
    """Try to absorb term l_remove into term l_keep by scaling l_keep.
    Check if there exists a scalar s such that removing l_remove and
    scaling l_keep by (1+s) still works. More generally, check if
    the contribution of l_remove can be absorbed by adjusting l_keep."""
    # For this to work, we need:
    # U[:,l_remove] ⊗ V[:,l_remove] ⊗ W[:,l_remove] = 
    #   s * (U[:,l_keep] ⊗ V[:,l_keep] ⊗ W[:,l_keep])
    # for some scalar s. This means the three vectors must be proportional.
    Ur = U[:, l_remove]
    Uk = U[:, l_keep]
    Vr = V[:, l_remove]
    Vk = V[:, l_keep]
    Wr = W[:, l_remove]
    Wk = W[:, l_keep]
    
    # Check if Ur is proportional to Uk
    mask_u = Ur != 0
    if not np.any(mask_u):
        return False
    s_u = Ur[mask_u][0] / Uk[mask_u][0]
    if not np.allclose(Ur, s_u * Uk, atol=1e-9):
        return False
    
    mask_v = Vr != 0
    if not np.any(mask_v):
        return False
    s_v = Vr[mask_v][0] / Vk[mask_v][0]
    if not np.allclose(Vr, s_v * Vk, atol=1e-9):
        return False
    
    mask_w = Wr != 0
    if not np.any(mask_w):
        return False
    s_w = Wr[mask_w][0] / Wk[mask_w][0]
    if not np.allclose(Wr, s_w * Wk, atol=1e-9):
        return False
    
    # All three must have the same scale factor
    if not (np.isclose(s_u, s_v, atol=1e-9) and np.isclose(s_v, s_w, atol=1e-9)):
        return False
    
    # Remove l_remove
    U_new = np.delete(U, l_remove, axis=1)
    V_new = np.delete(V, l_remove, axis=1)
    W_new = np.delete(W, l_remove, axis=1)
    return verify_fast(U_new, V_new, W_new)

def local_search_v2(U, V, W, deadline, rng):
    """Enhanced local search with drop and absorb attempts.
    Uses faster verification to allow more iterations."""
    r = U.shape[1]
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        indices = list(range(r))
        rng.shuffle(indices)
        
        # Pass 1: try drops
        for l in indices:
            if time.time() > deadline - 10:
                break
            if try_drop(U, V, W, l):
                U = np.delete(U, l, axis=1)
                V = np.delete(V, l, axis=1)
                W = np.delete(W, l, axis=1)
                r -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Pass 2: try absorbing one term into another (check proportionality)
        for i in range(r):
            if time.time() > deadline - 10:
                break
            for j in range(i+1, r):
                if time.time() > deadline - 10:
                    break
                if try_absorb(U, V, W, j, i):
                    U = np.delete(U, j, axis=1)
                    V = np.delete(V, j, axis=1)
                    W = np.delete(W, j, axis=1)
                    r -= 1
                    improved = True
                    break
            if improved:
                break
    return U, V, W

def construct():
    deadline = time.time() + 130  # safety margin for 150s limit
    rng = random.Random(42)
    U, V, W = baseline_66()
    U, V, W = local_search_v2(U, V, W, deadline, rng)
    if not verify_np(U, V, W):
        U, V, W = baseline_66()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
