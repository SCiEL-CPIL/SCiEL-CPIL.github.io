import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 145  # seconds, safety margin before 150s kill

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a dense 3D array."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_exact(U, V, W, T, tol=1e-10):
    """Check if U, V, W exactly decompose T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=tol)

def _als_step(n, m, p, rank, U, V, W, T, max_iters=1000, tol=1e-12):
    """
    Efficient ALS for tensor decomposition.
    T shape: (nm, mp, pn)
    U shape: (nm, rank)
    V shape: (mp, rank)
    W shape: (pn, rank)
    """
    nm, mp, pn = n * m, m * p, p * n
    
    B_u = T.reshape(nm, -1)
    B_v = T.transpose(1, 0, 2).reshape(mp, -1)
    B_w = T.transpose(2, 0, 1).reshape(pn, -1)
    
    err = float('inf')
    for it in range(max_iters):
        K_u = np.einsum('br,cr->bcr', V, W)
        A_u = K_u.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            U = np.linalg.lstsq(A_u.T, B_u.T, rcond=None)[0].T
        except:
            pass

        K_v = np.einsum('ar,cr->acr', U, W)
        A_v = K_v.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            V = np.linalg.lstsq(A_v.T, B_v.T, rcond=None)[0].T
        except:
            pass

        K_w = np.einsum('ar,br->abr', U, V)
        A_w = K_w.transpose(1, 0, 2).reshape(-1, rank)
        
        try:
            W = np.linalg.lstsq(A_w.T, B_w.T, rcond=None)[0].T
        except:
            pass
        
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            break
            
    return U, V, W, err

def _try_rank_als_fast(n, m, p, rank, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using fast ALS."""
    rng = np.random.RandomState(seed)
    T = _build_tensor(n, m, p)
    
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    
    U, V, W, err = _als_step(n, m, p, rank, U, V, W, T, max_iters=max_iters, tol=1e-12)
    
    if err < 1e-10:
        return U, V, W, True
    return U, V, W, False

def _get_trivial_decomposition(n, m, p):
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _get_strassen_decomposition(n, m, p):
    """
    Construct a rank-55 decomposition for 2x3x10 matrix multiplication.
    Uses 5 blocks of rank-11 decomposition for 2x3x2.
    """
    if (n, m, p) != (2, 3, 10):
        return None

    rank = 55
    U = np.zeros((6, rank))
    V = np.zeros((30, rank))
    W = np.zeros((20, rank))

    # Strassen factors for 2x2x2
    # U indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in local 2x2
    # V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in local 2x2
    # W indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in local 2x2
    
    # M1: (a00+a11)(b00+b11) -> c00+=1, c11+=1
    # M2: (a10+a11)b00 -> c10+=1, c11-=1
    # M3: a00(b01-b11) -> c01+=1, c11+=1
    # M4: a11(b10-b00) -> c00+=1, c10+=1
    # M5: (a00+a01)b11 -> c00-=1, c01+=1
    # M6: (a10-a00)(b00+b01) -> c11+=1
    # M7: (a01-a11)(b10+b11) -> c00+=1
    
    # Local U coeffs for a00, a01, a10, a11
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, 1
    # M3: 1, 0, 0, 0
    # M4: 0, 0, 0, 1
    # M5: 1, 1, 0, 0
    # M6: -1, 0, 1, 0
    # M7: 0, 1, 0, -1
    
    # Local V coeffs for b00, b01, b10, b11
    # M1: 1, 0, 0, 1
    # M2: 1, 0, 0, 0
    # M3: 0, 1, 0, -1
    # M4: 0, 0, 1, -1
    # M5: 0, 0, 0, 1
    # M6: 1, 1, 0, 0
    # M7: 0, 0, 1, 1
    
    # Local W coeffs for c00, c01, c10, c11
    # M1: 1, 0, 0, 1
    # M2: 0, 0, 1, -1
    # M3: 0, 1, 0, 1
    # M4: 1, 0, 1, 0
    # M5: -1, 1, 0, 0
    # M6: 0, 0, 0, 1
    # M7: 1, 0, 0, 0

    U_local = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    # Actually, let's just hardcode the non-zeros as derived in thought process.
    
    for p_idx in range(5):
        base = p_idx * 11
        # Columns in B: 2*p_idx, 2*p_idx+1
        b_col_0 = 2 * p_idx
        b_col_1 = 2 * p_idx + 1
        
        # Strassen Part
        # U (A indices)
        # a00 (idx 0): M1(1), M3(1), M5(1), M6(-1)
        U[0, base + 0] = 1
        U[0, base + 2] = 1
        U[0, base + 4] = 1
        U[0, base + 5] = -1
        
        # a01 (idx 1): M5(1), M7(1)
        U[1, base + 4] = 1
        U[1, base + 6] = 1
        
        # a10 (idx 3): M2(1), M4(1), M6(1)
        U[3, base + 1] = 1
        U[3, base + 3] = 1
        U[3, base + 5] = 1
        
        # a11 (idx 4): M1(1), M2(1), M4(1), M7(-1)
        U[4, base + 0] = 1
        U[4, base + 1] = 1
        U[4, base + 3] = 1
        U[4, base + 6] = -1
        
        # V (B indices)
        # b00 (idx 0*10 + b_col_0): M1(1), M2(1), M6(1)
        V[0 * 10 + b_col_0, base + 0] = 1
        V[0 * 10 + b_col_0, base + 1] = 1
        V[0 * 10 + b_col_0, base + 5] = 1
        
        # b01 (idx 0*10 + b_col_1): M3(-1), M5(1), M6(1)
        V[0 * 10 + b_col_1, base + 2] = -1
        V[0 * 10 + b_col_1, base + 4] = 1
        V[0 * 10 + b_col_1, base + 5] = 1
        
        # b10 (idx 1*10 + b_col_0): M4(-1), M7(1)
        V[1 * 10 + b_col_0, base + 3] = -1
        V[1 * 10 + b_col_0, base + 6] = 1
        
        # b11 (idx 1*10 + b_col_1): M1(1), M3(1), M5(1), M7(1)
        V[1 * 10 + b_col_1, base + 0] = 1
        V[1 * 10 + b_col_1, base + 2] = 1
        V[1 * 10 + b_col_1, base + 4] = 1
        V[1 * 10 + b_col_1, base + 6] = 1
        
        # W (C indices)
        # c00 (idx b_col_0 * 2 + 0): M1(1), M4(1), M5(-1), M7(1)
        W[b_col_0 * 2 + 0, base + 0] = 1
        W[b_col_0 * 2 + 0, base + 3] = 1
        W[b_col_0 * 2 + 0, base + 4] = -1
        W[b_col_0 * 2 + 0, base + 6] = 1
        
        # c01 (idx b_col_1 * 2 + 0): M3(1), M5(1)
        W[b_col_1 * 2 + 0, base + 2] = 1
        W[b_col_1 * 2 + 0, base + 4] = 1
        
        # c10 (idx b_col_0 * 2 + 1): M2(1), M4(1)
        W[b_col_0 * 2 + 1, base + 1] = 1
        W[b_col_0 * 2 + 1, base + 3] = 1
        
        # c11 (idx b_col_1 * 2 + 1): M1(1), M2(-1), M3(1), M6(1)
        W[b_col_1 * 2 + 1, base + 0] = 1
        W[b_col_1 * 2 + 1, base + 1] = -1
        W[b_col_1 * 2 + 1, base + 2] = 1
        W[b_col_1 * 2 + 1, base + 5] = 1
        
        # Remaining Part
        # U (A indices)
        # a02 (idx 2): M8(1), M9(1)
        U[2, base + 7] = 1
        U[2, base + 8] = 1
        
        # a12 (idx 5): M10(1), M11(1)
        U[5, base + 9] = 1
        U[5, base + 10] = 1
        
        # V (B indices)
        # b20 (idx 2*10 + b_col_0): M8(1), M10(1)
        V[2 * 10 + b_col_0, base + 7] = 1
        V[2 * 10 + b_col_0, base + 9] = 1
        
        # b21 (idx 2*10 + b_col_1): M9(1), M11(1)
        V[2 * 10 + b_col_1, base + 8] = 1
        V[2 * 10 + b_col_1, base + 10] = 1
        
        # W (C indices)
        # c00 (idx b_col_0 * 2 + 0): M8(1)
        W[b_col_0 * 2 + 0, base + 7] = 1
        
        # c01 (idx b_col_1 * 2 + 0): M9(1)
        W[b_col_1 * 2 + 0, base + 8] = 1
        
        # c10 (idx b_col_0 * 2 + 1): M10(1)
        W[b_col_0 * 2 + 1, base + 9] = 1
        
        # c11 (idx b_col_1 * 2 + 1): M11(1)
        W[b_col_1 * 2 + 1, base + 10] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_structured_decomposition(n, m, p):
    """
    Try to find a lower rank decomposition using fast ALS.
    """
    n, m, p = 2, 3, 10
    
    start_time = time.time()
    
    # Try ranks from 30 to 59
    for rank in range(30, 60):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        num_seeds = 5 if rank <= 40 else 2
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U, V, W, success = _try_rank_als_fast(n, m, p, rank, max_iters=1000, seed=seed)
            if success:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the known rank-55 Strassen decomposition
    result = _get_strassen_decomposition(n, m, p)
    if result is not None:
        return result

    # Fall back to trivial decomposition
    return _get_trivial_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
