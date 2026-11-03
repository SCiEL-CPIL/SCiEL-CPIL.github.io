"""Rewritten program for 5x5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Build the target tensor for matrix multiplication."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                T[a*m + b, b*p + c, c*n + a] = 1
    return T

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _quantize_half_integer(mat):
    """Quantize matrix to nearest half-integers."""
    return np.round(np.array(mat, dtype=np.float64) * 2) / 2.0

def _als_step(U, V, W, T, n, m, p, rank, fix='W'):
    """
    Perform one ALS step, fixing the specified matrix and solving for the others.
    T: (nm, mp, pn) target tensor
    Returns updated U, V, W
    """
    nm, mp, pn = n*m, m*p, p*n
    
    if fix == 'W':
        # Fix W, solve for U and V jointly via least squares
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For each k: T[:,:,k] = sum_r (U[:,r] V[:,r]) W[k,r]
        # Let M_r = outer(U[:,r], V[:,r])
        # T[:,:,k] = sum_r M_r * W[k,r]
        # This is a bilinear problem. Use alternating:
        
        # Fix V, solve for U:
        # T[i,j,k] = sum_r U[i,r] (V[j,r] W[k,r])
        # Let B[j,k,r] = V[j,r] W[k,r]
        # T[i,j,k] = sum_r U[i,r] B[j,k,r]
        # For each i: U[i,:] is solution to: sum_r U[i,r] B[j,k,r] = T[i,j,k]
        # Vectorize: for fixed i, U[i,:] @ B_flat.T = T[i,:,:].flatten()
        # where B_flat has shape (mp*pn, rank)
        
        B = np.zeros((mp, pn, rank))
        for r in range(rank):
            B[:, :, r] = np.outer(V[:, r], W[:, r])
        B_flat = B.reshape(mp * pn, rank)
        
        # T[i, j, k] -> reshape to (nm, mp*pn)
        T_reshaped = T.reshape(nm, mp * pn)
        
        U_new, _, _, _ = np.linalg.lstsq(B_flat.T, T_reshaped.T, rcond=1e-12)
        U_new = U_new.T
        
        # Fix U, solve for V:
        # T[i,j,k] = sum_r (U[i,r] W[k,r]) V[j,r]
        # Let A[i,k,r] = U[i,r] W[k,r]
        # T[i,j,k] = sum_r A[i,k,r] V[j,r]
        # For each j: V[j,:] @ A_flat.T = T[:,j,:].flatten()
        
        A = np.zeros((nm, pn, rank))
        for r in range(rank):
            A[:, :, r] = np.outer(U_new[:, r], W[:, r])
        A_flat = A.reshape(nm * pn, rank)
        
        T_for_V = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
        
        V_new, _, _, _ = np.linalg.lstsq(A_flat.T, T_for_V.T, rcond=1e-12)
        V_new = V_new.T
        
        return U_new, V_new, W
    
    elif fix == 'U':
        # Fix U, solve for V and W
        B = np.zeros((mp, pn, rank))
        for r in range(rank):
            B[:, :, r] = np.outer(V[:, r], W[:, r])
        B_flat = B.reshape(mp * pn, rank)
        T_reshaped = T.reshape(nm, mp * pn)
        U_new, _, _, _ = np.linalg.lstsq(B_flat.T, T_reshaped.T, rcond=1e-12)
        U_new = U_new.T
        
        A = np.zeros((nm, pn, rank))
        for r in range(rank):
            A[:, :, r] = np.outer(U[:, r], W[:, r])
        A_flat = A.reshape(nm * pn, rank)
        T_for_V = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
        V_new, _, _, _ = np.linalg.lstsq(A_flat.T, T_for_V.T, rcond=1e-12)
        V_new = V_new.T
        
        # Fix U, V, solve for W
        C = np.zeros((nm, mp, rank))
        for r in range(rank):
            C[:, :, r] = np.outer(U[:, r], V[:, r])
        C_flat = C.reshape(nm * mp, rank)
        T_for_W = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
        W_new, _, _, _ = np.linalg.lstsq(C_flat.T, T_for_W.T, rcond=1e-12)
        W_new = W_new.T
        
        return U, V_new, W_new
    
    else:  # fix == 'V'
        # Fix V, solve for U and W
        B = np.zeros((mp, pn, rank))
        for r in range(rank):
            B[:, :, r] = np.outer(V[:, r], W[:, r])
        B_flat = B.reshape(mp * pn, rank)
        T_reshaped = T.reshape(nm, mp * pn)
        U_new, _, _, _ = np.linalg.lstsq(B_flat.T, T_reshaped.T, rcond=1e-12)
        U_new = U_new.T
        
        C = np.zeros((nm, mp, rank))
        for r in range(rank):
            C[:, :, r] = np.outer(U_new[:, r], V[:, r])
        C_flat = C.reshape(nm * mp, rank)
        T_for_W = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
        W_new, _, _, _ = np.linalg.lstsq(C_flat.T, T_for_W.T, rcond=1e-12)
        W_new = W_new.T
        
        return U_new, V, W_new

def _als_decompose(U_init, V_init, W_init, T, n, m, p, rank, max_iters=20, tol=1e-10):
    """
    Perform ALS iterations to refine a decomposition.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.array(U_init, dtype=np.float64).copy()
    V = np.array(V_init, dtype=np.float64).copy()
    W = np.array(W_init, dtype=np.float64).copy()
    
    for it in range(max_iters):
        # Cycle through fixing each matrix
        U, V, W = _als_step(U, V, W, T, n, m, p, rank, fix='W')
        U, V, W = _als_step(U, V, W, T, n, m, p, rank, fix='U')
        U, V, W = _als_step(U, V, W, T, n, m, p, rank, fix='V')
        
        # Check convergence
        T_computed = np.einsum('ic,jc,kc->ijk', U, V, W, optimize=True)
        err = np.max(np.abs(T_computed - T))
        
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank_als(target_rank, n, m, p, T, deadline, max_inits=30):
    """Try to find a valid decomposition of given rank using ALS with quantization."""
    if time.time() > deadline - 20:
        return None
    
    nm, mp, pn = n*m, m*p, p*n
    
    for init in range(max_inits):
        if time.time() > deadline - 15:
            break
        
        # Random initialization with half-integer values
        # Use different strategies
        strategy = init % 5
        
        if strategy == 0:
            # Sparse random
            U = np.random.choice([0, 0, 0, 0, 1, -1, 0.5, -0.5], size=(nm, target_rank))
            V = np.random.choice([0, 0, 0, 0, 1, -1, 0.5, -0.5], size=(mp, target_rank))
            W = np.random.choice([0, 0, 0, 0, 1, -1, 0.5, -0.5], size=(pn, target_rank))
        elif strategy == 1:
            # Dense small values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, target_rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, target_rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, target_rank))
        elif strategy == 2:
            # Structured blocks
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            for r in range(target_rank):
                u_idx = np.random.choice(nm, size=min(4, nm), replace=False)
                v_idx = np.random.choice(mp, size=min(4, mp), replace=False)
                w_idx = np.random.choice(pn, size=min(4, pn), replace=False)
                U[u_idx, r] = np.random.choice([-1, 1])
                V[v_idx, r] = np.random.choice([-1, 1])
                W[w_idx, r] = np.random.choice([-1, 1])
        elif strategy == 3:
            # Gaussian quantized
            U = np.round(np.random.randn(nm, target_rank) * 0.8) / 2
            V = np.round(np.random.randn(mp, target_rank) * 0.8) / 2
            W = np.round(np.random.randn(pn, target_rank) * 0.8) / 2
        else:
            # Mix of large and small
            U = np.where(np.random.rand(nm, target_rank) < 0.3, 
                        np.random.choice([-1, 1], size=(nm, target_rank)),
                        np.random.choice([-0.5, 0, 0.5], size=(nm, target_rank)))
            V = np.where(np.random.rand(mp, target_rank) < 0.3, 
                        np.random.choice([-1, 1], size=(mp, target_rank)),
                        np.random.choice([-0.5, 0, 0.5], size=(mp, target_rank)))
            W = np.where(np.random.rand(pn, target_rank) < 0.3, 
                        np.random.choice([-1, 1], size=(pn, target_rank)),
                        np.random.choice([-0.5, 0, 0.5], size=(pn, target_rank)))
        
        # Run ALS refinement
        U_ref, V_ref, W_ref, err = _als_decompose(U, V, W, T, n, m, p, target_rank, max_iters=15)
        
        # Quantize to half-integers
        U_q = _quantize_half_integer(U_ref)
        V_q = _quantize_half_integer(V_ref)
        W_q = _quantize_half_integer(W_ref)
        
        # Check if quantized solution works
        if _verify(U_q.tolist(), V_q.tolist(), W_q.tolist(), n, m, p, target_rank):
            return U_q.tolist(), V_q.tolist(), W_q.tolist()
        
        # Also try direct quantization of initial
        if _verify(_quantize_half_integer(U).tolist(), _quantize_half_integer(V).tolist(), _quantize_half_integer(W).tolist(), n, m, p, target_rank):
            return _quantize_half_integer(U).tolist(), _quantize_half_integer(V).tolist(), _quantize_half_integer(W).tolist()
    
    return None

def _strassen_2x2():
    """Build Strassen 2x2 decomposition (rank 7)."""
    # Strassen's algorithm for 2x2 matrix multiplication
    # C = AB where A, B are 2x2
    # Uses 7 multiplications instead of 8
    rank = 7
    nm = mp = pn = 4
    
    # U[i, r] where i = a*2 + b (row of A, col of B in input)
    # V[j, r] where j = b*2 + c
    # W[k, r] where k = c*2 + a
    
    # Strassen's 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1  # A11
    U[3, 0] = 1  # A22
    V[0, 0] = 1  # B11
    V[3, 0] = 1  # B22
    W[0, 0] = 1  # C11
    W[3, 0] = 1  # C22
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1  # A21
    U[3, 1] = 1  # A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1 # C22
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1  # A11
    V[1, 2] = 1  # B12
    V[3, 2] = -1 # B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1  # A22
    V[2, 3] = 1  # B21
    V[0, 3] = -1 # B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1  # A11
    U[1, 4] = 1  # A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1 # C11
    W[1, 4] = 1  # C12
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1  # A21
    U[0, 5] = -1 # A11
    V[0, 5] = 1  # B11
    V[1, 5] = 1  # B12
    W[3, 5] = 1  # C22
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1  # A12
    U[3, 6] = -1 # A22
    V[2, 6] = 1  # B21
    V[3, 6] = 1  # B22
    W[0, 6] = 1  # C11
    
    return U.tolist(), V.tolist(), W.tolist()

def _embed_2x2_into_5x5(U2, V2, W2, n=5):
    """Embed a 2x2 decomposition into 5x5 by padding with zeros."""
    nm = mp = pn = n * n
    rank = len(U2[0])
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Map 2x2 indices to 5x5 indices
    # For 2x2: i = a*2 + b, j = b*2 + c, k = c*2 + a where a,b,c in {0,1}
    # For 5x5: i = a*5 + b, j = b*5 + c, k = c*5 + a where a,b,c in {0,1,2,3,4}
    # We embed the 2x2 solution into the top-left 2x2 block
    
    for a in range(2):
        for b in range(2):
            for c in range(2):
                # 2x2 index
                i2 = a * 2 + b
                j2 = b * 2 + c
                k2 = c * 2 + a
                # 5x5 index (top-left 2x2 block)
                i5 = a * 5 + b
                j5 = b * 5 + c
                k5 = c * 5 + a
                
                for r in range(rank):
                    U[i5, r] = U2[i2][r]
                    V[j5, r] = V2[j2][r]
                    W[k5, r] = W2[k2][r]
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_construct():
    """Main construction function."""
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    T = _build_target(n, m, p)
    
    # Try known decompositions first
    # 1. Strassen 2x2 embedded (rank 7 for 2x2, but embedded in 5x5 it's still rank 7
    #    but only covers the 2x2 subproblem, so it won't work directly)
    #    Actually, embedding doesn't give a valid 5x5 decomposition.
    #    We need a proper 5x5 decomposition.
    
    # Try ranks from known bounds downward
    # Theoretical lower bound is around 104, upper bound is 125 (schoolbook)
    # Known best for 5x5 is around 104-110
    
    # Try ranks from 110 down to 100
    for target_rank in range(110, 99, -1):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank_als(target_rank, n, m, p, T, deadline, max_inits=20)
        if result is not None:
            U_r, V_r, W_r = result
            best_rank = target_rank
            best_result = {"U": U_r, "V": V_r, "W": W_r}
            print(f"Found valid rank {best_rank}")
            # Continue trying lower ranks
            break
    
    # If we found a rank, try to go lower
    if best_rank < 110:
        for target_rank in range(best_rank - 1, 99, -1):
            if time.time() > deadline - 10:
                break
            result = _try_rank_als(target_rank, n, m, p, T, deadline, max_inits=15)
            if result is not None:
                U_r, V_r, W_r = result
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                print(f"Found valid rank {best_rank}")
                break
    
    return best_result

def construct():
    """Entry point: construct the tensor decomposition."""
    try:
        result = _try_construct()
        return result
    except Exception as e:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        U_sb, V_sb, W_sb = _schoolbook(n, m, p)
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
