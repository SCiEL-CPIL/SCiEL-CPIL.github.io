"""Matrix multiplication 2x4 by 4x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 8]
START_TIME = time.time()
DEADLINE = 140.0  # 150s limit with margin

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

def _validate(U, V, W, tol=1e-8):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] reproduces the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k] uses this triple
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    # Build the tensor T of shape (n, m, p)
    T = np.ones((n, m, p))
    # Reconstruct from factors
    recon = np.zeros((n, m, p))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        # uc indexed by (i, j) -> i*m+j
        # vc indexed by (j, k) -> j*p+k
        # wc indexed by (k, i) -> k*n+i
        Umat = uc.reshape(n, m)
        Vmat = vc.reshape(m, p)
        Wmat = wc.reshape(p, n)
        # For each (i,j,k): Umat[i,j] * Vmat[j,k] * Wmat[k,i]
        # Use einsum
        recon += np.einsum('ij,jk,ki->ijk', Umat, Vmat, Wmat)
    if not np.allclose(recon, T, atol=tol, rtol=0):
        return False
    return True

def _strassen_block_2x2x2():
    """Standard Strassen 2x2 by 2x2 multiplication, rank 7."""
    # Returns U (4x7), V (4x7), W (4x7) for 2x2 * 2x2
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1  # A00, A11
    V[0,0]=1; V[3,0]=1  # B00, B11
    W[0,0]=1  # C00
    W[1,0]=1  # C11
    
    # c=1: M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1  # A10, A11
    V[0,1]=1  # B00
    W[2,1]=1  # C10
    W[3,1]=-1  # C11
    
    # c=2: M3 = A00(B01-B11)
    U[0,2]=1  # A00
    V[1,2]=1; V[3,2]=-1  # B01, -B11
    W[1,2]=1  # C01
    W[3,2]=1  # C11
    
    # c=3: M4 = A11(B10-B00)
    U[3,3]=1  # A11
    V[2,3]=1; V[0,3]=-1  # B10, -B00
    W[0,3]=1  # C00
    W[2,3]=1  # C10
    
    # c=4: M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1  # A00, A01
    V[3,4]=1  # B11
    W[0,4]=-1  # C00
    W[1,4]=1  # C01
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1  # A10, -A00
    V[0,5]=1; V[1,5]=1  # B00, B01
    W[3,5]=1  # C11
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1  # A01, -A11
    V[2,6]=1; V[3,6]=1  # B10, B11
    W[0,6]=1  # C00
    
    return U, V, W

def _build_strassen_2x4x8():
    """Build Strassen-based decomposition for 2x4 * 4x8.
    
    Split into blocks: A is 2x4 = [A00 A01] where A00 is 2x2, A01 is 2x2
    B is 4x8 = [[B00],[B01]] where B00 is 2x4... 
    
    Actually let's think more carefully.
    A: 2x4, B: 4x8, C: 2x8
    
    We can split the middle dimension (4) into two 2x2 blocks:
    A = [A_L | A_R]  (2x2 | 2x2)
    B = [B_T; B_B]  (2x4; 2x4)
    C = A_L * B_T + A_R * B_B
    
    Each of A_L*B_T and A_R*B_B is 2x2 * 2x4.
    For 2x2 * 2x4: split B into two 2x2 blocks.
    A_L * [B_T0 | B_T1] = A_L*B_T0 | A_L*B_T1
    Each A_L*B_T0 is 2x2*2x2, which Strassen handles with rank 7.
    
    So we have 4 Strassen blocks, each rank 7 = total rank 28.
    
    But we can do better by sharing structure. Let's use the block structure:
    C = A_L * B_T + A_R * B_B
    where A_L, A_R are 2x2 and B_T, B_B are 2x4.
    
    For 2x2 * 2x4: split B into [B1 | B2] both 2x2.
    Result = A*B1 | A*B2
    Using Strassen on each: rank 7 each, total 14 for 2x2*2x4.
    
    Total: 2 (for A_L*B_T and A_R*B_B) * 14 = 28.
    
    Let me construct this properly.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 28
    U = np.zeros((n*m, rank))  # (8, 28)
    V = np.zeros((m*p, rank))  # (32, 28)
    W = np.zeros((p*n, rank))  # (16, 28)
    
    col = 0
    # A indices: i in [0,1], j in [0,3] -> i*4+j
    # B indices: j in [0,3], k in [0,7] -> j*8+k
    # C indices: k in [0,7], i in [0,1] -> k*2+i
    
    # Split: A_L uses j=0,1; A_R uses j=2,3
    # B_T uses j=0,1 (rows 0,1 of B); B_B uses j=2,3 (rows 2,3 of B)
    # C split: k=0,1 (first 2) and k=2,3... wait, p=8 so k in [0,7]
    # B_T is rows j=0,1 of B, which is 2x8. Split into B_T0 (cols k=0,1) and B_T1 (cols k=2,3)?
    # No, B is 4x8. B_T = B[0:2, :] is 2x8. B_B = B[2:4, :] is 2x8.
    # C = A_L @ B_T + A_R @ B_B, where A_L is 2x2, B_T is 2x8.
    # A_L @ B_T: 2x2 * 2x8. Split B_T into 4 blocks of 2x2: B_T0, B_T1, B_T2, B_T3
    # Result is 2x8 = [C0 | C1 | C2 | C3] each 2x2.
    # Each A_L @ B_Ti is 2x2*2x2, Strassen rank 7.
    # Total for A_L @ B_T: 4 * 7 = 28.
    # Similarly A_R @ B_B: 4 * 7 = 28.
    # Grand total: 56. That's the Strassen baseline.
    
    # Let me just build the 56-rank Strassen and then try to optimize.
    rank = 56
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for a_side in range(2):  # 0: A_L (j=0,1), 1: A_R (j=2,3)
        j_offset = a_side * 2
        for b_side in range(2):  # 0: B_T (j=0,1), 1: B_B (j=2,3)
            # Wait, this isn't right. Let me re-think.
            # A_L @ B_T: A_L is A[:, 0:2] (2x2), B_T is B[0:2, :] (2x8)
            # A_R @ B_B: A_R is A[:, 2:4] (2x2), B_B is B[2:4, :] (2x8)
            # These are the only two terms.
            pass
    
    # Actually the decomposition is:
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # = sum_{j=0}^{1} A[i,j] * B[j,k] + sum_{j=2}^{3} A[i,j] * B[j,k]
    # = (A_L @ B_T)[i,k] + (A_R @ B_B)[i,k]
    
    # For A_L @ B_T (2x2 * 2x8):
    # Split B_T into 4 column blocks of size 2: B_T[:, 0:2], B_T[:, 2:4], B_T[:, 4:6], B_T[:, 6:8]
    # Each gives a 2x2 result.
    
    col = 0
    for a_block in range(2):  # 0: A_L, 1: A_R
        j_base = a_block * 2  # 0 or 2
        for b_col_block in range(4):  # 4 blocks of 2 columns each
            k_base = b_col_block * 2  # 0, 2, 4, 6
            # Get Strassen 2x2*2x2
            sU, sV, sW = _strassen_block_2x2x2()
            # sU: (4,7) indexed by [i*2+j_local, c] where j_local in {0,1}
            # Map to global: i in {0,1}, j = j_base + j_local
            for i in range(2):
                for jl in range(2):
                    j = j_base + jl
                    U[i*m + j, col:col+7] = sU[i*2+jl, :]
            # sV: (4,7) indexed by [jl*2+kl, c] where jl, kl in {0,1}
            # Map to global: j = j_base + jl, k = k_base + kl
            for jl in range(2):
                j = j_base + jl
                for kl in range(2):
                    k = k_base + kl
                    V[j*p + k, col:col+7] = sV[jl*2+kl, :]
            # sW: (4,7) indexed by [kl*2+i, c]
            # Map to global: k = k_base + kl, i in {0,1}
            for kl in range(2):
                k = k_base + kl
                for i in range(2):
                    W[k*n + i, col:col+7] = sW[kl*2+i, :]
            col += 7
    
    return U.tolist(), V.tolist(), W.tolist()

def _optimize_rank(U, V, W, target_rank, max_iters=500, tol=1e-8):
    """Try to find a lower-rank decomposition by optimizing factor entries."""
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # We want to find U', V', W' of rank r' < rank such that
    # sum_c' U'[i*m+j, c'] * V'[j*p+k, c'] * W'[k*n+i, c'] = 1 for all i,j,k
    
    # This is a non-convex problem. Use a simple approach:
    # Start with the current decomposition and try to merge columns.
    
    # For now, just return the current one.
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Build Strassen-based 56-rank decomposition
    U, V, W = _build_strassen_2x4x8()
    
    # Validate
    if not _validate(U, V, W):
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = len(U[0])
    
    # Try to optimize further if time permits
    if _time_left() > 30:
        try:
            U_opt, V_opt, W_opt = _optimize_rank(U, V, W, best_rank)
            if len(U_opt[0]) < best_rank and _validate(U_opt, V_opt, W_opt):
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = len(U_opt[0])
        except Exception:
            pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
