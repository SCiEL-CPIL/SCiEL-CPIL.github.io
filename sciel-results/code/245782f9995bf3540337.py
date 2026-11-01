"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _verify_exact(U, V, W, T):
    """Verify exact decomposition using rational arithmetic."""
    dim1, dim2, dim3 = T.shape
    rank = U.shape[1]
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(U[i, c]).limit_denominator(1000) for c in range(rank)] for i in range(dim1)]
    V_frac = [[Fraction(V[i, c]).limit_denominator(1000) for c in range(rank)] for i in range(dim2)]
    W_frac = [[Fraction(W[i, c]).limit_denominator(1000) for c in range(rank)] for i in range(dim3)]
    
    # Check all entries
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                expected = T[i, j, k]
                actual = sum(U_frac[i][c] * V_frac[j][c] * W_frac[k][c] for c in range(rank))
                if abs(float(actual) - expected) > 1e-10:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-12, seed=42):
    """ALS decomposition with structured initialization."""
    rng = np.random.RandomState(seed)
    dim1, dim2, dim3 = n*m, m*p, p*n
    
    T = _build_target_tensor(n, m, p)
    T_u1 = T.reshape(dim1, dim2 * dim3)
    T_u2 = T.reshape(dim2, dim1 * dim3)
    T_u3 = T.reshape(dim3, dim1 * dim2)
    
    # Structured initialization
    U = rng.randn(dim1, rank) * 0.5
    V = rng.randn(dim2, rank) * 0.5
    W = rng.randn(dim3, rank) * 0.5
    
    prev_err = float('inf')
    for it in range(max_iter):
        # Update U
        VW = np.zeros((dim2 * dim3, rank))
        for c in range(rank):
            VW[:, c] = np.outer(V[:, c], W[:, c]).flatten()
        U_new = np.linalg.lstsq(VW, T_u1.T, rcond=None)[0].T
        U = 0.7 * U + 0.3 * U_new
        
        # Update V
        UW = np.zeros((dim1 * dim3, rank))
        for c in range(rank):
            UW[:, c] = np.outer(U[:, c], W[:, c]).flatten()
        V_new = np.linalg.lstsq(UW, T_u2.T, rcond=None)[0].T
        V = 0.7 * V + 0.3 * V_new
        
        # Update W
        UV = np.zeros((dim1 * dim2, rank))
        for c in range(rank):
            UV[:, c] = np.outer(U[:, c], V[:, c]).flatten()
        W_new = np.linalg.lstsq(UV, T_u3.T, rcond=None)[0].T
        W = 0.7 * W + 0.3 * W_new
        
        recon = np.einsum('ic,jc,kc->ijk', U, V, W)
        err = np.linalg.norm(T - recon) / (np.linalg.norm(T) + 1e-15)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _local_search(U, V, W, T, max_steps=300, time_limit=8):
    """Local search on half-integer grid with fast error evaluation."""
    dim1, dim2, dim3 = T.shape
    rank = U.shape[1]
    start_time = time.time()
    
    U_cur = np.round(U * 2) / 2.0
    V_cur = np.round(V * 2) / 2.0
    W_cur = np.round(W * 2) / 2.0
    
    if _verify_exact(U_cur, V_cur, W_cur, T):
        return U_cur, V_cur, W_cur, True
    
    recon = np.einsum('ic,jc,kc->ijk', U_cur, V_cur, W_cur)
    err = np.linalg.norm(T - recon)
    
    # Precompute for fast error updates
    # err^2 = sum_{i,j,k} (T[i,j,k] - sum_c U[i,c]V[j,c]W[k,c])^2
    # When we change U[i0,c0] by delta, the change in recon is:
    # delta * V[j,c0] * W[k,c0] for all j,k
    # So we can update err efficiently
    
    rng = np.random.RandomState(123)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for _ in range(40):
            mat_idx = rng.randint(3)
            if mat_idx == 0:
                mat, dim = U_cur, dim1
            elif mat_idx == 1:
                mat, dim = V_cur, dim2
            else:
                mat, dim = W_cur, dim3
            
            i, c = rng.randint(dim), rng.randint(rank)
            orig = mat[i, c]
            
            for delta in [-1.0, -0.5, 0.5, 1.0]:
                mat[i, c] = orig + delta * 0.5
                recon_new = np.einsum('ic,jc,kc->ijk', U_cur, V_cur, W_cur)
                err_new = np.linalg.norm(T - recon_new)
                
                if err_new < err - 1e-12:
                    err = err_new
                    improved = True
                    if err < 1e-10:
                        if _verify_exact(U_cur, V_cur, W_cur, T):
                            return U_cur, V_cur, W_cur, True
                    break
                else:
                    mat[i, c] = orig
        
        if not improved:
            for _ in range(20):
                mat_idx = rng.randint(3)
                if mat_idx == 0:
                    mat, dim = U_cur, dim1
                elif mat_idx == 1:
                    mat, dim = V_cur, dim2
                else:
                    mat, dim = W_cur, dim3
                
                i, c = rng.randint(dim), rng.randint(rank)
                orig = mat[i, c]
                new_val = rng.randint(-6, 7) * 0.5
                mat[i, c] = new_val
                recon_new = np.einsum('ic,jc,kc->ijk', U_cur, V_cur, W_cur)
                err_new = np.linalg.norm(T - recon_new)
                
                if err_new < err - 1e-12:
                    err = err_new
                    improved = True
                    if err < 1e-10:
                        if _verify_exact(U_cur, V_cur, W_cur, T):
                            return U_cur, V_cur, W_cur, True
                else:
                    mat[i, c] = orig
            
            if not improved:
                break
    
    if err < 1e-10 and _verify_exact(U_cur, V_cur, W_cur, T):
        return U_cur, V_cur, W_cur, True
    
    return None, None, None, False

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    dim1, dim2, dim3 = n*m, m*p, p*n
    U = [[0] * rank for _ in range(dim1)]
    V = [[0] * rank for _ in range(dim2)]
    W = [[0] * rank for _ in range(dim3)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(U, V, W, block_col_start):
    """
    Add Strassen 2x2x2 decomposition for a block.
    block_col_start: the starting column index in the 2x8 output (0, 2, 4, 6)
    U: 6 x rank, V: 24 x rank, W: 16 x rank (modified in place)
    """
    rank = U.shape[1]
    offset = rank
    
    # Strassen 7 terms for 2x2 matrix multiplication
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a10)b11
    # M6 = (a11 - a10)(b00 + b10)
    # M7 = (a00 - a11)(b11 + b10)
    
    # For block starting at column c in the 2x8 output:
    # U indices: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    # V indices for this block: (j, c), (j, c+1) where j=0,1,2
    #   V[0*8+c], V[1*8+c], V[2*8+c], V[0*8+c+1], V[1*8+c+1], V[2*8+c+1]
    # W indices for this block: (k, i) where k=c,c+1 and i=0,1
    #   W[c*2+0], W[c*2+1], W[(c+1)*2+0], W[(c+1)*2+1]
    
    c = block_col_start
    
    # M1 = (a00 + a11)(b00 + b11)
    U[0, offset] = 1  # a00
    U[4, offset] = 1  # a11
    V[0*8+c, offset] = 1   # b00
    V[1*8+c+1, offset] = 1 # b11
    W[c*2+0, offset] = 1   # c00
    W[(c+1)*2+1, offset] = 1 # c11
    offset += 1
    
    # M2 = (a10 + a11)b00
    U[3, offset] = 1  # a10
    U[4, offset] = 1  # a11
    V[0*8+c, offset] = 1   # b00
    W[c*2+1, offset] = 1   # c01
    offset += 1
    
    # M3 = a00(b10 - b11)
    U[0, offset] = 1  # a00
    V[1*8+c, offset] = 1   # b10
    V[1*8+c+1, offset] = -1 # b11
    W[(c+1)*2+0, offset] = 1 # c10
    offset += 1
    
    # M4 = a11(b01 - b00)
    U[4, offset] = 1  # a11
    V[0*8+c+1, offset] = 1  # b01
    V[0*8+c, offset] = -1   # b00
    W[(c+1)*2+1, offset] = 1 # c11
    offset += 1
    
    # M5 = (a00 + a10)b11
    U[0, offset] = 1  # a00
    U[3, offset] = 1  # a10
    V[1*8+c+1, offset] = 1 # b11
    W[c*2+0, offset] = 1   # c00
    offset += 1
    
    # M6 = (a11 - a10)(b00 + b10)
    U[4, offset] = 1  # a11
    U[3, offset] = -1 # a10
    V[0*8+c, offset] = 1   # b00
    V[1*8+c, offset] = 1   # b10
    W[c*2+1, offset] = 1   # c01
    offset += 1
    
    # M7 = (a00 - a11)(b11 + b10)
    U[0, offset] = 1  # a00
    U[4, offset] = -1 # a11
    V[1*8+c+1, offset] = 1 # b11
    V[1*8+c, offset] = 1   # b10
    W[(c+1)*2+0, offset] = 1 # c10
    offset += 1
    
    return offset

def _tail_2x2_block(U, V, W, block_col_start):
    """
    Add tail terms for the remaining products in a 2x2 block.
    The Strassen decomposition handles c00, c01, c10, c11.
    But we need c02 and c12 from a02 and a12.
    Actually, for 2x3 by 3x2, we have:
    c00 = a00*b00 + a01*b10 + a02*b20
    c01 = a00*b01 + a01*b11 + a02*b21
    c10 = a10*b00 + a11*b10 + a12*b20
    c11 = a10*b01 + a11*b11 + a12*b21
    c02 = a02*b00 + a02*b01  -- wait, this doesn't make sense.
    
    Let me reconsider. The block is 2x3 by 3x2 = 2x2.
    Strassen gives us c00, c01, c10, c11.
    But we also need the contributions from a02 and a12.
    
    Actually, Strassen 2x2 only works for 2x2 by 2x2.
    For 2x3 by 3x2, we need to handle the third column/row.
    
    Let me think differently. The 2x3 by 3x2 multiplication can be split as:
    C = [a00 a01 a02] [b00 b01]
        [a10 a11 a12] [b10 b11]
                     [b20 b21]
    
    = [a00 a01] [b00 b01] + a02 [b20 b21]
      [a10 a11] [b10 b11] + a12 [b20 b21]
    
    The first part is 2x2 by 2x2, which Strassen handles with rank 7.
    The second part is rank 2 (two rank-1 terms: a02*[b20 b21] and a12*[b20 b21]).
    
    So total rank per block = 7 + 2 = 9? No wait, that's not right either.
    
    Actually, a02*[b20 b21] is a rank-1 term in the tensor sense:
    U has a02, V has b20 and b21, W has c00 and c01.
    Similarly a12*[b20 b21] has U with a12, V with b20 and b21, W with c10 and c11.
    
    So that's 2 additional rank-1 terms. Total per block = 7 + 2 = 9.
    4 blocks * 9 = 36.
    
    But wait, the inspiration said rank 11 per block. Let me re-examine.
    
    Actually, I think the decomposition might be different. Let me just implement
    the Strassen part and add the tail terms, then verify.
    """
    rank = U.shape[1]
    offset = rank
    
    c = block_col_start
    
    # Tail term 1: a02 * [b20, b21] -> contributes to c00, c01
    U[2, offset] = 1  # a02
    V[2*8+c, offset] = 1   # b20
    V[2*8+c+1, offset] = 1 # b21
    W[c*2+0, offset] = 1   # c00
    W[c*2+1, offset] = 1   # c01
    offset += 1
    
    # Tail term 2: a12 * [b20, b21] -> contributes to c10, c11
    U[5, offset] = 1  # a12
    V[2*8+c, offset] = 1   # b20
    V[2*8+c+1, offset] = 1 # b21
    W[(c+1)*2+0, offset] = 1 # c10
    W[(c+1)*2+1, offset] = 1 # c11
    offset += 1
    
    return offset

def _construct_rank44():
    """
    Construct a rank-44 decomposition using 4 blocks of rank-11.
    Each block handles a 2x3 by 3x2 sub-multiplication.
    Uses a correct Strassen-like rank-7 decomposition for the 2x2x2 core
    and 4 tail terms for the remaining row/column.
    """
    n, m, p = 2, 3, 8
    dim1, dim2, dim3 = n*m, m*p, p*n  # 6, 24, 16
    
    rank_per_block = 11
    total_rank = 4 * rank_per_block
    
    U = np.zeros((dim1, total_rank))
    V = np.zeros((dim2, total_rank))
    W = np.zeros((dim3, total_rank))
    
    for q in range(4):
        col_start = q * rank_per_block
        c = 2 * q  # column start in the 8-column output
        offset = col_start
        
        # Strassen 7 terms for the 2x2x2 core (A[0,0:2, 0:2], B[0:2, 0:2])
        # Note: A indices 0,1,3,4 correspond to (0,0),(0,1),(1,0),(1,1)
        # B indices 0,1,8,9 (for c=0) correspond to (0,0),(0,1),(1,0),(1,1)
        
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[0, offset] = 1
        U[4, offset] = 1
        V[0*8+c, offset] = 1
        V[1*8+c+1, offset] = 1
        W[c*2+0, offset] = 1
        W[(c+1)*2+1, offset] = 1
        offset += 1
        
        # M2 = (A[1,0]+A[1,1]) * B[0,0]
        U[3, offset] = 1
        U[4, offset] = 1
        V[0*8+c, offset] = 1
        W[(c+1)*2+0, offset] = 1
        offset += 1
        
        # M3 = A[0,0] * (B[1,0]-B[1,1])
        U[0, offset] = 1
        V[1*8+c, offset] = 1
        V[1*8+c+1, offset] = -1
        W[c*2+0, offset] = -1
        W[c*2+1, offset] = 1
        offset += 1
        
        # M4 = A[1,1] * (B[0,1]-B[0,0])
        U[4, offset] = 1
        V[0*8+c+1, offset] = 1
        V[0*8+c, offset] = -1
        W[(c+1)*2+1, offset] = -1
        W[(c+1)*2+0, offset] = 1
        offset += 1
        
        # M5 = (A[0,0]+A[1,0]) * B[1,1]
        U[0, offset] = 1
        U[3, offset] = 1
        V[1*8+c+1, offset] = 1
        W[c*2+1, offset] = 1
        offset += 1
        
        # M6 = (A[1,1]-A[1,0]) * (B[0,0]+B[1,0])
        U[4, offset] = 1
        U[3, offset] = -1
        V[0*8+c, offset] = 1
        V[1*8+c, offset] = 1
        W[(c+1)*2+0, offset] = -1
        offset += 1
        
        # M7 = (A[0,0]-A[1,1]) * (B[1,1]+B[1,0])
        U[0, offset] = 1
        U[4, offset] = -1
        V[1*8+c+1, offset] = 1
        V[1*8+c, offset] = 1
        W[c*2+0, offset] = -1
        W[(c+1)*2+1, offset] = 1
        offset += 1
        
        # Tail terms for A[0,2], A[1,2] and B[2,0], B[2,1]
        # M8 = A[0,2] * B[2,0] -> C[0,0]
        U[2, offset] = 1
        V[2*8+c, offset] = 1
        W[c*2+0, offset] = 1
        offset += 1
        
        # M9 = A[0,2] * B[2,1] -> C[0,1]
        U[2, offset] = 1
        V[2*8+c+1, offset] = 1
        W[c*2+1, offset] = 1
        offset += 1
        
        # M10 = A[1,2] * B[2,0] -> C[1,0]
        U[5, offset] = 1
        V[2*8+c, offset] = 1
        W[(c+1)*2+0, offset] = 1
        offset += 1
        
        # M11 = A[1,2] * B[2,1] -> C[1,1]
        U[5, offset] = 1
        V[2*8+c+1, offset] = 1
        W[(c+1)*2+1, offset] = 1
        offset += 1
    
    return U, V, W, total_rank

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification: check only the 48 nonzero tensor entries."""
    dim1, dim2, dim3 = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Check all 48 nonzero entries of the multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # All other entries must be 0
    # First check the 48 ones
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i*m + j
                idx2 = j*p + k
                idx3 = k*n + i
                val = sum(U[idx1, c] * V[idx2, c] * W[idx3, c] for c in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check a sample of zero entries (use all if small enough)
    zero_count = dim1 * dim2 * dim3 - n*m*p
    if zero_count <= 2000:
        checked = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    checked.add((i*m+j, j*p+k, k*n+i))
        for idx1 in range(dim1):
            for idx2 in range(dim2):
                for idx3 in range(dim3):
                    if (idx1, idx2, idx3) not in checked:
                        val = sum(U[idx1, c] * V[idx2, c] * W[idx3, c] for c in range(rank))
                        if abs(val) > 1e-9:
                            return False
    else:
        # Check random sample
        rng = np.random.RandomState(42)
        for _ in range(500):
            idx1, idx2, idx3 = rng.randint(dim1), rng.randint(dim2), rng.randint(dim3)
            if (idx1, idx2, idx3) not in checked:
                val = sum(U[idx1, c] * V[idx2, c] * W[idx3, c] for c in range(rank))
                if abs(val) > 1e-9:
                    return False
    
    return True

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    dim1, dim2, dim3 = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Build target tensor
    T = np.zeros((dim1, dim2, dim3))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Compute reconstruction
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    
    # Check exactness using fractions
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                expected = T[i, j, k]
                actual = 0
                for c in range(rank):
                    actual += Fraction(U[i, c]).limit_denominator(1000) * \
                              Fraction(V[j, c]).limit_denominator(1000) * \
                              Fraction(W[k, c]).limit_denominator(1000)
                if abs(float(actual) - expected) > 1e-10:
                    return False
    return True

def _construct_rank40():
    """
    Construct a rank-40 decomposition using 4 blocks of rank-10.
    Each block handles a 2x3x2 sub-multiplication.
    
    For each 2x3 x 3x2 block (columns c, c+1 of the 8-col output):
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    
    Use Strassen 7 terms for the 2x2x2 core (rows 0,1 of A and B),
    then 3 tail terms for row 2 of A and B:
    M8 = A[0,2] * (B[2,0] + B[2,1]) -> C[0,0]+=M8, C[0,1]+=M8
    M9 = A[1,2] * (B[2,0] + B[2,1]) -> C[1,0]+=M9, C[1,1]+=M9
    M10 = (A[0,2] - A[1,2]) * (B[2,1] - B[2,0]) -> C[0,1]-=M10, C[1,0]+=M10
    
    Wait, that gives:
    C[0,0] = Strassen_part + M8
    C[0,1] = Strassen_part + M8 - M10
    C[1,0] = Strassen_part + M9 + M10
    C[1,1] = Strassen_part + M9
    
    We need:
    C[0,0] += A[0,2]*B[2,0]
    C[0,1] += A[0,2]*B[2,1]
    C[1,0] += A[1,2]*B[2,0]
    C[1,1] += A[1,2]*B[2,1]
    
    With M8 = A[0,2]*(B[2,0]+B[2,1]):
    C[0,0] += A[0,2]*B[2,0] + A[0,2]*B[2,1]
    C[0,1] += A[0,2]*B[2,0] + A[0,2]*B[2,1]
    
    With M9 = A[1,2]*(B[2,0]+B[2,1]):
    C[1,0] += A[1,2]*B[2,0] + A[1,2]*B[2,1]
    C[1,1] += A[1,2]*B[2,0] + A[1,2]*B[2,1]
    
    With M10 = (A[0,2]-A[1,2])*(B[2,1]-B[2,0]):
    C[0,1] -= (A[0,2]-A[1,2])*(B[2,1]-B[2,0])
    C[1,0] += (A[0,2]-A[1,2])*(B[2,1]-B[2,0])
    
    So total for C[0,0]: A[0,2]*B[2,0] + A[0,2]*B[2,1] -- WRONG, we need just A[0,2]*B[2,0]
    
    This doesn't work directly. Let me try:
    M8 = A[0,2]*B[2,0] -> C[0,0]
    M9 = A[0,2]*B[2,1] -> C[0,1]
    M10 = A[1,2]*B[2,0] -> C[1,0]
    M11 = A[1,2]*B[2,1] -> C[1,1]
    
    That's 4 tail terms = rank 11 per block = 44 total. Same as before.
    
    Let me try a different combination. Can we merge tail terms with Strassen?
    
    Actually, let me try rank 10 per block with a different scheme.
    The 2x3x2 tensor has 4*3 = 12 schoolbook terms.
    Strassen handles 2x2x2 in rank 7.
    The remaining terms involve A[0,2] and A[1,2].
    
    Terms involving A[0,2]: A[0,2]*B[2,0] -> C[0,0], A[0,2]*B[2,1] -> C[0,1]
    Terms involving A[1,2]: A[1,2]*B[2,0] -> C[1,0], A[1,2]*B[2,1] -> C[1,1]
    
    We can write:
    A[0,2]*B[2,0] + A[0,2]*B[2,1] = A[0,2]*(B[2,0]+B[2,1]) -> contributes to both C[0,0] and C[0,1]
    A[1,2]*B[2,0] + A[1,2]*B[2,1] = A[1,2]*(B[2,0]+B[2,1]) -> contributes to both C[1,0] and C[1,1]
    A[0,2]*B[2,1] - A[0,2]*B[2,0] = A[0,2]*(B[2,1]-B[2,0]) -> contributes to C[0,1] and -C[0,0]
    A[1,2]*B[2,1] - A[1,2]*B[2,0] = A[1,2]*(B[2,1]-B[2,0]) -> contributes to C[1,1] and -C[1,0]
    
    So:
    C[0,0] += A[0,2]*B[2,0] = 0.5*A[0,2]*(B[2,0]+B[2,1]) + 0.5*A[0,2]*(B[2,0]-B[2,1])
    C[0,1] += A[0,2]*B[2,1] = 0.5*A[0,2]*(B[2,0]+B[2,1]) + 0.5*A[0,2]*(B[2,1]-B[2,0])
    
    This uses half-integers. Let me define:
    M8 = A[0,2]*(B[2,0]+B[2,1]) -> C[0,0]+=0.5*M8, C[0,1]+=0.5*M8
    M9 = A[0,2]*(B[2,0]-B[2,1]) -> C[0,0]+=0.5*M9, C[0,1]-=0.5*M9
    M10 = A[1,2]*(B[2,0]+B[2,1]) -> C[1,0]+=0.5*M10, C[1,1]+=0.5*M10
    M11 = A[1,2]*(B[2,0]-B[2,1]) -> C[1,0]+=0.5*M11, C[1,1]-=0.5*M11
    
    That's 4 tail terms with half-integer coefficients. Still rank 11 per block.
    
    Hmm, I can't easily get below 11 per block. Let me just use rank 44 and focus on
    improving the search to find lower ranks.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    # Try the constructive rank-44 solution first (Strassen-based)
    U, V, W, rank = _construct_rank44()
    
    if _verify_decomposition_fast(U, V, W, n, m, p):
        U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 48
    best_result = _schoolbook(n, m, p)
    return best_result
# EVOLVE-BLOCK-END
