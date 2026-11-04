"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_START = time.time()
_DEADLINE = _START + 140.0

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_unified_4x4x8():
    """
    Unified Strassen decomposition for 4x4 * 4x8.
    
    View B as 4x4 matrix of 1x2 vectors, C as 4x4 matrix of 1x2 vectors.
    Apply Strassen at the 2x2 block level (blocks are 2x2 scalars for A,
    2x2 vectors for B/C).
    
    For each of the 7 Strassen terms, we need to multiply a 2x2 scalar
    matrix (from A-block combination) by a 2x2 vector matrix (from B-block
    combination) to get a 2x2 vector result. This requires rank 7*2=14.
    
    Total rank: 7*14 = 98, but with unified structure enabling potential reduction.
    
    U: (16, rank) indexed by (i,j), i,j in 0..3
    V: (32, rank) indexed by (j,k), j in 0..3, k in 0..7
    W: (32, rank) indexed by (k,i), k in 0..7, i in 0..3
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # For each Strassen term t (0..6), we need 14 rank-1 components.
    # The scalar part uses the 2x2x2 Strassen structure (rank 7 for 2x2*2x2).
    # The vector part: each "scalar" multiplication result is a 1x2 vector,
    # so we need 2 components per scalar multiplication.
    
    # Actually, let me think more carefully.
    # A is 4x4, partitioned into 2x2 blocks of 2x2 matrices:
    #   A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B is 4x8, partitioned into 2x2 blocks of 2x4 matrices:
    #   B = [[B00, B01], [B10, B11]] where each Bj is 2x4
    # C is 4x8, partitioned into 2x2 blocks of 2x4 matrices:
    #   C = [[C00, C01], [C10, C11]] where each Ck is 2x4
    #
    # Strassen for C = A*B:
    #   M1 = (A00+A11)(B00+B11)  -> 2x2 * 2x4 = 2x4
    #   M2 = (A10+A11)B00        -> 2x2 * 2x4 = 2x4
    #   M3 = A00(B01-B11)        -> 2x2 * 2x4 = 2x4
    #   M4 = A11(B10-B00)        -> 2x2 * 2x4 = 2x4
    #   M5 = (A00+A01)B11        -> 2x2 * 2x4 = 2x4
    #   M6 = (A10-A00)(B00+B01)  -> 2x2 * 2x4 = 2x4
    #   M7 = (A01-A11)(B10+B11)  -> 2x2 * 2x4 = 2x4
    #
    # Each Mk is 2x4. We can decompose each 2x2 * 2x4 as:
    #   Mk = (2x2 scalar mult) for each of the 2 columns of the 2x4 result
    #   So each Mk needs rank 7 (for 2x2*2x2) * 2 (for the 2 columns) = 14
    #
    # Total: 7 * 14 = 98
    #
    # But we can do better by noting that the 2x2*2x4 multiplication
    # can be seen as 2x2 * [2x2 | 2x2] = [2x2*2x2 | 2x2*2x2]
    # So we need two independent 2x2*2x2 Strassen decompositions.
    
    rank = 98
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)
    
    # Block structure:
    # i in 0..3: i_block = i//2 (0 or 1), i_local = i%2
    # j in 0..3: j_block = j//2 (0 or 1), j_local = j%2
    # k in 0..7: k_block = k//4 (0 or 1 for the 2x4 block), k_col = k%4
    #   Actually k in 0..7: the 4x8 matrix has rows j (0..3) and cols k (0..7)
    #   For the 2x2 block structure of B:
    #     B00: rows 0-1, cols 0-3
    #     B01: rows 0-1, cols 4-7
    #     B10: rows 2-3, cols 0-3
    #     B11: rows 2-3, cols 4-7
    #   So j_block = j//2, k_block = k//4, and within each block:
    #     j_local = j%2, k_local = k%4
    
    for t in range(7):  # Strassen term
        for sub in range(2):  # 2 columns of the 2x4 result (k_block)
            base = t * 14 + sub * 7
            
            # U factor: depends on t and the 2x2 scalar structure
            # For Strassen term t, the A-block combination gives a 2x2 matrix
            # We use the 2x2x2 Strassen U factor for the scalar part
            for i in range(4):
                i_block = i // 2
                i_local = i % 2
                U[i*4 + 0, base + 0] = 0  # placeholder
                # Actually, let me use the Kronecker product approach
                pass
            
            # Let me use a cleaner approach: build via Kronecker product
            # U[i,j] for the 4x4 A matrix
            # The 2x2 block structure: i = (i_block, i_local), j = (j_block, j_local)
            # U[i_block*2+i_local, j_block*2+j_local, t] = U2[i_block*2+i_local, j_block*2+j_local, t]
            # Wait, that's not right either. Let me think again.
            
            # The 4x4 matrix A has indices (i,j) where i,j in 0..3.
            # Block structure: A = [[A00, A01], [A10, A11]] where each is 2x2.
            # A[i,j] where i = 2*i_block + i_local, j = 2*j_block + j_local.
            # For Strassen term t, the coefficient of A[i_block, j_block] in the
            # 2x2 block combination is U2[i_block, j_block, t] (from the 2x2x2 Strassen).
            # But within each 2x2 block, we need the full 2x2 structure.
            # So U[i,j,t] = U2[i_block, j_block, t] * 1 (for the local 2x2 identity?)
            # No, that's not right. The 2x2 block combination means:
            #   (A00+A11)[i_local, j_local] = A00[i_local,j_local] + A11[i_local,j_local]
            # So for term t, the coefficient of A[i,j] is:
            #   sum over block combinations that include (i_block, j_block)
            #   with coefficient from U2[i_block, j_block, t]
            # And within the block, it's the identity (1 for matching local indices).
            
            # Actually, the correct way: the 2x2 block matrix for A in term t
            # is a 2x2 matrix M_t where M_t[i_block, j_block] = U2[i_block, j_block, t].
            # The actual 4x4 coefficient matrix is M_t ⊗ I_2 (Kronecker with 2x2 identity).
            # So U[i,j,t] = M_t[i_block, j_block] * (1 if i_local == j_local else 0)
            # Wait, no. The Kronecker product M ⊗ I gives:
            #   (M ⊗ I)[2*i_block+i_local, 2*j_block+j_local] = M[i_block,j_block] * I[i_local,j_local]
            # = M[i_block,j_block] if i_local == j_local, else 0.
            # That's not right for matrix multiplication. Let me reconsider.
            
            # Actually, when we write A = [[A00, A01], [A10, A11]] and do
            # (A00+A11), this means element-wise addition of the 2x2 blocks.
            # So (A00+A11)[i_local, j_local] = A00[i_local,j_local] + A11[i_local,j_local].
            # In the full 4x4 matrix, A[i,j] where i=2*ib+il, j=2*jb+jl:
            #   coefficient in (A00+A11) is 1 if (ib,jb)=(0,0) or (1,1), else 0.
            #   And it's 1 for all (il,jl) pairs (not just il==jl).
            # So the coefficient matrix for A in term t is M_t ⊗ J where J is 2x2 all-ones?
            # No, it's M_t ⊗ I_2 only if we're doing block-diagonal. But we're doing
            # block-wise operations, so it's M_t ⊗ (2x2 identity)? No...
            
            # Let me just use the Kronecker product approach directly.
            # The 4x4x4 tensor T[i,j,k] = delta_{i,k} (for 4x4*4x4 -> 4x4).
            # Wait, no. For matrix multiplication C = A*B where A is 4x4, B is 4x4:
            #   C[i,k] = sum_j A[i,j] * B[j,k]
            # The tensor is T[i,j,k] = delta_{i,k}... no, that's not right either.
            # The multiplication tensor has T[i,j,k] such that
            #   C[i,k] = sum_{j} A[i,j] * B[j,k]
            # So the tensor entry for (i,j,k) is 1 if the indices match the
            # multiplication pattern. Actually, the tensor is:
            #   T[i, j, k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
            # But that's not a tensor in the right sense. Let me use the standard
            # formulation: the multiplication tensor for n×n * n×n has
            #   T[i, j, k] = delta_{i,k} for the "diagonal" but that's schoolbook.
            # 
            # OK I'm overcomplicating this. Let me just directly construct the
            # rank-98 decomposition using the same approach as the current program
            # but with a unified structure, and then attempt rank reduction.
            
            pass
    
    # Let me just build it the same way as before (two blocks) but then
    # attempt to find linear dependencies for rank reduction.
    
    U4, V4, W4 = _build_4x4x4_tensors()
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49
        U[:, col_start:col_end] = U4
        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j*8 + k_global, col_start:col_end] = V4[j*4 + k_local, :]
        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global*4 + i, col_start:col_end] = W4[k_local*4 + i, :]
    
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _verify(U, V, W, n=4, m=4, p=8):
    """Verify exact decomposition. Returns True if valid."""
    rank = U.shape[1]
    # Use numpy for speed
    U = np.asarray(U)
    V = np.asarray(V)
    W = np.asarray(W)
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if total != 1:
                    return False
    return True

def _verify_fast(U, V, W, n=4, m=4, p=8):
    """Fast verification using matrix operations."""
    rank = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # For each r, the contribution to C[i,k] from A[i,j]*B[j,k] is:
    #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Sum over r should give 1 for all i,j,k (the multiplication tensor).
    
    # Build the tensor and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if val != 1.0:
                    return False
    return True

def _schoolbook(n=4, m=4, p=8):
    """Fallback schoolbook decomposition, rank 128."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int16)
    V = np.zeros((m*p, rank), dtype=np.int16)
    W = np.zeros((p*n, rank), dtype=np.int16)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _try_rank_reduction(U, V, W, n=4, m=4, p=8, max_time=30.0):
    """
    Attempt to reduce rank by finding linear dependencies among columns.
    
    The key insight: if we have a decomposition with rank R, and there exist
    coefficients c_r such that sum_r c_r * (U[:,r] ⊗ V[:,r] ⊗ W[:,r]) = 0,
    then we can eliminate one column.
    
    More practically: if two columns r1, r2 have the property that
    U[:,r1] = a*U[:,r2], V[:,r1] = b*V[:,r2], W[:,r1] = c*W[:,r2]
    with a*b*c = 1, then they can be merged.
    
    For a more general approach, we look for the minimal rank by checking
    if the tensor can be expressed with fewer terms.
    """
    if time.time() > _DEADLINE - 15:
        return U, V, W
    
    rank = U.shape[1]
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # Try to find a column that is a linear combination of others
    # in a way that preserves the tensor.
    
    # For each column r, check if there exist coefficients alpha_s (s != r)
    # such that for all i,j,k:
    #   U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] =
    #   sum_s alpha_s * U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # This is a system of n*m*p equations in (rank-1) unknowns.
    # For rank 98, this is 128 equations in 97 unknowns - overdetermined.
    # But we can try smaller subsets.
    
    # Alternative: try to find pairs of columns that can be merged.
    # If column r1 = a * column r2 in U, b * column r2 in V, c * column r2 in W
    # with a*b*c = 1, then they represent the same rank-1 tensor (up to scaling).
    
    # Check for proportional columns
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # For each pair (r1, r2), check if they are proportional
    for r1 in range(rank):
        if time.time() > _DEADLINE - 15:
            break
        for r2 in range(r1 + 1, rank):
            # Check if U[:,r1] is proportional to U[:,r2]
            if np.all(U[:, r1] == 0) and np.all(U[:, r2] == 0):
                continue
            if np.all(U[:, r1] == 0) or np.all(U[:, r2] == 0):
                continue
            
            # Find scaling factor
            nonzero = U[:, r1] != 0
            if not np.any(nonzero):
                continue
            a = U[nonzero, r1] / U[nonzero, r2] if np.any(U[nonzero, r2] != 0) else None
            if a is None or not np.allclose(U[nonzero, r1], a * U[nonzero, r2]):
                continue
            
            # Check V
            nonzero_v = V[:, r1] != 0
            if not np.any(nonzero_v):
                b = 0.0
            elif np.all(V[nonzero_v, r2] == 0):
                continue
            else:
                b = V[nonzero_v, r1] / V[nonzero_v, r2]
                if not np.allclose(V[nonzero_v, r1], b * V[nonzero_v, r2]):
                    continue
            
            # Check W
            nonzero_w = W[:, r1] != 0
            if not np.any(nonzero_w):
                c = 0.0
            elif np.all(W[nonzero_w, r2] == 0):
                continue
            else:
                c = W[nonzero_w, r1] / W[nonzero_w, r2]
                if not np.allclose(W[nonzero_w, r1], c * W[nonzero_w, r2]):
                    continue
            
            # Check a*b*c = 1
            if abs(a * b * c - 1.0) < 1e-10:
                # These two columns are proportional and can be merged
                # Remove r2 and scale r1 by (1 + a*b*c) = 2? No, just remove r2
                # Actually, if U[:,r1] = a*U[:,r2], V[:,r1] = b*V[:,r2], W[:,r1] = c*W[:,r2]
                # and a*b*c = 1, then the two rank-1 tensors are identical.
                # So we can just remove one of them.
                new_rank = rank - 1
                if new_rank < best_rank:
                    best_rank = new_rank
                    # Remove column r2
                    mask = [x for x in range(rank) if x != r2]
                    best_U = U[:, mask].copy()
                    best_V = V[:, mask].copy()
                    best_W = W[:, mask].copy()
                    U = best_U
                    V = best_V
                    W = best_W
                    rank = new_rank
                    break
        if time.time() > _DEADLINE - 15:
            break
    
    if best_rank < rank:
        return (best_U.astype(np.int16), best_V.astype(np.int16), best_W.astype(np.int16))
    
    return U.astype(np.int16), V.astype(np.int16), W.astype(np.int16)

def _try_3way_merge(U, V, W, n=4, m=4, p=8, max_time=20.0):
    """
    Try to find 3 columns that can be merged into 2 (or fewer) using
    a more general linear combination.
    """
    if time.time() > _DEADLINE - 10:
        return U, V, W
    
    rank = U.shape[1]
    if rank < 3:
        return U, V, W
    
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    # For triples (r1, r2, r3), check if there exist coefficients
    # alpha, beta such that:
    #   U[:,r3] = alpha*U[:,r1] + beta*U[:,r2]
    #   V[:,r3] = alpha*V[:,r1] + beta*V[:,r2]  (same alpha, beta)
    #   W[:,r3] = alpha*W[:,r1] + beta*W[:,r2]  (same alpha, beta)
    # If so, we can eliminate r3.
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # This is expensive, so only try a subset
    sample_size = min(rank, 20)
    indices = list(range(rank))
    
    for r3 in range(0, rank, 2):  # Try every other column as candidate for removal
        if time.time() > _DEADLINE - 10:
            break
        for r1 in range(0, r3, 3):
            if time.time() > _DEADLINE - 10:
                break
            for r2 in range(r1+1, r3, 3):
                if time.time() > _DEADLINE - 10:
                    break
                
                # Solve for alpha, beta from U
                # U[:,r3] = alpha*U[:,r1] + beta*U[:,r2]
                # Use least squares
                A_mat = np.column_stack([U[:, r1], U[:, r2]])
                b_vec = U[:, r3]
                
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    alpha, beta = coeffs
                except:
                    continue
                
                # Verify on V
                pred_V = alpha * V[:, r1] + beta * V[:, r2]
                if not np.allclose(pred_V, V[:, r3], atol=1e-10):
                    continue
                
                # Verify on W
                pred_W = alpha * W[:, r1] + beta * W[:, r2]
                if not np.allclose(pred_W, W[:, r3], atol=1e-10):
                    continue
                
                # Found a merge! Remove r3
                new_rank = rank - 1
                if new_rank < best_rank:
                    best_rank = new_rank
                    mask = [x for x in range(rank) if x != r3]
                    best_U = U[:, mask].copy()
                    best_V = V[:, mask].copy()
                    best_W = W[:, mask].copy()
                    U = best_U
                    V = best_V
                    W = best_W
                    rank = new_rank
                break
            if time.time() > _DEADLINE - 10:
                break
        if time.time() > _DEADLINE - 10:
            break
    
    if best_rank < rank:
        return (best_U.astype(np.int16), best_V.astype(np.int16), best_W.astype(np.int16))
    
    return U.astype(np.int16), V.astype(np.int16), W.astype(np.int16)

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Primary strategy: rank-98 Strassen-Kronecker decomposition
    try:
        U, V, W = _build_unified_4x4x8()
        if _verify(U, V, W, n, m, p):
            # Attempt rank reduction
            if time.time() < _DEADLINE - 20:
                U, V, W = _try_rank_reduction(U, V, W, n, m, p, max_time=20.0)
                if _verify(U, V, W, n, m, p):
                    pass  # Valid, use reduced rank
            
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook rank-128
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
