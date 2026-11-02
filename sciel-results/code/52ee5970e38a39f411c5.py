"""Matrix multiplication 2x5x6 tensor decomposition via structured block search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
DEADLINE_SEC = 140.0  # safety margin below 150s evaluator kill

# --- Known rank-7 Strassen-style decomposition for 2x2x2 ---
# The 2x2x2 matmul tensor has border rank 7 (Strassen).
# We use the standard 7-term decomposition.

def _strassen_222():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Convention: C[i,k] = sum_r U[i,k? no...]
    
    Actually for M2,2,2: C[i,k] = sum_j A[i,j] B[j,k]
    Tensor T[i,j,k] = delta_ij? No. T[i,j,k] = 1 if this term appears.
    Decomposition: T = sum_r U[r][i,j] * V[r][j,k] ... 
    Standard: C[i,k] = sum_j A[i,j]*B[j,k]
    Term r contributes: A_row_r(i) * B_col_r(k) -> C[i,k] with coeff W_r(i,k)
    
    U[r, i*m+j] for A indices, V[r, j*p+k] for B indices, W[r, k*n+i] for C.
    For n=m=p=2:
    U[r, i*2+j], V[r, j*2+k], W[r, k*2+i]
    """
    # Strassen's 7 products for 2x2:
    # M1 = A[0,0]*(B[0,0]-B[0,1]) -> C[0,0] += M1, C[0,1] += M1
    # M2 = A[0,0]+A[0,1]*B[1,1]  -> C[0,1] += M2... 
    # Let me use the standard form:
    # M1 = A11*(B11 - B21)
    # M2 = (A11 + A21)*B22  ... hmm indices get confusing.
    
    # Standard Strassen (0-indexed, A[i,j], B[j,k], C[i,k]):
    # M1 = A[0,0]*(B[0,0] - B[0,1])
    # M2 = (A[0,0] + A[0,1])*B[1,1]
    # M3 = (A[1,0] + A[1,1])*B[0,0]
    # M4 = A[1,1]*(B[1,0] - B[0,0])
    # M5 = (A[0,0] + A[1,1])*(B[0,0] + B[1,1])
    # M6 = (A[0,1] - A[1,1])*(B[1,0] + B[1,1])
    # M7 = (A[1,0] - A[0,0])*(B[0,0] + B[0,1])
    
    # C[0,0] = M1 - M2 + M3 + M5
    # C[0,1] = M2 + M4
    # C[1,0] = M5 + M6
    # C[1,1] = -M1 + M3 - M4 + M5
    
    # Build U, V, W for each term r in 0..6:
    # For term r: U[r, i*2+j] = a_ri,j ; V[r, j*2+k] = b_rj,k ; W[r, k*2+i] = c_rik
    U = np.zeros((7, 4), dtype=float)
    V = np.zeros((7, 4), dtype=float)
    W = np.zeros((7, 4), dtype=float)
    
    # Term 0: M1 = A[0,0]*(B[0,0] - B[0,1])
    U[0, 0] = 1   # A[0,0]
    V[0, 0] = 1   # B[0,0]
    V[0, 1] = -1  # B[0,1]
    W[0, 0] = 1   # C[0,0]
    W[0, 2] = 1   # C[0,1] (k=0,i=1? no. k*n+i: k=0,i=1 -> 0*2+1=1? wait n=2 so k*n+i)
    # W[r, k*n + i] where n=2 (output rows). C[i,k]: i is row, k is col.
    # W index: k*n + i. So C[0,0] -> k=0,i=0 -> 0. C[0,1] -> k=1,i=0 -> 2.
    # C[1,0] -> k=0,i=1 -> 1. C[1,1] -> k=1,i=1 -> 3.
    W[0, 0] = 1   # C[0,0]
    W[0, 2] = 1   # C[0,1]
    
    # Term 1: M2 = (A[0,0] + A[0,1])*B[1,1]
    U[1, 0] = 1   # A[0,0]
    U[1, 1] = 1   # A[0,1]
    V[1, 3] = 1   # B[1,1]
    W[1, 0] = -1  # C[0,0]
    W[1, 2] = 1   # C[0,1]
    
    # Term 2: M3 = (A[1,0] + A[1,1])*B[0,0]
    U[2, 2] = 1   # A[1,0]
    U[2, 3] = 1   # A[1,1]
    V[2, 0] = 1   # B[0,0]
    W[2, 0] = 1   # C[0,0]
    W[2, 1] = 1   # C[1,0]
    
    # Term 3: M4 = A[1,1]*(B[1,0] - B[0,0])
    U[3, 3] = 1   # A[1,1]
    V[3, 2] = 1   # B[1,0]
    V[3, 0] = -1  # B[0,0]
    W[3, 2] = 1   # C[0,1]
    W[3, 3] = -1  # C[1,1]
    
    # Term 4: M5 = (A[0,0] + A[1,1])*(B[0,0] + B[1,1])
    U[4, 0] = 1   # A[0,0]
    U[4, 3] = 1   # A[1,1]
    V[4, 0] = 1   # B[0,0]
    V[4, 3] = 1   # B[1,1]
    W[4, 0] = 1   # C[0,0]
    W[4, 1] = 1   # C[1,0]
    W[4, 2] = 1   # C[0,1]  wait no. C[0,1] is k=1,i=0 -> index 2. 
    # Actually: C[0,0]=M1-M2+M3+M5, C[0,1]=M2+M4, C[1,0]=M5+M6, C[1,1]=-M1+M3-M4+M5
    # So M5 contributes to C[0,0], C[1,0], C[1,1]. Not C[0,1].
    W[4, 0] = 1   # C[0,0]
    W[4, 1] = 1   # C[1,0]
    W[4, 3] = 1   # C[1,1]
    
    # Term 5: M6 = (A[0,1] - A[1,1])*(B[1,0] + B[1,1])
    U[5, 1] = 1   # A[0,1]
    U[5, 3] = -1  # A[1,1]
    V[5, 2] = 1   # B[1,0]
    V[5, 3] = 1   # B[1,1]
    W[5, 1] = 1   # C[1,0]
    
    # Term 6: M7 = (A[1,0] - A[0,0])*(B[0,0] + B[0,1])
    U[6, 2] = 1   # A[1,0]
    U[6, 0] = -1  # A[0,0]
    V[6, 0] = 1   # B[0,0]
    V[6, 1] = 1   # B[0,1]
    W[6, 1] = 1   # C[1,0]
    # Wait, M7: C[1,1] -= M1... let me recheck.
    # C[1,1] = -M1 + M3 - M4 + M5. M7 doesn't appear in C? That can't be right.
    # I think I mislabeled. Let me use the correct Strassen decomposition.
    
    # Actually the standard 7-term Strassen for 2x2 is:
    # Let A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = a*(e-f), M2 = (a+b)*h, M3 = (c+d)*e, M4 = d*(g-e)
    # M5 = (a+d)*(e+h), M6 = (b-d)*(g+h), M7 = (c-a)*(e+f)
    # C = [[M1 - M2 + M3 + M5, M2 + M4], [M5 + M6, -M1 + M3 - M4 + M5]]
    
    # So:
    # C[0,0] = M1 - M2 + M3 + M5
    # C[0,1] = M2 + M4
    # C[1,0] = M5 + M6
    # C[1,1] = -M1 + M3 - M4 + M5
    
    # M7 does NOT appear in C directly?? That seems wrong. Let me verify.
    # Actually I think there's a different standard form. Let me use a known correct one.
    
    # Let me just verify computationally instead of hardcoding.
    pass
    
    # For safety, let me build the 2x2x2 decomposition by solving the linear system.
    # Actually for this task, the simpler approach is to use the schoolbook for each block
    # and use Strassen for the 2x2x2 case if we can get it right.
    
    # Let me just return None and use schoolbook for now.
    return None


def _build_schoolbook_block(n, m, p):
    """Build schoolbook decomposition for n x m x p block. Returns (U,V,W) as (nm x r, mp x r, pn x r)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication.
    U: (nm x r), V: (mp x r), W: (pn x r)
    Check: for all i,j,k: delta(i,j,k) = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where delta = 1 if this is a valid (i,j,k) triple, 0 otherwise.
    Actually the tensor is T[i,j,k] = 1 for all valid i,j,k (it's the matmul tensor).
    Wait no. The matmul tensor T satisfies: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] with T[i,j,k]=1.
    So T[i,j,k] = 1 for all i,j,k.
    The decomposition: T[i,j,k] = sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i]
    So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # U is (nm, r), so U[i*m+j, r]
    # We need sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for rr in range(r):
                    s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _combine_blocks(block_list):
    """Combine a list of (U_block, V_block, W_block, i_offset, j_offset, k_offset) tuples.
    Each block is for a sub-matrix multiplication at offset position.
    U_block: (nb*mb, rb), V_block: (mb*pb, rb), W_block: (pb*nb, rb)
    Returns combined (U, V, W).
    """
    n, m, p = DIMENSIONS
    total_rank = sum(rb for (_, _, _, rb) in [(b[0].shape[1] for b in block_list)] for _ in [0])
    # Simpler:
    total_rank = sum(b[0].shape[1] for b in block_list)
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for (Ub, Vb, Wb, i0, j0, k0) in block_list:
        nb, mb, pb = Ub.shape[0] // max(1,Ub.shape[0]//(max(1,1))) if False else (None,None,None)
        # We need to know the block dimensions. Let me store them.
        pass
    
    # Let me restructure.
    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # First, build the schoolbook decomposition (rank 60)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb))
    V = np.zeros((m * p, rank_sb))
    W = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try structured block decompositions
    # Strategy: decompose 2x5x6 into blocks:
    # - Split the 5 dimension into 1+4: 2x1x6 + 2x4x6
    # - Or 2+3: 2x2x6 + 2x3x6
    # - Or split the 6 dimension: 2x5x3 + 2x5x3
    # - Or various combinations
    
    # For each block, we can use schoolbook or try to optimize.
    
    # Let's try: 2x1x6 (schoolbook rank 12) + 2x4x6 (schoolbook rank 48) = total 60. Same as schoolbook.
    # To improve, we need to find a better decomposition for some block.
    
    # Key insight: For 2xn xn, the known minimum rank is around 2n^2 - n for some n.
    # For 2x5x5: schoolbook is 50. Known better?
    # For 2x3x3: schoolbook is 18. Known rank is 14 (or similar).
    # For 2x2x2: rank 7 (Strassen).
    
    # Let's try to build a decomposition using 2x2x2 Strassen blocks where possible.
    
    # Time budget check
    if time.time() - start_time > DEADLINE_SEC - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a decomposition: split 5 into 2+3
    # Block 1: 2x2x6 at (i=0..1, j=0..1, k=0..5)
    # Block 2: 2x3x6 at (i=0..1, j=2..4, k=0..5)
    
    # For 2x2x6: use Strassen for the 2x2 part, extended to 6 columns.
    # Strassen for 2x2x2 gives rank 7 per 2x2x2. For 2x2x6, we can do 3 blocks of 2x2x2 -> rank 21.
    # Or find a direct 2x2x6 decomposition with rank < 21.
    
    # Actually, the knowledge says 2x2x6 uses 3 Strassen instances (rank 21). And 2x1x6 is rank 12.
    # So 2x5x6 = 2x1x6 + 2x2x6 + 2x2x6 would be 12 + 21 + 21 = 54. That's an improvement over 60!
    
    # Let's build this.
    
    def build_2x1x6_schoolbook():
        """2x1x6 schoolbook: rank 12."""
        nb, mb, pb = 2, 1, 6
        r = nb * mb * pb
        Ub = np.zeros((nb * mb, r))
        Vb = np.zeros((mb * pb, r))
        Wb = np.zeros((pb * nb, r))
        c = 0
        for i in range(nb):
            for j in range(mb):
                for k in range(pb):
                    Ub[i * mb + j, c] = 1
                    Vb[j * pb + k, c] = 1
                    Wb[k * nb + i, c] = 1
                    c += 1
        return Ub, Vb, Wb, nb, mb, pb
    
    def build_2x2x6_strassen():
        """2x2x6 using 3 Strassen 2x2x2 blocks. Rank 21."""
        # Three 2x2x2 Strassen blocks for k=0..1, k=2..3, k=4..5
        blocks = []
        for k0 in range(0, 6, 2):
            Ub2, Vb2, Wb2, _, _, _ = _build_strassen_222()
            blocks.append((Ub2, Vb2, Wb2, 0, 0, k0))
        
        nb, mb, pb = 2, 2, 6
        total_r = 0
        for (ub, vb, wb, _, _, _, _) in blocks:
            total_r += ub.shape[1]
        
        Ub = np.zeros((nb * mb, total_r))
        Vb = np.zeros((mb * pb, total_r))
        Wb = np.zeros((pb * nb, total_r))
        
        col = 0
        for (ub, vb, wb, i0, j0, k0) in blocks:
            r_block = ub.shape[1]
            # Place ub into U
            for ii in range(2):
                for jj in range(2):
                    Ui_idx = (i0 + ii) * mb + (j0 + jj)  # wait, global indices
                    # Actually the block is at i_offset=0, j_offset=0, k_offset=k0
                    # Within the block, local indices are 0..1 for i,j,k
                    # Global i = 0 + local_i, global j = 0 + local_j, global k = k0 + local_k
                    pass
            
            # This is getting complicated. Let me build it differently.
            col_end = col + r_block
            # Ub is 4 x r_block (2*2 = 4 entries for i,j)
            # Place at global U positions: i in {0,1}, j in {0,1}
            for local_i in range(2):
                for local_j in range(2):
                    global_i = local_i  # i_offset = 0
                    global_j = local_j  # j_offset = 0
                    U_idx = global_i * m + global_j  # but m=2 for this block
                    # Wait, we're building a sub-block. The global m is 5, but this block has mb=2.
                    # We need to be careful.
                    pass
            
            col = col_end
        
        return None, None, None, 2, 2, 6
    
    # Let me take a cleaner approach: build the full decomposition directly.
    
    # Decomposition plan:
    # Split j (the 5-dim) into: j=0 (1 block), j=1,2 (2 blocks), j=3,4 (2 blocks)
    # So: block A: 2x1x6 (j=0), block B: 2x2x6 (j=1,2), block C: 2x2x6 (j=3,4)
    # Each 2x2x6 uses 3 Strassen 2x2x2 instances.
    
    def strassen_222_UVW():
        """Returns (U, V, W) for 2x2x2 matmul, rank 7, using known coefficients.
        U: 4x7 (indices i*2+j), V: 4x7 (indices j*2+k), W: 4x7 (indices k*2+i)
        """
        # Let me derive this by solving the linear constraints.
        # We need 7 terms such that sum_r U[r][i,j]*V[r][j,k]*W[r][k,i] = 1 for all i,j,k in {0,1}
        
        # This is equivalent to: the 8x8 system for 8 (i,j,k) tuples with 7 unknowns per term...
        # Actually each term r contributes a rank-1 tensor (outer product of three vectors of length 4).
        # We need sum of 7 such rank-1 tensors to equal the all-ones tensor.
        
        # The all-ones tensor T[i,j,k] = 1 for i,j,k in {0,1} has 8 components.
        # We need 7 rank-1 tensors to sum to this.
        
        # This is the Strassen decomposition. Let me hardcode a known one.
        # From literature: for 2x2x2, one valid rank-7 decomposition is:
        
        U_s = np.array([
            [1,  1,  1,  0,  1,  0,  1],
            [0,  0,  0,  1,  1,  1,  0],
            [0,  1,  1,  0,  0,  0,  1],
            [1,  0,  1,  1,  0,  1,  0],
        ], dtype=float)
        # U indices: row = i*2+j
        # Row 0: i=0,j=0 -> U[0] = [1,1,1,0,1,0,1]
        # Row 1: i=0,j=1 -> U[1] = [0,0,0,1,1,1,0]
        # Row 2: i=1,j=0 -> U[2] = [0,1,1,0,0,0,1]
        # Row 3: i=1,j=1 -> U[3] = [1,0,1,1,0,1,0]
        
        V_s = np.array([
            [1,  0,  0,  1,  1,  1,  1],
            [1,  1,  0,  0,  1,  0,  0],
            [0,  0,  1,  1,  0,  1,  0],
            [0,  1,  1,  0,  0,  0,  1],
        ], dtype=float)
        # V indices: row = j*2+k
        # Row 0: j=0,k=0
        # Row 1: j=0,k=1
        # Row 2: j=1,k=0
        # Row 3: j=1,k=1
        
        W_s = np.array([
            [1,  1,  1,  0,  1,  0,  1],
            [0,  0,  0,  1,  1,  1,  0],
            [1,  0,  1,  1,  0,  1,  0],
            [0,  1,  1,  0,  0,  0,  1],
        ], dtype=float)
        # W indices: row = k*2+i
        # Row 0: k=0,i=0
        # Row 1: k=0,i=1
        # Row 2: k=1,i=0
        # Row 3: k=1,i=1
        
        # Verify: for all i,j,k in {0,1}: sum_r U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r] == 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(7):
                        s += U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r]
                    if abs(s - 1.0) > 1e-9:
                        # Coefficients are wrong. Need to find correct ones.
                        pass
        
        # Let me verify numerically and if wrong, fall back to a different approach.
        ok = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = sum(U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r] for r in range(7))
                    if abs(s - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if not ok:
            # Try to find correct coefficients via linear programming / random search
            # For now, return schoolbook 2x2x2 (rank 8) as fallback
            U_s = np.eye(4).repeat(2, axis=1)[:4, :8] if False else None
            # Schoolbook for 2x2x2:
            U_sb = np.zeros((4, 8))
            V_sb = np.zeros((4, 8))
            W_sb = np.zeros((4, 8))
            for idx, (ii, jj, kk) in enumerate(product(range(2), range(2), range(2))):
                U_sb[ii*2+jj, idx] = 1
                V_sb[jj*2+kk, idx] = 1
                W_sb[kk*2+ii, idx] = 1
            return U_sb, V_sb, W_sb
        
        return U_s, V_s, W_s
    
    if time.time() - start_time > DEADLINE_SEC - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Build the combined decomposition: 2x1x6 + 2x2x6 + 2x2x6
    # Using Strassen for the 2x2x2 sub-blocks
    
    # Get Strassen coefficients
    S_U, S_V, S_W = strassen_222_UVW()
    strassen_rank = S_U.shape[1]  # 7 if Strassen works, 8 if schoolbook fallback
    
    # Build block A: 2x1x6 schoolbook, rank 12
    A_U, A_V, A_W, _, _, _ = build_2x1x6_schoolbook()
    
    # Build block B: 2x2x6 using 3 Strassen 2x2x2 blocks
    # k splits: (0,1), (2,3), (4,5)
    B_parts = []
    for k0 in [0, 2, 4]:
        B_parts.append((S_U.copy(), S_V.copy(), S_W.copy(), k0))
    B_rank = 3 * strassen_rank
    B_U = np.zeros((4, B_rank))
    B_V = np.zeros((12, B_rank))
    B_W = np.zeros((12, B_rank))
    for part_idx, (su, sv, sw, k0) in enumerate(B_parts):
        c0 = part_idx * strassen_rank
        # U: 4 entries (i*2+j), same for all k blocks
        B_U[:, c0:c0+strassen_rank] = su
        # V: 12 entries (j*2+k), j in 0..1, k in 0..5
        for lj in range(2):
            for lk in range(2):
                local_idx = lj * 2 + lk
                global_idx = lj * 2 + (k0 + lk)
                B_V[global_idx, c0:c0+strassen_rank] = sv[local_idx, :]
        # W: 12 entries (k*2+i), k in 0..5, i in 0..1
        for lk in range(2):
            for li in range(2):
                local_idx = lk * 2 + li
                global_idx = (k0 + lk) * 2 + li
                B_W[global_idx, c0:c0+strassen_rank] = sw[local_idx, :]
    
    # Build block C: 2x2x6 using 3 Strassen 2x2x2 blocks, j offset = 3
    C_rank = 3 * strassen_rank
    C_U = np.zeros((4, C_rank))
    C_V = np.zeros((12, C_rank))
    C_W = np.zeros((12, C_rank))
    for part_idx, k0 in enumerate([0, 2, 4]):
        c0 = part_idx * strassen_rank
        C_U[:, c0:c0+strassen_rank] = S_U
        for lj in range(2):
            for lk in range(2):
                local_idx = lj * 2 + lk
                global_idx = (3 + lj) * 2 + (k0 + lk)  # j offset = 3, so global j = 3+lj
                # But V has mb*pb = 2*6 = 12 entries indexed by j*6+k where j is local to block (0,1)
                # In the full V (m*p = 5*6 = 30), the index is global_j * p + global_k
                # For block C, global_j = 3 + local_j, global_k = k0 + local_k
                C_V[global_idx, c0:c0+strassen_rank] = S_V[local_idx, :]
        for lk in range(2):
            for li in range(2):
                local_idx = lk * 2 + li
                global_idx = (k0 + lk) * 2 + li
                C_W[global_idx, c0:c0+strassen_rank] = S_W[local_idx, :]
    
    # Now assemble the full decomposition
    # A: j=0, 2x1x6, rank 12
    # B: j=1,2, 2x2x6, rank 21 (or 24 if schoolbook)
    # C: j=3,4, 2x2x6, rank 21 (or 24)
    
    total_rank = 12 + B_rank + C_rank
    
    full_U = np.zeros((10, total_rank))  # n*m = 2*5 = 10
    full_V = np.zeros((30, total_rank))  # m*p = 5*6 = 30
    full_W = np.zeros((12, total_rank))  # p*n = 6*2 = 12
    
    # Block A: j=0
    # A_U is 2x12 (i*1+j, i in 0..1, j=0)
    # A_V is 1x12 (j*6+k, j=0, k in 0..5) -> but it's actually (mb*pb = 1*6 = 6) x 12
    # Wait, A_V shape is (1*6, 12) = (6, 12)
    # A_W is (6*2, 12) = (12, 12)
    
    col = 0
    # Place A
    # A_U: row i*1+0 = i (i=0,1). Full U row: i*5+0 = 5*i
    for li in range(2):
        full_U[li * 5 + 0, col:col+12] = A_U[li * 1 + 0, :]
    # A_V: row 0*6+k = k. Full V row: 0*6+k = k
    for lk in range(6):
        full_V[0 * 6 + lk, col:col+12] = A_V[0 * 6 + lk, :] if False else A_V[lk, :]
    # A_V is (6, 12), indexed by k (since j=0 only). A_V[k, r]
    # Actually A_V has shape (mb*pb, rank) = (1*6, 12) = (6, 12). Row index = j*pb+k = 0*6+k = k.
    # Full V row index = global_j * p + global_k = 0*6+k = k.
    # So A_V[k, r] -> full_V[k, r]
    for lk in range(6):
        full_V[lk, col:col+12] = A_V[lk, :]
    
    # A_W: shape (6*2, 12) = (12, 12). Row index = k*2+i.
    # Full W row index = k*n+i = k*2+i. Same.
    full_W[:, col:col+12] = A_W[:, :]
    
    col += 12
    
    # Place B: j=1,2
    # B_U: 4x21, row = i*2+lj where lj in 0..1 (local j). Full: i*5+(1+lj)
    for li in range(2):
        for lj in range(2):
            B_U_idx = li * 2 + lj
            full_U[li * 5 + (1 + lj), col:col+B_rank] = B_U[B_U_idx, :]
    
    # B_V: 12x21, row = lj*2+lkk where lkk in 0..5. Full: (1+lj)*6+lkk
    for lj in range(2):
        for lkk in range(6):
            B_V_idx = lj * 2 + lkk if False else None
            # B_V was built with row index = lj_local * 6 + lkk? No.
            # B_V has shape (12, 21) where 12 = 2*6. Row = local_j * 6 + k.
            # Wait, I built B_V as (mb*pb, rank) = (2*6, 21) = (12, 21).
            # Row index in B_V = local_j * pb + k = lj * 6 + k.
            for lkk in range(6):
                B_V_idx = lj * 6 + lkk
                full_V[(1 + lj) * 6 + lkk, col:col+B_rank] = B_V[B_V_idx, :]
    
    # B_W: 12x21, row = k*2+li. Full W row = k*2+li. Same.
    full_W[:, col:col+B_rank] = B_W[:, :]
    
    col += B_rank
    
    # Place C: j=3,4
    # C_U: 4x21, row = li*2+lj. Full: li*5+(3+lj)
    for li in range(2):
        for lj in range(2):
            C_U_idx = li * 2 + lj
            full_U[li * 5 + (3 + lj), col:col+C_rank] = C_U[C_U_idx, :]
    
    # C_V: 12x21, row = lj*6+k. Full: (3+lj)*6+k
    for lj in range(2):
        for lkk in range(6):
            C_V_idx = lj * 6 + lkk
            full_V[(3 + lj) * 6 + lkk, col:col+C_rank] = C_V[C_V_idx, :]
    
    # C_W: 12x21, row = k*2+li. Full W row = k*2+li.
    full_W[:, col:col+C_rank] = C_W[:, :]
    
    col += C_rank
    
    # Verify
    if _verify_decomposition(full_U, full_V, full_W):
        best_rank = total_rank
        best_U, best_V, best_W = full_U.copy(), full_V.copy(), full_W.copy()
        # Convert to half-integers if all entries are multiples of 0.5
        # They're all integers here, so fine.
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Decomposition is wrong. Fall back to schoolbook.
        pass
    
    # Fallback: return schoolbook
    if time.time() - start_time > DEADLINE_SEC - 2:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try a simple heuristic: randomly perturb and optimize? Too risky in time.
    # Just return the schoolbook decomposition.
    U_out = [[0] * 60 for _ in range(10)]
    V_out = [[0] * 60 for _ in range(30)]
    W_out = [[0] * 60 for _ in range(12)]
    col = 0
    for i in range(2):
        for j in range(5):
            for k in range(6):
                U_out[i*5+j][col] = 1
                V_out[j*6+k][col] = 1
                W_out[k*2+i][col] = 1
                col += 1
    return {"U": U_out, "V": V_out, "W": W_out}
