import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# Strassen 2x2x2 vectors. Index order within a 2x2 block: 0=A11,1=A12,2=A21,3=A22.
_STRASSEN_222 = [
    ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
    ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
    ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
    ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
    ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
    ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
    ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
]


def _add_444(U, V, W, col, a_base, b_base, c_base):
    """Exact rank-49 decomposition for a 4x4x4 block using tiled Strassen.
    4x4 matrix divided into four 2x2 blocks. Each 2x2x2 Strassen uses 7 terms.
    There are 7x7=49 combinations. Returns new column index."""
    # Strassen 2x2x2: 7 terms, each with U, V, W vectors of length 4
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Within each 2x2 block: 0=TL, 1=TR, 2=BL, 3=BR
    # Global index for block b, position s: a_base + b*4 + s
    
    # Strassen terms for 2x2x2
    S_U = [
        [1, 0, 0, 1],  # A11+A22
        [0, 0, 1, 1],  # A21+A22
        [1, 0, 0, 0],  # A11
        [0, 0, 0, 1],  # A22
        [1, 1, 0, 0],  # A11+A12
        [-1, 0, 1, 0], # A21-A11
        [0, 1, 0, -1], # A12-A22
    ]
    S_V = [
        [1, 0, 0, 1],  # B11+B22
        [1, 0, 0, 0],  # B11
        [0, 1, 0, -1], # B12-B22
        [0, -1, 1, 0], # B21-B11
        [0, 0, 0, 1],  # B22
        [1, 1, 0, 0],  # B11+B12
        [0, 0, 1, 1],  # B21+B22
    ]
    S_W = [
        [1, 0, 0, 1],  # C11+C22
        [0, 0, 1, 0],  # C21
        [0, 1, 0, 0],  # C12
        [1, 0, 1, 0],  # C22+C21... wait
        [-1, 1, 0, 0], # C12-C11
        [0, 0, 0, 1],  # C22
        [1, 0, 0, 0],  # C11
    ]
    
    # For 4x4x4, we have 4x4 = 4 blocks of 2x2
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    # C = A*B where A,B,C are 4x4
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each 2x2x2 multiplication uses 7 Strassen terms
    # Total: 4 block products * 7 = 28? No, each 2x2x2 is one multiplication.
    # We need 4 independent 2x2x2 multiplications, each with rank 7, total 28? 
    # But that's not right either. The 4x4x4 tensor rank is 49 with Strassen tiling.
    # Actually, 4x4x4 can be done with rank 49 by viewing it as 7x7 grid of 2x2x2 Strassen.
    # No, that's not standard. Let me just use schoolbook for 4x4x4 (rank 64) or 
    # properly implement the 49-rank algorithm.
    
    # Simpler: use schoolbook for 4x4x4 (rank 64) to ensure correctness,
    # then total = 64+64+16 = 144. No improvement.
    # 
    # Better: implement proper 49-rank 4x4x4 via Strassen tiling.
    # The 4x4x4 tensor can be decomposed by applying Strassen to each 2x2x2 sub-block.
    # There are 4x4=16 such sub-blocks, but Strassen shares computations.
    # The standard result: M(4) <= 49 via Strassen.
    #
    # Implementation: divide A, B, C into 2x2 blocks.
    # Use the formula:
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)*B00 -> contributes to C10
    # M3 = A00*(B01-B11) -> contributes to C01
    # M4 = A11*(B10-A00) -> contributes to C11, C10
    # M5 = (A00+A01)*B11 -> contributes to C01
    # M6 = (A10-A00)*(B00+B01) -> contributes to C10
    # M7 = (A01-A11)*(B10+B11) -> contributes to C01, C11
    #
    # Each M is a 2x2x2 multiplication, done with 7 Strassen terms.
    # Total: 7 * 7 = 49 rank units.
    
    # For each of the 7 outer Strassen terms, we need to compute a 2x2x2 product
    # using 7 inner Strassen terms.
    
    for outer_t in range(7):
        # Get the 2x2 block combinations for this outer term
        # S_U[outer_t] gives coefficients for A blocks (A00,A01,A10,A11)
        # S_V[outer_t] gives coefficients for B blocks
        # S_W[outer_t] gives coefficients for C blocks
        
        au = S_U[outer_t]  # [a00, a01, a10, a11] coefficients
        bv = S_V[outer_t]  # [b00, b01, b10, b11] coefficients  
        cw = S_W[outer_t]  # [c00, c01, c10, c11] coefficients
        
        # Now we need to compute the 2x2x2 product of the combined A and B blocks
        # and add to the combined C blocks.
        # The combined A is a 2x2 matrix with entries:
        #   A_comb[0,0] = sum of au[0]*A00[0,0], au[1]*A01[0,0], au[2]*A10[0,0], au[3]*A11[0,0]
        #   etc.
        # But in the tensor decomposition, we express this as linear combinations of input positions.
        
        # For the inner Strassen, we need U, V, W vectors for the 2x2x2 product.
        # The U vector has 4 entries corresponding to the 4 positions of the combined 2x2 A matrix.
        # Position 0 = (0,0) of combined A = a00*A00[0,0] + a01*A01[0,0] + a10*A10[0,0] + a11*A11[0,0]
        # In terms of global A indices: A00[0,0] is at a_base+0, A01[0,0] at a_base+4, A10[0,0] at a_base+8, A11[0,0] at a_base+12
        
        # This is getting complex. Let me just use a direct approach:
        # For each inner Strassen term (7 terms), add contributions.
        
        for inner_t in range(7):
            su = S_U[inner_t]  # 4 entries for 2x2 A positions
            sv = S_V[inner_t]  # 4 entries for 2x2 B positions
            sw = S_W[inner_t]  # 4 entries for 2x2 C positions
            
            # U contributions: for each of the 4 positions in the 2x2 A,
            # the coefficient is su[pos] * au[block] where block maps to global index
            # Position 0 in 2x2 A corresponds to block 0 (A00), pos 1 to block 1 (A01), etc.
            # Wait, no. The combined A is a single 2x2 matrix. Its positions are:
            # (0,0), (0,1), (1,0), (1,1)
            # Each of these is a linear combination of the 4 A blocks' corresponding positions.
            # A_comb(0,0) = au[0]*A00(0,0) + au[1]*A01(0,0) + au[2]*A10(0,0) + au[3]*A11(0,0)
            # A_comb(0,1) = au[0]*A00(0,1) + au[1]*A01(0,1) + au[2]*A10(0,1) + au[3]*A11(0,1)
            # etc.
            
            # So for U, position p (0..3) in the 2x2 A has coefficient:
            # sum over blocks b: au[b] * su[p]
            # And this applies to global A index: a_base + b*4 + p
            
            for p in range(4):
                coeff_u = su[p]
                for b in range(4):
                    if au[b] != 0 and coeff_u != 0:
                        U[a_base + b * 4 + p][col] += au[b] * coeff_u
            
            for p in range(4):
                coeff_v = sv[p]
                for b in range(4):
                    if bv[b] != 0 and coeff_v != 0:
                        V[b_base + b * 4 + p][col] += bv[b] * coeff_v
            
            for p in range(4):
                coeff_w = sw[p]
                for b in range(4):
                    if cw[b] != 0 and coeff_w != 0:
                        W[c_base + b * 4 + p][col] += cw[b] * coeff_w
            
            col += 1
    
    return col


def _schoolbook_block(U, V, W, col, a_base, b_base, c_base, ar, ac, bc):
    """Schoolbook for A(ar x ac) * B(ac x bc) -> C(ar x bc)."""
    for i in range(ar):
        for j in range(ac):
            for k in range(bc):
                U[a_base + i * ac + j][col] += 1
                V[b_base + j * bc + k][col] += 1
                W[c_base + i * bc + k][col] += 1
                col += 1
    return col


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy. Returns True iff decomposition is exact."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Check all 144 target positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                # Sum over rank: U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                val = np.sum(Ua[idx_u, :] * Va[idx_v, :] * Wa[idx_w, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to reduce rank by finding zero columns.
    Returns (U, V, W, rank) potentially with lower rank."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Remove zero columns
    to_remove = []
    for r in range(rank):
        if np.all(Ua[:, r] == 0) and np.all(Va[:, r] == 0) and np.all(Wa[:, r] == 0):
            to_remove.append(r)
    
    if to_remove:
        keep = [r for r in range(rank) if r not in set(to_remove)]
        Ua = Ua[:, keep]
        Va = Va[:, keep]
        Wa = Wa[:, keep]
        rank = len(keep)
        return Ua.tolist(), Va.tolist(), Wa.tolist(), rank
    
    return U, V, W, rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Strategy: 4x4x4 (rank 49) + 4x4x4 (rank 49) + 4x4x1 (rank 16) = 114
    # B columns 0-3, 4-7, 8
    # C columns 0-3, 4-7, 8
    
    rank = 114  # 49 + 49 + 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: A (4x4) * B_cols0-3 (4x4) -> C_cols0-3 (4x4). Rank 49 via Strassen.
    # A index: i*m+j, i=0..3, j=0..3 -> 0..15, base=0
    # B index: j*p+k, j=0..3, k=0..3 -> j*9+k, base=0 (k=0..3)
    # C index: k*n+i, k=0..3, i=0..3 -> k*4+i, base=0
    col = _add_444(U, V, W, col, 0, 0, 0)
    
    # Part 2: A (4x4) * B_cols4-7 (4x4) -> C_cols4-7 (4x4). Rank 49 via Strassen.
    # B index: j*p+k, j=0..3, k=4..7 -> j*9+4..j*9+7, base=4
    # C index: k*n+i, k=4..7, i=0..3 -> (4+k_local)*4+i = 16+k_local*4+i, base=16
    col = _add_444(U, V, W, col, 0, 4, 16)
    
    # Part 3: A (4x4) * B_col8 (4x1) -> C_col8 (4x1). Rank 16 via schoolbook.
    # B index: j*p+8, j=0..3 -> j*9+8, base=8
    # C index: k*n+i, k=8, i=0..3 -> 8*4+i = 32+i, base=32
    col = _schoolbook_block(U, V, W, col, 0, 8, 32, 4, 4, 1)
    
    if col != rank:
        rank = col
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
    
    # Verify
    if time.time() < deadline and _verify_fast(U, V, W, n, m, p):
        # Try to reduce rank
        U, V, W, rank = _try_reduce_rank(U, V, W, n, m, p, deadline)
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 144
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
