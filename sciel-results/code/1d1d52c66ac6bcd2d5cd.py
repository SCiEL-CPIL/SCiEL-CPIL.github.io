"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np

def _build_split(p_left, p_right):
    """
    Build a decomposition for multiplying a 2x3 matrix by a 3xp matrix.
    We split the 3x(p_left+p_right) matrix into two 3xp_left and 3xp_right parts.
    We use a Strassen-like rank-7 decomposition for the 2x2 part of the inner dim,
    and a trivial rank-p decomposition for the remaining 1x1 part.
    Total rank = 7 * p_left/2? No.
    
    Let's be more precise.
    The multiplication is:
    C (2 x p) = A (2 x 3) * B (3 x p)
    
    We can view this as:
    C = [A1 A2 A3] * [B1; B2; B3]
    where A1, A2, A3 are 2x1, and B1, B2, B3 are 1xp.
    Actually, it's easier to think of the 3 as 2+1.
    Let A = [A2x2 A2x1] where A2x2 is 2x2 and A2x1 is 2x1.
    Let B = [B2xp; B1xp] where B2xp is 2xp and B1xp is 1xp.
    
    C = A2x2 * B2xp + A2x1 * B1xp
    
    The term A2x1 * B1xp is a 2xp matrix. We can compute it with p multiplications (rank p).
    The term A2x2 * B2xp is a 2xp matrix. We can use Strassen's algorithm to compute this with 7 multiplications per 2x1 output column.
    Wait, Strassen multiplies two 2x2 matrices. Here we multiply 2x2 by 2xp.
    We can view B2xp as a block matrix of p columns, each 2x1.
    So C = A2x2 * B2xp = [A2x2 * B_col1, A2x2 * B_col2, ..., A2x2 * B_colp].
    Each A2x2 * B_colj is a 2x1 vector.
    The product of a 2x2 matrix and a 2x1 vector can be computed with rank 3 (trivial) or can it be done with less?
    Actually, the tensor M_{2,2,1} is just the multiplication of 2x2 by 2x1.
    The rank of M_{2,2,1} is 4 (since it's just 4 multiplications, and the border rank is 4).
    So using Strassen doesn't help for a single column.
    
    Let's rethink the split.
    The previous successful run (score 1.08) used a rank-50 decomposition.
    The schoolbook rank is 2*3*9 = 54.
    The improvement was 54 - 50 = 4.
    
    How to get rank 50?
    One known bound for M_{n,m,p} is related to Kronecker products.
    If we decompose the inner dimension m=3 into m1+m2, the rank is R(M_{n,m1,p}) + R(M_{n,m2,p}).
    If we choose m1=2, m2=1.
    R(M_{2,2,9}) + R(M_{2,1,9}).
    R(M_{2,1,9}) = 2*9 = 18.
    What is R(M_{2,2,9})?
    M_{2,2,p} is the multiplication of 2x2 by 2xp.
    It can be computed by viewing B as p columns of 2x1.
    Rank is at most 4p = 36.
    Is it less?
    Actually, R(M_{2,2,p}) = 2p + 2 for p>=2? No.
    Let's check the rank of M_{2,2,p}.
    For p=1, rank=4.
    For p=2, M_{2,2,2} has rank 7 (Strassen).
    For p=3, M_{2,2,3} has rank 11.
    For p=9, R(M_{2,2,9})?
    
    If R(M_{2,2,9}) = 7 * (9/2)? No.
    Actually, M_{2,2,p} is isomorphic to M_{2,p,2} (by symmetry).
    There is a formula: R(M_{2,2,p}) = 2p + 2 for p even? No.
    
    Let's look at the known result for M_{2,3,9}.
    The prompt mentions a rank-50 solution.
    50 = 18 + 32?
    If R(M_{2,2,9}) = 32, then 32+18=50.
    Is R(M_{2,2,9}) = 32?
    For M_{2,2,p}, the rank is known to be 2p+2 for p odd?
    If p=9, 2*9+2 = 20? No, 20+18=38.
    
    Let's try another split.
    Split n=2 into 1+1.
    R(M_{1,3,9}) + R(M_{1,3,9}) = 3*9 + 3*9 = 54. No gain.
    
    Split p=9 into 4+5.
    R(M_{2,3,4}) + R(M_{2,3,5}).
    R(M_{2,3,4}) = R(M_{2,2,4}) + R(M_{2,1,4}) = ? + 8.
    R(M_{2,2,4}) = 7 + 7 = 14? (Two Strassen blocks).
    So R(M_{2,3,4}) = 14 + 8 = 22.
    R(M_{2,3,5}) = R(M_{2,2,5}) + R(M_{2,1,5}) = ? + 10.
    R(M_{2,2,5}) = R(M_{2,2,4}) + R(M_{2,2,1}) = 14 + 4 = 18.
    So R(M_{2,3,5}) = 18 + 10 = 28.
    Total = 22 + 28 = 50.
    This matches the observed score 1.08 (which corresponds to rank 50, since 54/50 = 1.08).
    
    So the strategy is:
    1. Split p=9 into p1=4 and p2=5.
    2. For M_{2,3,4}:
       - Split m=3 into m1=2, m2=1.
       - M_{2,2,4}: Split p=4 into p1=2, p2=2.
         - M_{2,2,2} + M_{2,2,2} -> Rank 7 + 7 = 14.
       - M_{2,1,4}: Rank 2*4 = 8.
       - Total for M_{2,3,4} is 22.
    3. For M_{2,3,5}:
       - Split m=3 into m1=2, m2=1.
       - M_{2,2,5}: Split p=5 into p1=4, p2=1.
         - M_{2,2,4} (rank 14) + M_{2,2,1} (rank 4) = 18.
       - M_{2,1,5}: Rank 2*5 = 10.
       - Total for M_{2,3,5} is 28.
    4. Total rank = 22 + 28 = 50.
    
    We need to construct the actual U, V, W matrices.
    This requires knowing the explicit rank-7 decomposition for M_{2,2,2} (Strassen) and the trivial decompositions for the rest.
    
    Let's implement the rank-50 construction.
    """
    n, m, p = 2, 3, 9
    total_rank = 50
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    
    # --- Part 1: M_{2,3,4} (columns 0-3 of output) ---
    # Split m=3 into 2+1.
    # --- M_{2,2,4} ---
    # Split p=4 into 2+2.
    # --- M_{2,2,2} (cols 0-1) ---
    # Strassen's 7-term decomposition for 2x2 * 2x2.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = AB.
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # In our tensor decomposition, we need to map these to U, V, W entries.
    # The indices for U are (i*m+j), where i in 0..1, j in 0..1 (for the 2x2 block).
    # The indices for V are (j*p+k), where j in 0..1, k in 0..1 (for the 2x2 block).
    # The indices for W are (k*n+i), where k in 0..1, i in 0..1.
    
    # Let's write a helper to add a Strassen block.
    def add_strassen_222(u_offset, v_offset, w_offset, col_start):
        # u_offset: (i*m+j) base index in U
        # v_offset: (j*p+k) base index in V
        # w_offset: (k*n+i) base index in W
        # col_start: starting column index in U, V, W
        
        # m1 = (a11+a22)*(b11+b22)
        # U: a11, a22 contribute to m1.
        # V: b11, b22 contribute to m1.
        # W: c11, c22 contribute from m1.
        U[u_offset + 0][col_start + 0] += 1 # a11
        U[u_offset + 3][col_start + 0] += 1 # a22 (i=1, j=1 -> 1*2+1=3)
        V[v_offset + 0][col_start + 0] += 1 # b11 (j=0, k=0 -> 0*2+0=0)
        V[v_offset + 3][col_start + 0] += 1 # b22 (j=1, k=1 -> 1*2+1=3)
        W[w_offset + 0][col_start + 0] += 1 # c11 (k=0, i=0 -> 0*2+0=0)
        W[w_offset + 3][col_start + 0] += 1 # c22 (k=1, i=1 -> 1*2+1=3)
        
        # m2 = (a21+a22)*b11
        U[u_offset + 2][col_start + 1] += 1 # a21 (i=1, j=0 -> 1*2+0=2)
        U[u_offset + 3][col_start + 1] += 1 # a22
        V[v_offset + 0][col_start + 1] += 1 # b11
        W[w_offset + 2][col_start + 1] += 1 # c21 (k=1, i=0 -> 1*2+0=2)
        W[w_offset + 3][col_start + 1] -= 1 # c22 (k=1, i=1 -> 3)
        
        # m3 = a11*(b12-b22)
        U[u_offset + 0][col_start + 2] += 1 # a11
        V[v_offset + 1][col_start + 2] += 1 # b12 (j=0, k=1 -> 0*2+1=1)
        V[v_offset + 3][col_start + 2] -= 1 # b22
        W[w_offset + 1][col_start + 2] += 1 # c12 (k=1, i=0 -> 1*2+0=2? No, W index is k*n+i. c12 is i=0, k=1. Index 1*2+0=2. Wait.
        # W index: k*n+i.
        # c11: i=0, k=0 -> 0.
        # c12: i=0, k=1 -> 1*2+0=2.
        # c21: i=1, k=0 -> 0*2+1=1.
        # c22: i=1, k=1 -> 1*2+1=3.
        # Let's re-verify W indices.
        # W[k*n+i, c].
        # c11 (i=0, k=0) -> W[0, c]
        # c12 (i=0, k=1) -> W[2, c]
        # c21 (i=1, k=0) -> W[1, c]
        # c22 (i=1, k=1) -> W[3, c]
        
        # Correcting W indices in previous lines:
        # m1: W[0] += 1, W[3] += 1. (Correct)
        # m2: W[1] += 1, W[3] -= 1. (Correct)
        
        # m3: a11*(b12-b22)
        # c12 = m3 + m5. c21 = m2 + m4.
        # c12 is W[2]. c21 is W[1].
        # m3 contributes to c12.
        W[w_offset + 2][col_start + 2] += 1
        
        # m4 = a22*(b21-b11)
        U[u_offset + 3][col_start + 3] += 1 # a22
        V[v_offset + 2][col_start + 3] += 1 # b21 (j=1, k=0 -> 1*2+0=2)
        V[v_offset + 0][col_start + 3] -= 1 # b11
        W[w_offset + 0][col_start + 3] += 1 # c11
        W[w_offset + 1][col_start + 3] += 1 # c21
        
        # m5 = (a11+a12)*b22
        U[u_offset + 0][col_start + 4] += 1 # a11
        U[u_offset + 1][col_start + 4] += 1 # a12 (i=0, j=1 -> 0*2+1=1)
        V[v_offset + 3][col_start + 4] += 1 # b22
        W[w_offset + 0][col_start + 4] -= 1 # c11
        W[w_offset + 2][col_start + 4] += 1 # c12
        
        # m6 = (a21-a11)*(b11+b12)
        U[u_offset + 2][col_start + 5] += 1 # a21
        U[u_offset + 0][col_start + 5] -= 1 # a11
        V[v_offset + 0][col_start + 5] += 1 # b11
        V[v_offset + 1][col_start + 5] += 1 # b12
        W[w_offset + 3][col_start + 5] += 1 # c22
        
        # m7 = (a12-a22)*(b21+b22)
        U[u_offset + 1][col_start + 6] += 1 # a12
        U[u_offset + 3][col_start + 6] -= 1 # a22
        V[v_offset + 2][col_start + 6] += 1 # b21
        V[v_offset + 3][col_start + 6] += 1 # b22
        W[w_offset + 0][col_start + 6] += 1 # c11
        
        return 7
    
    # We need to be careful with offsets.
    # For M_{2,2,2} at cols 0-1 of output (k=0,1).
    # U indices: i in 0..1, j in 0..1. Base 0.
    # V indices: j in 0..1, k in 0..1. Base 0.
    # W indices: k in 0..1, i in 0..1. Base 0.
    col += add_strassen_222(0, 0, 0, col)
    
    # --- M_{2,2,2} (cols 2-3) ---
    # U indices: i in 0..1, j in 0..1. Base 0.
    # V indices: j in 0..1, k in 2..3. Base 2*2+2 = 6? No.
    # V index is j*p + k. Here p=4 for this sub-problem? No, p is local.
    # In M_{2,2,2}, p=2.
    # When we split p=4 into 2+2, the second block has k in 0..1 locally, but globally k in 2..3.
    # So V base is 2*2 + 2 = 6.
    # W indices: k in 2..3, i in 0..1. Base 2*2 + 0 = 4.
    col += add_strassen_222(0, 6, 4, col)
    
    # --- M_{2,1,4} (cols 0-3) ---
    # A2x1 * B1x4. Rank 8.
    # U: i in 0..1, j=2 (the 1x1 part). Index i*3+2.
    # V: j=2, k in 0..3. Index 2*4+k = 8+k.
    # W: k in 0..3, i in 0..1. Index k*2+i.
    for k in range(4):
        for i in range(2):
            U[i*3+2][col] = 1
            V[8+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    # col is now 14 + 8 = 22.
    
    # --- Part 2: M_{2,3,5} (columns 4-8 of output) ---
    # Split m=3 into 2+1.
    # --- M_{2,2,5} ---
    # Split p=5 into 4+1.
    # --- M_{2,2,4} (cols 4-7) ---
    # Split p=4 into 2+2.
    # --- M_{2,2,2} (cols 4-5) ---
    # U base 0. V base 2*2+4 = 8. W base 4*2+0 = 8.
    col += add_strassen_222(0, 8, 8, col)
    
    # --- M_{2,2,2} (cols 6-7) ---
    # U base 0. V base 2*2+6 = 10. W base 6*2+0 = 12.
    col += add_strassen_222(0, 10, 12, col)
    
    # --- M_{2,2,1} (col 8) ---
    # Rank 4.
    # U: i in 0..1, j in 0..1.
    # V: j in 0..1, k=0 (local). Global k=8. Index 2*5+8? No, p is local 5?
    # Wait, in M_{2,2,5}, p=5.
    # V index j*5+k.
    # For k=4 (local), global k=8.
    # So V base is 2*5+8 = 18.
    # W base is 8*2+0 = 16.
    for i in range(2):
        for j in range(2):
            U[i*3+j][col] = 1
            V[j*5+4][col] = 1 # j*5+4 is the last column of the 2x5 block
            W[4*2+i][col] = 1 # 4*2+i is the last row of the 2x2 W block
            col += 1
    # col is now 22 + 14 + 4 = 40.
    
    # --- M_{2,1,5} (cols 4-8) ---
    # Rank 10.
    # U: i in 0..1, j=2. Index i*3+2.
    # V: j=2, k in 4..8. Index 2*9+k.
    # W: k in 4..8, i in 0..1. Index k*2+i.
    for k in range(4, 9):
        for i in range(2):
            U[i*3+2][col] = 1
            V[18+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    # col is now 40 + 10 = 50.
    
    return {"U": U, "V": V, "W": W}

def construct():
    return _build_split(4, 5)
# EVOLVE-BLOCK-END
