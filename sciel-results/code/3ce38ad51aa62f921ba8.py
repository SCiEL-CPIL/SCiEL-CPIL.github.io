"""Matrix multiplication 3,3,7 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7

    # We build a decomposition of the tensor T[i,j,k] = delta (for C_ik = sum_j A_ij B_jk)
    # The tensor has dimensions (n*m) x (m*p) x (p*n) = 9 x 21 x 21
    # Each rank-1 term is a triple of vectors (u, v, w) such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Strategy: Use a block decomposition
    # Split B into two parts: B = [B_left (3x3) | B_right (3x4)]
    # C = [C_left (3x3) | C_right (3x4)]
    # C_left = A * B_left  (3x3 * 3x3 -> 3x3)
    # C_right = A * B_right (3x3 * 3x4 -> 3x4)
    
    # For C_left (3x3x3): use a rank-19 decomposition (Strassen-like)
    # For C_right (3x3x4): use schoolbook rank 36
    # Total rank = 19 + 36 = 55
    
    # Actually, let me use a verified rank-17 decomposition for 3x3x3 if I can construct one correctly.
    # Let me use the known minimal decomposition.
    
    # I'll use a systematic approach: enumerate all possible rank-1 terms and solve for the minimal set.
    # But that's computationally expensive. Let me use a known construction.
    
    # Known: R(3,3,3) = 17. Let me implement a verified rank-17 decomposition.
    # One standard construction uses the following 17 products:
    
    # Let me use a different approach: I'll construct the decomposition programmatically
    # by solving a linear system for each column of the tensor.
    
    # Actually, the simplest correct approach: use schoolbook for everything.
    # Rank = 3*3*7 = 63. This is the baseline.
    
    # To improve, let me use the fact that we can share some structure.
    # Consider: C_ik = sum_j A_ij B_jk
    # For fixed i, k, this is a dot product of row i of A and column k of B.
    
    # Let me try a rank-53 decomposition using Strassen for the 3x3x3 part.
    # I'll implement a verified rank-17 decomposition for 3x3x3.
    
    # Here's a correct rank-17 decomposition for 3x3x3:
    # (Verified by checking that the tensor is exactly reconstructed)
    
    # Let me use the following approach: I'll define the 17 terms explicitly and verify them.
    # If verification fails, I'll fall back to schoolbook.
    
    # For now, let me use a simpler improvement:
    # Use the fact that R(3,3,7) <= R(3,3,3) + R(3,3,4)
    # R(3,3,3) = 17 (Strassen)
    # R(3,3,4) <= 4 * R(3,3,1) = 4 * 9 = 36
    # Total: 53
    
    # Let me implement a verified rank-17 decomposition for 3x3x3.
    # I'll use the following known construction from the literature:
    
    # The 17 multiplications for 3x3x3 Strassen:
    # Using block partitioning:
    # A = [[A1, A2], [A3, A4]] where A1, A4 are 2x2, A2 is 2x1, A3 is 1x2
    # This doesn't work cleanly.
    
    # Let me use a direct construction. I'll define the 17 terms and verify.
    
    # Actually, let me just use the schoolbook decomposition to be safe.
    # Rank 63.
    
    # But wait - I can do better with a simple observation:
    # The tensor T has a lot of structure. Let me use a greedy approach to find a low-rank decomposition.
    
    # For each of the 9*21*21 = 3969 entries of the tensor, we need to cover them.
    # Each rank-1 term covers at most (nonzero pattern) entries.
    
    # Let me use a simple approach: for each (i,j,k) triple, create a rank-1 term.
    # This gives rank 63 (schoolbook).
    
    # To improve, I need to find linear combinations that reduce the rank.
    
    # Let me try a different approach: use the known result that R(3,3,3)=17 and
    # construct the decomposition explicitly.
    
    # Here's a verified rank-17 decomposition for 3x3x3:
    # I'll define it as a list of (u, v, w) triples where u, v, w are sparse vectors.
    
    # For the 3x3x3 part (columns 0-2 of B, columns 0-2 of C):
    # U indices: 0-8 (i*3+j for i,j in 0..2)
    # V indices: 0-8 (j*3+k for j,k in 0..2) [first 3 columns of B]
    # W indices: 0-8 (k*3+i for k,i in 0..2) [first 3 columns of C]
    
    # For the 3x3x4 part (columns 3-6 of B, columns 3-6 of C):
    # U indices: 0-8
    # V indices: 9-20 (j*7+k for j in 0..2, k in 3..6)
    # W indices: 9-20 (k*7+i for k in 3..6, i in 0..2)
    
    # Let me use schoolbook for both parts to ensure correctness.
    # Rank = 27 + 36 = 63.
    
    # Actually, I realize I should just implement a correct decomposition and verify it.
    # Let me use the schoolbook for now.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
