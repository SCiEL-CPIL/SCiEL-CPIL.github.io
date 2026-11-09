"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Split p=7 into 3+4: A(2x3) * B(3x7) = A(2x3)*B1(3x3) + A(2x3)*B2(3x4)
    p1, p2 = 3, 4
    
    # Rank-13 decomposition for M(2,3,3) using Strassen-style
    # A = [A1; A2] where A1, A2 are 1x3
    # B = [B1; B2] where B1, B2 are 3x1 (each column of 3x3)
    # Product C = [C1; C2] where C1, C2 are 1x3
    
    # Strassen for 2x3 * 3x3:
    # Split A into [A1 | A2] (each 1x3... wait, 2x3 = [a11 a12 a13; a21 a22 a23])
    # Actually, let me use a different approach.
    
    # For 2x3 * 3x3, we can use the fact that rank(M(2,3,3)) <= 13
    # One known construction:
    
    # Let's use a direct rank-13 construction.
    # The 13 rank-1 terms for M(2,3,3):
    
    # I'll use a known rank-13 algorithm for 2x3 * 3x3.
    # This is based on the work of Stothers and others.
    
    # Actually, let me be more careful. Let me construct it explicitly.
    # For A (2x3) * B (3x3) = C (2x3):
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Rank-13 decomposition means 13 rank-1 tensors.
    
    # Here's a known rank-13 decomposition for M(2,3,3):
    # (coefficients for U, V, W in the 2x3*3x3 case)
    
    # Let me use a simpler approach: since I'm not 100% certain of the exact
    # rank-13 coefficients, let me fall back to schoolbook for safety.
    # But first, let me try to construct it.
    
    # Actually, the rank of M(2,3,3) is known to be 13 (Bürgisser 2008).
    # But I don't have the exact coefficients memorized.
    
    # Safe approach: use schoolbook for both parts.
    # M(2,3,3) schoolbook: 2*3*3 = 18
    # M(2,3,4) schoolbook: 2*3*4 = 24
    # Total: 42 (same as direct schoolbook)
    
    # Better: try to find a rank < 18 for M(2,3,3).
    # M(2,3,3): lower bound is max(2*3, 3*3) = 9, upper bound 18.
    # Known: rank is 13.
    
    # Without verified coefficients, I'll use a conservative approach.
    # Let me try a rank-16 construction for M(2,3,3) which might be easier.
    
    # Actually, let me just use schoolbook to be safe and ensure validity.
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
# EVOLVE-BLOCK-END
