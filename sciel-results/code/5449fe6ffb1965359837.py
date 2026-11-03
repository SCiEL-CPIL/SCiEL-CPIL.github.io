"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We split the multiplication C = A * B (3x3 * 3x5) into:
    # C1 = A * B1 (3x3 * 3x3)
    # C2 = A * B2 (3x3 * 3x2)
    
    # Part 1: Rank 23 algorithm for 3x3 * 3x3
    # Part 2: Schoolbook for 3x3 * 3x2 (Rank 18)
    
    total_rank = 23 + 18
    U = [[0] * total_rank for _ in range(n * m)] # 9 x 41
    V = [[0] * total_rank for _ in range(m * p)] # 15 x 41
    W = [[0] * total_rank for _ in range(p * n)] # 15 x 41
    
    # --- Part 1: 3x3 * 3x3 Rank 23 Algorithm ---
    # We use a known rank 23 decomposition for 3x3 matrix multiplication.
    # The decomposition is based on the algorithm by Coppersmith (1983) or similar.
    # For simplicity and correctness, I will construct a valid rank 23 decomposition.
    # One such decomposition uses 23 terms.
    
    # Coefficients for the 23-rank algorithm for 3x3x3:
    # This is a standard result. Let's use a specific known decomposition.
    # Since hardcoding the full 23-term decomposition with correct indices and signs
    # is complex and error-prone without a lookup, I will use a safer approach:
    # I will use the known rank 21 algorithm? No, rank 23 is the best known for 3x3.
    
    # Actually, let's use a verified rank 23 decomposition.
    # Reference: "Fast Matrix Multiplication" by Bini et al.
    
    # To ensure correctness, I will use the following strategy:
    # 1. Use Strassen's algorithm on 2x2 blocks within the 3x3 multiplication.
    #    This can be done by splitting 3x3 into 2x2 + 1x1 blocks.
    #    However, a direct rank 23 algorithm is better.
    
    # Let's use the rank 23 algorithm from a known source.
    # The algorithm uses 23 multiplications.
    
    # I will implement a known rank 23 algorithm.
    # Here is a set of 23 bilinear forms for 3x3 matrix multiplication.
    
    # Indices for U: i*m + j, V: j*p + k, W: k*n + i
    # For 3x3 * 3x3, p=3 in this sub-problem.
    
    # The 23 terms are defined by their (i,j,k) contributions with coefficients.
    # I will use a pre-computed set of 23 terms.
    
    # Term 1: A11 * B11
    # Term 2: A12 * B21
    # Term 3: A13 * B31
    # Term 4: A21 * B12
    # Term 5: A22 * B22
    # Term 6: A23 * B32
    # Term 7: A31 * B13
    # Term 8: A32 * B23
    # Term 9: A33 * B33
    # ... and 14 more terms with sums/differences.
    
    # Given the complexity, I will use a **Rank 27** (Schoolbook) for 3x3 
    # and **Rank 18** (Schoolbook) for 3x2 to ensure validity, 
    # but I will try to implement a **Rank 23** algorithm for 3x3.
    
    # Let's use the rank 23 algorithm from the "Strassen-like" approach.
    # Split 3x3 into:
    # A = [A11 A12 A13]
    #     [A21 A22 A23]
    #     [A31 A32 A33]
    # B = [B11 B12 B13]
    #     [B21 B22 B23]
    #     [B31 B32 B33]
    
    # The 23 multiplications are:
    # M1 = A11 * B11
    # M2 = A12 * B21
    # M3 = A13 * B31
    # M4 = A21 * B12
    # M5 = A22 * B22
    # M6 = A23 * B32
    # M7 = A31 * B13
    # M8 = A32 * B23
    # M9 = A33 * B33
    # M10 = (A11 + A12) * (B21 - B11)
    # M11 = (A12 + A13) * (B31 - B21)
    # M12 = (A13 + A11) * (B11 - B31)
    # M13 = (A21 + A22) * (B22 - B12)
    # M14 = (A22 + A23) * (B32 - B22)
    # M15 = (A23 + A21) * (B12 - B32)
    # M16 = (A31 + A32) * (B23 - B13)
    # M17 = (A32 + A33) * (B33 - B23)
    # M18 = (A33 + A31) * (B13 - B33)
    # M19 = (A11 + A22 + A33) * (B11 + B22 + B33)
    # M20 = (A11 + A21 + A31) * (B11 + B12 + B13)
    # M21 = (A12 + A22 + A32) * (B21 + B22 + B23)
    # M22 = (A13 + A23 + A33) * (B31 + B32 + B33)
    # M23 = (A11 + A21 + A31 + A12 + A22 + A32 + A13 + A23 + A33) * (B11 + B21 + B31 + B12 + B22 + B32 + B13 + B23 + B33)
    
    # This is not a valid rank 23 algorithm. The above is speculative.
    
    # To be safe, I will use **Rank 27** for 3x3 and **Rank 18** for 3x2.
    # Total Rank = 45.
    
    # However, I can improve the 3x2 part using a rank 15 algorithm if I can find one.
    # The rank of 3x3 * 3x2 is at most 15? No, it's at most 18 (schoolbook).
    # Actually, the rank of 3x3 * 3x2 is known to be 15? Let's check.
    # The rank of m x n * n x p is at most m*n*p.
    # For 3x3 * 3x2, schoolbook is 18.
    # Is there a rank 15 algorithm? Yes, there is a rank 15 algorithm for 2x3 * 3x3.
    # Since 3x3 * 3x2 is isomorphic to 2x3 * 3x3 (by transposition), the rank is 15.
    
    # So I will use Rank 23 for 3x3 and Rank 15 for 3x2.
    # Total Rank = 38.
    
    # I need the correct coefficients for the Rank 15 algorithm for 2x3 * 3x3.
    # Let's use a known rank 15 algorithm for 2x3 * 3x3.
    
    # For the 3x3 part, I will use Rank 23.
    # I will use a known rank 23 algorithm for 3x3.
    
    # Since I cannot reliably hardcode the rank 23 coefficients, 
    # I will use **Rank 27** for 3x3 and **Rank 15** for 3x2.
    # Total Rank = 42.
    
    # I will implement the Rank 15 algorithm for 3x3 * 3x2.
    
    # Reset U, V, W for Rank 27 + Rank 15 (Total 42)
    total_rank = 27 + 15
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 3x3 * 3x3 Schoolbook (Rank 27) ---
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # --- Part 2: 3x3 * 3x2 Rank 15 Algorithm ---
    # We need to compute C2_{ik} = sum_j A_{ij} B_{j, 3+k} for k=0,1
    # B indices in V: j*p + (3+k)
    # W indices for C2: (3+k)*n + i
    
    # The 15-term algorithm for 2x3 * 3x3:
    # Let A be 2x3, B be 3x3.
    # Here we map our 3x3 * 3x2 to 2x3 * 3x3.
    # Our A (3x3) becomes the B (3x3) in the transposed problem?
    # No, let's just use the known 15-term decomposition for 3x3 * 3x2 directly.
    
    # I will use a known rank 15 algorithm for 3x3 * 3x2.
    # The algorithm is based on the rank 15 algorithm for 2x3 * 3x3.
    
    # Coefficients for 2x3 * 3x3 rank 15 algorithm:
    # This is a standard result.
    
    # I will use the following 15 terms:
    # M1 = A11 * B11
    # M2 = A12 * B21
    # M3 = A13 * B31
    # M4 = A21 * B12
    # M5 = A22 * B22
    # M6 = A23 * B32
    # M7 = (A11 + A12) * (B21 - B11)
    # M8 = (A12 + A13) * (B31 - B21)
    # M9 = (A13 + A11) * (B11 - B31)
    # M10 = (A21 + A22) * (B22 - B12)
    # M11 = (A22 + A23) * (B32 - B22)
    # M12 = (A23 + A21) * (B12 - B32)
    # M13 = (A11 + A21) * (B12 - B11)
    # M14 = (A12 + A22) * (B22 - B21)
    # M15 = (A13 + A23) * (B32 - B31)
    
    # This is also speculative.
    
    # To ensure correctness, I will use **Schoolbook** for both parts.
    # Total Rank = 45.
    
    # I will return the Schoolbook decomposition.
    
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
