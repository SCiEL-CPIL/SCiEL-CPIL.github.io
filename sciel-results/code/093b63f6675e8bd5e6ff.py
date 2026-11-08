"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 3x3 * 3x8, the schoolbook rank is 3*3*8 = 72
    # But we can potentially do better.
    # The rank of 3x3 matrix multiplication is known to be 7 (Strassen).
    # For 3x3 * 3x8, we can view it as 8 independent 3x3 * 3x1 multiplications.
    # Each 3x3 * 3x1 has rank at most 9 (schoolbook) but potentially lower.
    # Actually, 3x3 * 3x1 is just a 3x3 matrix times a vector, which is a linear map from R^3 to R^3 applied to 3 basis vectors... 
    # Wait, 3x3 * 3x8: A is 3x3, B is 3x8, C = A*B is 3x8.
    # This is tensor MM_{3,3,8}.
    
    # The schoolbook decomposition has rank 72.
    # We can try to find a lower rank decomposition.
    
    # One approach: use the known rank for 3x3 (which is 7) and combine.
    # Actually, MM_{3,3,8} can be decomposed using 8 copies of MM_{3,3,1} if we treat each column of B separately.
    # MM_{3,3,1} has rank at most 9 (schoolbook), but is there a known better rank?
    # The rank of MM_{3,3,1} is the number of scalar multiplications needed to compute a 3x3 matrix times a 3-vector.
    # This is equivalent to a linear map from R^3 to R^9 (since 3x3 has 9 entries, and the vector has 3 entries, 
    # actually it's a bilinear map R^3 x R^3 -> R^9? No, A is 3x3, x is 3x1, Ax is 3x1.
    # So it's a bilinear map: (A, x) -> Ax, where A in R^{3x3}, x in R^3, result in R^3.
    # The rank of this tensor is the number of scalar multiplications.
    # For 3x3 * 3x1, the schoolbook rank is 9.
    # Is there a known lower bound or better algorithm? 
    # Actually, computing a 3x3 matrix times a vector requires 9 multiplications in schoolbook.
    # But can we do it with fewer? This is equivalent to the rank of the matrix multiplication tensor MM_{3,3,1}.
    # I believe the rank of MM_{3,3,1} is 9 (it's an open problem if it can be lower, but 9 is known).
    # Actually, wait: MM_{3,3,1} is the same as MM_{1,3,3} by symmetry. 
    # The rank of MM_{3,3,1} is at most 9.
    
    # So if we use 8 independent decompositions of MM_{3,3,1}, we get rank 8*9 = 72, same as schoolbook.
    
    # Alternatively, we can think of MM_{3,3,8} as a single tensor.
    # The naive rank is 72.
    
    # Let's try to use a randomized approach to find a lower rank decomposition.
    # But for now, let's return the schoolbook decomposition which is valid.
    
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
