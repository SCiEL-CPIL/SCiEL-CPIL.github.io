"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Partition indices
    # A: 2 x 5 -> A1 (2x2), A2 (2x3)
    # B: 5 x 5 -> B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    # C: 2 x 5 -> C1 (2x2), C2 (2x3)
    
    # We will build the decomposition by summing contributions from blocks.
    # Block 1: A1 * B11 (2x2x2) -> Use Strassen (Rank 7)
    # Block 2: A2 * B21 (2x3x2) -> Use Schoolbook (Rank 12)
    # Block 3: A1 * B12 (2x2x3) -> Use Schoolbook (Rank 12)
    # Block 4: A2 * B22 (2x3x3) -> Use Schoolbook (Rank 18)
    # Total Rank: 7 + 12 + 12 + 18 = 49
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val
        col += 1

    # Indices for A: i in 0..1, j in 0..4
    # Indices for B: j in 0..4, k in 0..4
    # Indices for C: i in 0..1, k in 0..4
    
    # 1. A1 * B11 (i=0,1; j=0,1; k=0,1)
    # Strassen for 2x2x2
    # A1 = [[a00, a01], [a10, a11]]
    # B11 = [[b00, b01], [b10, b11]]
    # C1 = [[c00, c01], [c10, c11]]
    
    # Strassen 7 terms:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b10)
    # M7 = (a01 - a11) * (b01 + b11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global U, V, W indices.
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Term 1: (a00 + a11) * (b00 + b11)
    # This is not a single rank-1 term in the original basis.
    # Strassen works by defining intermediate products.
    # We must expand the Strassen terms into the original basis.
    # M1 = (A00 + A11) * (B00 + B11)
    # This requires 4 rank-1 terms if we don't share?
    # No, Strassen replaces 8 multiplications with 7.
    # The "multiplications" are bilinear forms.
    # We define 7 new bilinear forms.
    # Form 1: (A00 + A11) * (B00 + B11)
    # This form is a sum of 4 rank-1 tensors in the original basis?
    # No, the rank of the tensor is the number of such forms.
    # Each form is a rank-1 tensor in the *Strassen basis*?
    # No, we need to express the final C as a linear combination of these forms.
    # And each form must be expressible as a rank-1 tensor in the original U,V,W?
    # No. A rank-1 tensor in the decomposition is $u v w$.
    # The product $u v w$ computes a scalar.
    # Strassen's "M1" is a scalar computed from A and B.
    # To implement this in the U,V,W framework, we need to decompose M1 into sum of rank-1 tensors?
    # No, M1 itself is a bilinear form.
    # If M1 can be written as a single rank-1 tensor $u v w$, then we are good.
    # But $(A00 + A11) (B00 + B11)$ is not a rank-1 tensor in the original indices.
    # It is a sum of 4 rank-1 tensors:
    # A00 B00 + A00 B11 + A11 B00 + A11 B11.
    # So Strassen does NOT reduce the number of rank-1 tensors in the U,V,W decomposition directly?
    # Wait. The rank of the tensor is the minimum number of rank-1 tensors.
    # Strassen reduces the number of scalar multiplications, but does it reduce the tensor rank?
    # Yes, the tensor rank of $2 \times 2 \times 2$ is 7.
    # This means there exist 7 rank-1 tensors $u_r, v_r, w_r$ such that the product is the sum.
    # These rank-1 tensors are not simply products of individual entries.
    # They are linear combinations.
    # So we need the explicit rank-7 decomposition of $2 \times 2 \times 2$.
    
    # Explicit Rank 7 Decomposition of 2x2x2:
    # u1 = (1, 0, 1, 1) -> A00, A10, A11? No.
    # Let's use the known decomposition.
    # u_r are vectors in R^4 (A), v_r in R^4 (B), w_r in R^4 (C).
    
    # u1 = (1, 0, 1, 0) ? No.
    # Standard Strassen decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M_i is a bilinear form.
    # M1 = A00 B00 + A00 B11 + A11 B00 + A11 B11.
    # This is a sum of 4 rank-1 terms.
    # So M1 alone requires 4 rank-1 tensors?
    # If so, 7 Ms would require 28 terms?
    # No. The rank is 7. This means there is a basis where it is 7.
    # The vectors u_r, v_r, w_r for the 7 terms are:
    
    # r=1: u=(1,0,1,0), v=(1,0,1,0), w=(1,0,0,1) ? No.
    # Let's look up the explicit vectors for rank 7.
    # u1 = (1, 0, 1, 0) -> A00, A10
    # v1 = (1, 0, 1, 0) -> B00, B10
    # w1 = (1, 0, 0, 1) -> C00, C10?
    
    # Actually, it is easier to just use the schoolbook for 2x2x2 (rank 8) and get total rank 50.
    # But we want 49.
    # If I use Strassen, I need the explicit 7 vectors.
    
    # Vectors for 2x2x2 Rank 7 (Strassen):
    # u1 = (1, 0, 1, 0)  # A00, A10
    # v1 = (1, 0, 1, 0)  # B00, B10
    # w1 = (1, 0, 0, 1)  # C00, C10
    
    # This is getting risky without the exact vectors.
    # I will fall back to rank 50 (schoolbook) to ensure correctness, or rank 49 if I can verify.
    # Given the time, I will return the schoolbook decomposition.
    
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
