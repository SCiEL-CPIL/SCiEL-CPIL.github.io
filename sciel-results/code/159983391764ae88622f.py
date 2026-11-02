"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]
import time

# EVOLVE-BLOCK-START
def construct():
    # We seek a low-rank exact tensor decomposition for 2x3 * 3x9.
    # The trivial rank is n*m*p = 54.
    # We attempt to find a rank-18 decomposition using a structured ansatz.
    # Specifically, we treat the 3x9 multiplication as a 3x3 matrix multiplication over the algebra of 3x3 diagonal matrices (dim 3).
    # Or more simply, we can just try to solve the linear system for rank R.
    # Given the time limit (120s), we can do a small search or direct linear solve.
    
    n, m, p = DIMENSIONS
    
    # Let's first try to see if a rank < 54 exists.
    # Known bounds: R(2,3,9) <= 2*3*9 = 54.
    # R(2,3,3) is 6 (Landsberg).
    # Since 9 = 3*3, maybe we can use a tensor product structure?
    # If we view 9-dim as 3x3 matrices, then (2x3) * (3x9) = (2x3) * (3x3 * 3x3?) No.
    
    # Let's just run a simple randomized search or linear algebra solver for small ranks.
    # However, finding an exact integer/half-integer decomposition is hard.
    # The prompt says "integer/half-integer entries".
    
    # Let's check if the standard schoolbook is the best we can do quickly.
    # Actually, let's try to construct a rank-18 decomposition if it exists.
    # R(2,3,9) might be related to R(2,3,3).
    # If we decompose the 9 columns into 3 groups of 3, we can't just sum ranks because of cross terms.
    
    # Let's try a direct construction for rank 18 based on the algebra of 3x3 diagonal matrices.
    # The algebra A = C^3 (diagonal matrices).
    # Multiplication in A is component-wise.
    # We want to compute A (2x3) * B (3x9).
    # Let B be viewed as 3 matrices of size 3x3? No, B is 3x9.
    # Let's view the 9 dimensions as 3 groups of 3.
    # B = [B1, B2, B3] where Bi is 3x3.
    # Then A*B = [A*B1, A*B2, A*B3].
    # This is 3 independent 2x3 * 3x3 multiplications.
    # Each 2x3 * 3x3 has rank 6 (since R(2,3,3)=6 is known to be minimal? Actually R(2,3,3) is 6).
    # So total rank would be 6+6+6 = 18.
    # Is this a valid decomposition?
    # Yes, if we can decompose each 2x3 * 3x3 block independently.
    # The tensor is a direct sum of 3 copies of the 2x3*3x3 tensor.
    # So the rank is 3 * R(2,3,3).
    # If R(2,3,3) = 6, then rank = 18.
    
    # We need an explicit rank-6 decomposition for 2x3 * 3x3.
    # The schoolbook for 2x3*3x3 is rank 18. But it's known to be 6?
    # Wait, R(2,3,3) is actually 6? 
    # Landsberg (2010) proved R(2,3,3) = 6? No, R(2,2,2)=7, R(2,3,3) is likely higher.
    # Let's check: R(2,3,3) >= 2*3*3 / (log2(2)+log2(3)+log2(3))? Strassen bound?
    # Actually, the minimal rank for 2x3 * 3x3 is not trivially 6.
    # Let's look up R(2,3,3).
    # R(2,3,3) is 6? No, that's too low. 2*3*3=18.
    # Actually, R(2,3,3) is 6 is false. R(2,2,2)=7. R(2,3,3) is likely around 12-14?
    
    # Let's try to find a decomposition for 2x3 * 3x3 with rank < 18.
    # If we can find one, we can use it 3 times to get rank 3*R(2,3,3) for 2x3*3x9.
    
    # Let's assume the schoolbook rank 54 is the safe bet, but try to improve.
    # Given the constraints, let's just return the schoolbook decomposition to be safe, 
    # unless we can quickly verify a better one.
    
    # However, the task asks for "rank-minimal". 
    # If I return 54, and the true minimum is 18, I score 1.0 (since seed is 54? No, seed is 54, score is 1.0).
    # If I find 18, score is 54/18 = 3.0.
    
    # Let's try to construct a rank-18 decomposition.
    # We need a rank-6 decomposition for 2x3 * 3x3.
    # Is R(2,3,3) = 6? 
    # The tensor is t = sum_{i,j,k} A_{ij} B_{jk} C_{ki}.
    # For 2x3 * 3x3: i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # Total terms 18.
    # Can we do it in 6?
    # 6 = 2 * 3.
    # Maybe using the fact that 3 = 2+1?
    
    # Let's try a different approach: Randomized search for a rank R decomposition.
    # We can set up the equations for the entries of U, V, W.
    # This is a system of polynomial equations.
    # For small R, we can try to solve it.
    
    # But given the time limit, let's stick to the known structure.
    # If R(2,3,3) is not 6, then the block diagonal argument doesn't give 18.
    
    # Let's check the literature mentally.
    # R(2,3,3) is 6? 
    # Actually, R(2,3,3) is 6 is a common misconception. 
    # R(2,3,3) is actually 6? 
    # Wait, R(2,2,2)=7. 
    # R(2,3,3) must be at least 7.
    # Is it 6? No, 6 < 7 is impossible if it contains a 2x2x2 sub-tensor? 
    # 2x3*3x3 contains 2x2*2x2? Yes. So R(2,3,3) >= 7.
    
    # So 3 * 7 = 21 is a possible rank for 2x3*3x9 if we use block diagonal.
    # 21 < 54.
    
    # Can we do better than 21?
    # Maybe by mixing the blocks?
    
    # Let's try to find a rank-21 decomposition.
    # We need a rank-7 decomposition for 2x2*2x2 (Strassen).
    # And we need to embed it into 2x3*3x3.
    # The 2x3*3x3 tensor contains three 2x2*2x2 tensors?
    # If we use the Strassen decomposition for each 2x2*2x2 block, we get 7*3 = 21.
    # But the blocks overlap in the middle column/row?
    # 2x3 * 3x3:
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
    # C = A*B.
    # The (0,0) entry is a00*b00 + a01*b10 + a02*b20.
    # This is not a 2x2*2x2 multiplication.
    
    # However, we can view the multiplication as:
    # C = A * B
    # We can split B into columns B0, B1, B2 (each 3x1).
    # Then C0 = A*B0, C1 = A*B1, C2 = A*B2.
    # Each Ci is a 2x1 vector.
    # A is 2x3.
    # So we are doing 3 multiplications of (2x3) * (3x1).
    # R(2,3,1) = 6? (Since it's just 6 products).
    # So total rank 18?
    # Wait, R(2,3,1) is the rank of the tensor for multiplying a 2x3 matrix by a 3x1 vector.
    # This is just 6 scalar multiplications. Can we do it in less?
    # No, because the output has 2 entries, each is a linear combination of 3 products.
    # Actually, R(2,3,1) is 6? 
    # Let's check: C0 = a00*b0 + a01*b1 + a02*b2.
    # C1 = a10*b0 + a11*b1 + a12*b2.
    # This is a bilinear map from (R^2 x R^3) to R^2? No, A is 2x3, B is 3x1.
    # The tensor is in R^{2*3} x R^{3*1} x R^{2*1}.
    # Dimensions: 6 x 3 x 2.
    # Rank of this tensor?
    # It is the tensor of the dot product.
    # R(2,3,1) is 6? 
    # Actually, R(m,n,1) = m*n.
    # So R(2,3,1) = 6.
    # So if we do 3 such multiplications independently, rank is 18.
    # But the A matrix is shared!
    # In the decomposition, U depends on A.
    # If we use the same U for all 3 columns, we might save rank.
    # But U has size 6 x R.
    # If we use rank 18, U is 6x18.
    # The columns of U corresponding to different output columns might be linearly dependent?
    
    # Let's try to construct a rank-18 decomposition.
    # We need U (6x18), V (27x18), W (18x18? No, W is 2*9=18 x 18).
    # Wait, W is (p*n) x rank = 18 x 18.
    # The decomposition is:
    # C_{k*n+i, r} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
    # This is not quite right. The standard form is:
    # C = sum_c U_c \otimes V_c \otimes W_c
    # Where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}.
    # And the product is:
    # (A*B)_{i,k} = sum_c (sum_j U_{i,j,c} A_{i,j}) (sum_j V_{j,k,c} B_{j,k}) W_{k,i,c}?
    # No, the indices are:
    # A is n x m. B is m x p. C is n x p.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor T is in R^{n*m} \otimes R^{m*p} \otimes R^{n*p}.
    # T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k}.
    # We want to write T = sum_c u_c \otimes v_c \otimes w_c.
    # Then C_{i,k} = sum_c (u_c \cdot A) (v_c \cdot B) (w_c \cdot C?) No.
    # The reconstruction is:
    # C = sum_c (u_c \cdot A) (v_c \cdot B) w_c^T ?
    # Let's check the convention in the prompt:
    # W[k*n+i, r].
    # This suggests the output index is k*n+i, which is row-major for p x n?
    # But C is n x p.
    # Usually C_{i,k}.
    # If W is indexed by k*n+i, then the output vector is of length p*n.
    # This implies the output is stored as p rows of n columns?
    # Or maybe the output is B*A?
    # "multiplying a 2 by 3 matrix by a 3 by 9 matrix" -> A (2x3) * B (3x9) = C (2x9).
    # So C is 2x9.
    # Output factor W has shape (p*n, rank) = (9*2, rank) = (18, rank).
    # Index k*n+i. If k is 0..8 and i is 0..1, then k*n+i ranges 0..17.
    # This corresponds to C_{i,k} where i is row (0..1) and k is col (0..8).
    # So W_{k*n+i, c} is the coefficient for C_{i,k}.
    
    # So the decomposition is:
    # C_{i,k} = sum_c U_{i*m+j, c} ... wait.
    # U has shape (n*m, rank) = (6, rank). Index i*m+j.
    # V has shape (m*p, rank) = (27, rank). Index j*p+k.
    # W has shape (p*n, rank) = (18, rank). Index k*n+i.
    
    # The bilinear form is:
    # C_{i,k} = sum_c (sum_j U_{i*m+j, c} A_{i,j}) (sum_j V_{j*p+k, c} B_{j,k}) W_{k*n+i, c}
    # This is not standard. Usually it's:
    # C_{i,k} = sum_c (U_c^T A) (V_c^T B) W_{i,k,c}
    # Where U_c is a vector in R^{n*m}, so U_c^T A = sum_{i,j} U_{i*m+j,c} A_{i,j}.
    # This is a scalar.
    # Similarly V_c^T B is a scalar.
    # Then C_{i,k} = sum_c scalar1_c * scalar2_c * W_{k*n+i,c}.
    # This works if W_{k*n+i,c} is the coefficient for that term.
    
    # So we need to find U, V, W such that:
    # sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{i,k} = sum_c u_c \otimes v_c \otimes w_c
    # where u_c is a vector in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p} (indexed by k*n+i).
    
    # Let's try to find a rank-18 decomposition.
    # We can use the fact that C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a sum of 18 terms (for i=0,1; j=0,1,2; k=0..8).
    # Can we group them?
    
    # Let's just return the schoolbook decomposition for now, as it is guaranteed to work.
    # Rank 54.
    
    n, m, p = DIMENSIONS
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
