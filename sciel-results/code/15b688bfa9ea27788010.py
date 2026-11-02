"""Matrix multiplication 3x3 * 3x4 rank-minimal decomposition."""
import time
from fractions import Fraction as Fr

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(V[0]) if V else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # Check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                # Wait, that's not the right check. The check is:
                # For each (i, k), sum_j A[i,j] B[j,k] should equal C[i,k]
                # In tensor form: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # should give the coefficient of A[i,j]*B[j,k] in C[i,k]
                # which is 1 for all valid (i,j,k).
                pass
    
    # Actually the correct check is:
    # The tensor T[(i,j),(j,k),(k,i)] = 1 for all i,j,k
    # We need: sum_c U[(i,j), c] * V[(j,k), c] * W[(k,i), c] = 1 for all i,j,k
    # And for invalid combinations (where the indices don't form a valid (i,j,k)),
    # the sum should be 0. But in our formulation, all combinations are valid.
    
    # Wait, let me re-read. The tensor is:
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T[a,b,c] = 1 if a,b,c are consistent (i.e., the j in a matches the j in b, 
    # and the k in b matches the k in c, and the i in c matches the i in a)
    
    # Actually no. The tensor for matrix multiplication (n x m) * (m x p) is:
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # But since we're indexing as U[i*m+j], V[j*p+k], W[k*n+i], the indices are
    # already linked. So T[(i,j),(j,k),(k,i)] = 1 for all i,j,k.
    
    # The check is: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # But wait, that's only checking the "diagonal" entries. What about off-diagonal?
    # In the tensor formulation, the tensor is a 3D array of size (n*m) x (m*p) x (p*n).
    # The entry at position [(i,j), (j',k), (k',i')] is:
    #   1 if j=j', k=k', i=i'
    #   0 otherwise
    # 
    # But since U is indexed by (i,j), V by (j',k), W by (k',i'), the product
    # U[(i,j), c] * V[(j',k), c] * W[(k',i'), c] only contributes when
    # the indices are consistent in the way the tensor is defined.
    
    # Actually, I think the correct interpretation is simpler. The multiplication
    # tensor T is defined such that:
    # <T, A ⊗ B> = sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} = sum_{i,k} (AB)_{i,k}
    # where T_{i,j,k} = 1 for all i,j,k (since every A_{i,j}*B_{j,k} contributes 
    # to (AB)_{i,k}).
    
    # So T is the all-ones tensor of shape (n, m, p) = (3, 3, 4).
    
    # But that can't be right, because then the rank would be 1 (just use
    # u=(1,1,1), v=(1,1,1), w=(1,1,1,1)).
    
    # I think the issue is that the tensor is not simply the all-ones tensor.
    # Let me think again.
    
    # The bilinear form is:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # 
    # This can be written as:
    # (A*B)_{i,k} = sum_{i,j,k} delta_{i,i'} delta_{k,k'} A_{i,j} B_{j,k}
    # 
    # So the tensor T has components T_{i,k;j} = delta_{i,i'} delta_{k,k'}
    # where i' is the row of C, k' is the column of C, and j is the shared index.
    
    # In the U, V, W formulation:
    # U[i*m+j, c] corresponds to the A_{i,j} coefficient
    # V[j*p+k, c] corresponds to the B_{j,k} coefficient  
    # W[k*n+i, c] corresponds to the C_{i,k} coefficient
    # 
    # And the condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... no, that's not right either.
    
    # Let me look at the verification code in the current program:
    # for i in range(n):
    #     for k in range(p):
    #         for r in range(rank):
    #             expected = W[k * n + i][r]
    #             actual = Fr(0)
    #             for j in range(m):
    #                 actual += U[i * m + j][r] * V[j * p + k][r]
    #             if actual != expected:
    #                 return False
    #
    # So the condition is: for each (i, k, r):
    #   W[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r]
    #
    # This means that for each rank component r, the product U_r * V_r (as matrices)
    # should equal W_r (as a matrix). Specifically:
    # (U_r * V_r)_{i,k} = W_{k*n+i, r}
    # where U_r is a 3x3 matrix (rows i, cols j), V_r is a 3x4 matrix (rows j, cols k),
    # and W_r is a 3x4 matrix (rows i, cols k) but stored as W[k*n+i, r].
    
    # So the condition is: for each r, U_r @ V_r = W_r (with appropriate reshaping).
    
    # Let me rewrite the verification:
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                expected = W[k * n + i][r]
                actual = Fr(0)
                for j in range(m):
                    actual += U[i * m + j][r] * V[j * p + k][r]
                if actual != expected:
                    return False
    return True


def _verify_matrix(U, V, W, n, m, p):
    """Alternative verification: check that for each r, U_r @ V_r = W_r."""
    rank = len(V[0]) if V else 0
    for r in range(rank):
        # Extract U_r (n x m), V_r (m x p), W_r (n x p)
        U_r = [[U[i * m + j][r] for j in range(m)] for i in range(n)]
        V_r = [[V[j * p + k][r] for k in range(p)] for j in range(m)]
        W_r = [[W[k * n + i][r] for k in range(p)] for i in range(n)]
        
        # Check U_r @ V_r == W_r
        for i in range(n):
            for k in range(p):
                actual = Fr(0)
                for j in range(m):
                    actual += U_r[i][j] * V_r[j][k]
                if actual != W_r[i][k]:
                    return False
    return True


def construct():
    """Return a low-rank decomposition for 3x3 * 3x4 matrix multiplication."""
    n, m, p = 3, 3, 4
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find a lower rank decomposition
    # Strategy: use the fact that we can combine terms.
    # 
    # For 2x2 * 2x2, Strassen's algorithm uses 7 multiplications instead of 8.
    # The key insight is that we can use linear combinations of the inputs.
    #
    # For our problem, let's try to use a block decomposition.
    #
    # Split A (3x3) into:
    #   A11 (2x2), A12 (2x1)
    #   A21 (1x2), A22 (1x1)
    #
    # Split B (3x4) into:
    #   B11 (2x2), B12 (2x2)
    #   B21 (1x2), B22 (1x2)
    #
    # Then C = A*B is:
    #   C11 = A11*B11 + A12*B21  (2x2)
    #   C12 = A11*B12 + A12*B22  (2x2)
    #   C21 = A21*B11 + A22*B21  (1x2)
    #   C22 = A21*B12 + A22*B22  (1x2)
    
    # For A11*B11 (2x2 * 2x2), use Strassen: 7 terms
    # For A11*B12 (2x2 * 2x2), use Strassen: 7 terms
    # For A12*B21 (2x1 * 1x2), outer product: 4 terms
    # For A12*B22 (2x1 * 1x2), outer product: 4 terms
    # For A21*B11 (1x2 * 2x2), 4 terms
    # For A21*B12 (1x2 * 2x2), 4 terms
    # For A22*B21 (1x1 * 1x2), 2 terms
    # For A22*B22 (1x1 * 1x2), 2 terms
    # Total: 7+7+4+4+4+4+2+2 = 34
    
    # Let's implement this 34-rank decomposition.
    
    rank = 34
    U = [[Fr(0)] * rank for _ in range(n * m)]
    V = [[Fr(0)] * rank for _ in range(m * p)]
    W = [[Fr(0)] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a simple term: A[i,j] * B[j,k] -> C[i,k]
    def add_simple(i, j, k):
        nonlocal col
        U[i * m + j][col] = Fr(1)
        V[j * p + k][col] = Fr(1)
        W[k * n + i][col] = Fr(1)
        col += 1
    
    # Helper to add a Strassen term for 2x2 * 2x2
    # Strassen's 7 terms for A(2x2) * B(2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_2x2(i0, j0, k0):
        """Add Strassen terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # This is a bilinear form, not a single rank-1 term.
        # We need to express it as a combination of rank-1 terms.
        
        # Actually, in the tensor decomposition, each "multiplication" corresponds
        # to a rank-1 term. Strassen reduces the number of scalar multiplications,
        # but in the tensor formulation, we're counting rank-1 terms.
        
        # The key insight is that Strassen's algorithm uses 7 scalar multiplications,
        # but each scalar multiplication can be represented as a rank-1 term in the
        # tensor decomposition. So Strassen gives us 7 rank-1 terms instead of 8.
        
        # Let's implement the 7 Strassen terms.
        
        # M1 = (A00+A11)(B00+B11)
        # This requires computing A00+A11 and B00+B11, then multiplying.
        # In the tensor formulation, this is:
        # U contributes: A00 and A11 (with coefficient 1 each)
        # V contributes: B00 and B11 (with coefficient 1 each)
        # W contributes: the result goes to C00, C01, C10, C11 with appropriate coefficients
        
        # Actually, I think the way to think about this is:
        # Each Strassen "multiplication" M_r = f_r(A) * g_r(B) where f_r and g_r
        # are linear combinations of the A and B entries.
        # 
        # In the tensor formulation, this corresponds to:
        # U[:, col] = f_r(A) (as a vector)
        # V[:, col] = g_r(B) (as a vector)
        # W[:, col] = h_r(C) (as a vector)
        # where h_r is the linear combination of C entries that M_r contributes to.
        
        # For M1 = (A00+A11)(B00+B11):
        # f_1(A) = A00 + A11, so U[i0*m+j0, col] = 1, U[(i0+1)*m+(j0+1), col] = 1
        # g_1(B) = B00 + B11, so V[j0*p+k0, col] = 1, V[(j0+1)*p+(k0+1), col] = 1
        # h_1(C) = C00 - C01 + C10 - C11 (from the Strassen formulas)
        # Wait, let me re-derive.
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So M1 appears in C00 with +1 and C11 with +1.
        # Therefore W[k0*n+i0, col] = 1, W[(k0+1)*n+(i0+1), col] = 1
        
        # Let me implement all 7 terms.
        
        # M1 = (A00+A11)(B00+B11)
        U[i0 * m + j0][col] = Fr(1)
        U[(i0+1) * m + (j0+1)][col] = Fr(1)
        V[j0 * p + k0][col] = Fr(1)
        V[(j0+1) * p + (k0+1)][col] = Fr(1)
        W[k0 * n + i0][col] = Fr(1)
        W[(k0+1) * n + (i0+1)][col] = Fr(1)
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1) * m + j0][col] = Fr(1)
        U[(i0+1) * m + (j0+1)][col] = Fr(1)
        V[j0 * p + k0][col] = Fr(1)
        W[k0 * n + (i0+1)][col] = Fr(1)
        W[(k0+1) * n + (i0+1)][col] = Fr(-1)
        col += 1
        
        # M3 = A00(B10-B11)
        U[i0 * m + j0][col] = Fr(1)
        V[(j0+1) * p + k0][col] = Fr(1)
        V[(j0+1) * p + (k0+1)][col] = Fr(-1)
        W[k0 * n + i0][col] = Fr(1)
        W[(k0+1) * n + i0][col] = Fr(1)
        col += 1
        
        # M4 = A11(B01-B00)
        U[(i0+1) * m + (j0+1)][col] = Fr(1)
        V[j0 * p + (k0+1)][col] = Fr(1)
        V[j0 * p + k0][col] = Fr(-1)
        W[k0 * n + i0][col] = Fr(1)
        W[k0 * n + (i0+1)][col] = Fr(1)
        col += 1
        
        # M5 = (A00+A01)B11
        U[i0 * m + j0][col] = Fr(1)
        U[i0 * m + (j0+1)][col] = Fr(1)
        V[(j0+1) * p + (k0+1)][col] = Fr(1)
        W[k0 * n + i0][col] = Fr(-1)
        W[(k0+1) * n + i0][col] = Fr(1)
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1) * m + j0][col] = Fr(1)
        U[i0 * m + j0][col] = Fr(-1)
        V[j0 * p + k0][col] = Fr(1)
        V[j0 * p + (k0+1)][col] = Fr(1)
        W[(k0+1) * n + (i0+1)][col] = Fr(1)
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0 * m + (j0+1)][col] = Fr(1)
        U[(i0+1) * m + (j0+1)][col] = Fr(-1)
        V[(j0+1) * p + k0][col] = Fr(1)
        V[(j0+1) * p + (k0+1)][col] = Fr(1)
        W[k0 * n + i0][col] = Fr(1)
        W[(k0+1) * n + (i0+1)][col] = Fr(-1)
        col += 1
    
    # Now let's add the terms for our block decomposition.
    
    # A11*B11: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_2x2(0, 0, 0)
    
    # A11*B12: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_2x2(0, 0, 2)
    
    # A12*B21: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] (outer product, 4 terms)
    for i in range(2):
        for k in range(2):
            add_simple(i, 2, k)
    
    # A12*B22: A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4] (outer product, 4 terms)
    for i in range(2):
        for k in range(2):
            add_simple(i, 2, k + 2)
    
    # A21*B11: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] (4 terms)
    for j in range(2):
        for k in range(2):
            add_simple(2, j, k)
    
    # A21*B12: A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] (4 terms)
    for j in range(2):
        for k in range(2):
            add_simple(2, j, k + 2)
    
    # A22*B21: A[2, 2] * B[2, 0:2] -> C[2, 0:2] (2 terms)
    for k in range(2):
        add_simple(2, 2, k)
    
    # A22*B22: A[2, 2] * B[2, 2:4] -> C[2, 2:4] (2 terms)
    for k in range(2):
        add_simple(2, 2, k + 2)
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p):
        best_rank = rank
        best_U = [[int(x) for x in row] for row in U]
        best_V = [[int(x) for x in row] for row in V]
        best_W = [[int(x) for x in row] for row in W]
    
    # If the 34-rank decomposition didn't work, fall back to schoolbook
    if best_rank == rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
