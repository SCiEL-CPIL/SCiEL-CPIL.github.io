"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p):
    """Check that U @ (V^T) reconstructs the multiplication tensor correctly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor T[i,j,k] for A*B
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We can check by computing the bilinear map
    # For each c, the term is: sum_{i,j,k} U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] * A[i,j] * B[j,k]
    # Let's just check a few random instances
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        AB = A @ B
        # Compute via tensor decomposition
        result = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]  # (n*m,)
            vc = V[:, c]  # (m*p,)
            wc = W[:, c]  # (p*n,)
            # uc[i*m+j] * vc[j*p+k] * wc[k*n+i] * A[i,j] * B[j,k]
            # Sum over i,j,k
            # We can do this by reshaping
            u_reshaped = uc.reshape(n, m)
            v_reshaped = vc.reshape(m, p)
            w_reshaped = wc.reshape(p, n)
            # Term = sum_{i,j,k} u_reshaped[i,j] * A[i,j] * v_reshaped[j,k] * B[j,k] * w_reshaped[k,i]
            # Let M1[i,j] = u_reshaped[i,j] * A[i,j]
            # Let M2[j,k] = v_reshaped[j,k] * B[j,k]
            # Then term = sum_{i,j,k} M1[i,j] * M2[j,k] * w_reshaped[k,i]
            M1 = u_reshaped * A
            M2 = v_reshaped * B
            # sum_{j} M1[i,j] * M2[j,k] * w_reshaped[k,i]
            # = sum_{j} M1[i,j] * (M2[j,k] * w_reshaped[k,i])
            # Let's compute: for each i, sum_j M1[i,j] * (sum_k M2[j,k] * w_reshaped[k,i])
            # M2 @ diag? No.
            # temp[i,k] = sum_j M1[i,j] * M2[j,k] = (M1 @ M2)[i,k]
            temp = M1 @ M2  # (n, p)
            # Now sum_k temp[i,k] * w_reshaped[k,i]
            # w_reshaped[k,i] = wc[k*n+i]
            # This is element-wise product then sum over k
            for i in range(n):
                for k in range(p):
                    result[i, k] += temp[i, k] * wc[k * n + i]
        if not np.allclose(result, AB, atol=1e-10):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Strassen-style decomposition for 2x4 by 4x6
    # Partition A (2x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    #   A00 = A[0, 0:2], A01 = A[0, 2:4], A10 = A[1, 0:2], A11 = A[1, 2:4]
    # Partition B (4x6) into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]]
    #   B00 = B[0:2, 0:3], B01 = B[0:2, 3:6], etc.
    #   B10 = B[2:4, 0:3], B11 = B[2:4, 3:6], B12 = B[2:4, 3:6] wait no
    #   B10 = B[2:4, 0:3], B11 = B[2:4, 3:6]
    #   Actually B is 4x6, so rows 0:2 and 2:4, cols 0:3 and 3:6
    #   B00 = B[0:2, 0:3], B01 = B[0:2, 3:6]
    #   B10 = B[2:4, 0:3], B11 = B[2:4, 3:6]
    #
    # Strassen for C = A*B where A is 2x4, B is 4x6:
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Standard Strassen uses 7 multiplications for 2x2 block matrices.
    # Here each "block multiplication" is (1x2)*(2x3) -> (1x3)
    # So each block product has rank 2*3=6 (schoolbook) but we use Strassen on the 2x2 structure.
    #
    # The 7 Strassen terms:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # Then:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Each Mi is a (1x2)*(2x3) product, which in the tensor decomposition
    # corresponds to rank 6 terms (2*3 schoolbook for each 1x3 output).
    # Wait, no. Each Mi = X * Y where X is a row vector (length 2) and Y is a 2x3 matrix.
    # The product X*Y is 1x3. In the tensor decomposition, each such product
    # contributes rank = 2*3 = 6 terms? No, that's schoolbook for a 2x3 multiplication.
    # Actually, X*Y where X is 1x2 and Y is 2x3: the bilinear map has rank 2*3=6 in schoolbook,
    # but we can do it in rank 6 (since it's just 2 scalar multiplications... wait).
    #
    # Let me reconsider. X*Y = [x0*y00 + x1*y10, x0*y01 + x1*y11, x0*y02 + x1*y12]
    # This is 3 outputs, each a sum of 2 products. The tensor rank of this bilinear map is 6 (schoolbook).
    # But actually, this is just 2 scalar multiplications (x0*y_row0 and x1*y_row1) combined linearly.
    # The rank of this specific tensor is actually 6 in general? No, it's 2*3=6 for the full bilinear map.
    #
    # So 7 Strassen terms * 6 rank each = 42 total rank.
    #
    # Let me define the 7 Strassen combinations explicitly.
    
    # U shape: (n*m, rank) = (8, 42)
    # V shape: (m*p, rank) = (24, 42)
    # W shape: (p*n, rank) = (12, 42)
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper: for a Strassen term with X = sum of A-row-segments and Y = sum of B-row-segments,
    # we need to add rank-6 contributions.
    # X is a linear combination of rows of A, Y is a linear combination of rows of B.
    # The product X*Y contributes to specific entries of C.
    
    # Let's define each Strassen term:
    # For each term, X is a 1x2 vector (linear combo of 2 rows of A), 
    # Y is a 2x3 matrix (linear combo of 2 rows of B).
    # X*Y is 1x3.
    
    # The 7 terms:
    # Term 0: M1 = (A00 + A11) * (B00 + B11)
    #   A00 = A[0, 0:2], A11 = A[1, 2:4]
    #   X = [A[0,0]+A[1,2], A[0,1]+A[1,3]]
    #   B00 = B[0:2, 0:3], B11 = B[2:4, 3:6]
    #   Y = B00 + B11 (2x3)
    #   Contributes to C00 and C11 (with signs from Strassen formulas)
    #   C00 += M1, C11 += M1
    #   So W for this term: for output position (i, k) in C, W[k*n+i, r] 
    #   C00 is C[0, 0:3], C11 is C[1, 3:6]
    #   So for k=0,1,2: W[k*2+0, r] += 1 and W[(k+3)*2+1, r] += 1
    
    # Term 1: M2 = (A10 + A11) * B00
    #   A10 = A[1, 0:2], A11 = A[1, 2:4]
    #   X = [A[1,0]+A[1,2], A[1,1]+A[1,3]]
    #   Y = B00 = B[0:2, 0:3]
    #   C10 += M2, C11 -= M2
    #   C10 is C[1, 0:3], C11 is C[1, 3:6]
    #   W[(k)*2+1, r] += 1 and W[(k+3)*2+1, r] -= 1 for k=0,1,2
    
    # Term 2: M3 = A00 * (B01 - B11)
    #   A00 = A[0, 0:2]
    #   X = [A[0,0], A[0,1]]
    #   Y = B01 - B11 = B[0:2, 3:6] - B[2:4, 3:6]
    #   C01 += M3, C11 += M3
    #   C01 is C[0, 3:6], C11 is C[1, 3:6]
    #   W[(k+3)*2+0, r] += 1 and W[(k+3)*2+1, r] += 1 for k=0,1,2
    
    # Term 3: M4 = A11 * (B10 - B00)
    #   A11 = A[1, 2:4]
    #   X = [A[1,2], A[1,3]]
    #   Y = B10 - B00 = B[2:4, 0:3] - B[0:2, 0:3]
    #   C00 += M4, C10 += M4
    #   C00 is C[0, 0:3], C10 is C[1, 0:3]
    #   W[(k)*2+0, r] += 1 and W[(k)*2+1, r] += 1 for k=0,1,2
    
    # Term 4: M5 = (A00 + A01) * B11
    #   A00 = A[0, 0:2], A01 = A[0, 2:4]
    #   X = [A[0,0]+A[0,2], A[0,1]+A[0,3]]
    #   Y = B11 = B[2:4, 3:6]
    #   C00 -= M5, C01 += M5
    #   C00 is C[0, 0:3], C01 is C[0, 3:6]
    #   W[(k)*2+0, r] -= 1 and W[(k+3)*2+0, r] += 1 for k=0,1,2
    
    # Term 5: M6 = (A10 - A00) * (B00 + B01)
    #   A10 = A[1, 0:2], A00 = A[0, 0:2]
    #   X = [A[1,0]-A[0,0], A[1,1]-A[0,1]]
    #   Y = B00 + B01 = B[0:2, 0:3] + B[0:2, 3:6]
    #   C11 += M6
    #   C11 is C[1, 3:6]
    #   W[(k+3)*2+1, r] += 1 for k=0,1,2
    
    # Term 6: M7 = (A01 - A11) * (B10 + B11)
    #   A01 = A[0, 2:4], A11 = A[1, 2:4]
    #   X = [A[0,2]-A[1,2], A[0,3]-A[1,3]]
    #   Y = B10 + B11 = B[2:4, 0:3] + B[2:4, 3:6]
    #   C00 += M7, C01 += M7 (wait, let me recheck Strassen)
    #
    # Standard Strassen:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22)B11
    # P3 = A11(B21 - B11)
    # P4 = A22(B12 - B11)
    # P5 = (A11 + A12)B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    #
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    #
    # Mapping to our notation:
    # A11=A00, A12=A01, A21=A10, A22=A11
    # B11=B00, B12=B01, B21=B10, B22=B11
    #
    # C00 = C11 in Strassen = P1 + P4 - P5 + P7
    # C01 = C12 in Strassen = P3 + P5
    # C10 = C21 in Strassen = P2 + P4
    # C11 = C22 in Strassen = P1 - P2 + P3 + P6
    #
    # So:
    # M1=P1, M2=P2, M3=P3, M4=P4, M5=P5, M6=P6, M7=P7
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now let's code each term.
    # For each term, X is a 1x2 vector (coefficients on A entries), Y is 2x3 (coefficients on B entries).
    # The product X*Y is 1x3.
    # For the tensor decomposition, each (X,Y) pair with X having 2 nonzero A-entries and Y having 
    # 2x3=6 nonzero B-entries gives rank 6 contributions? No.
    #
    # Actually, X*Y where X is 1x2 and Y is 2x3:
    # result[k] = X[0]*Y[0,k] + X[1]*Y[1,k] for k=0,1,2
    # This is a bilinear map from (X,Y) to result.
    # In the matrix multiplication tensor, this contributes:
    # sum_{a=0}^{1} sum_{b=0}^{2} X[a] * Y[a,b] * (output)[b]
    # 
    # In the U,V,W decomposition:
    # U[i*m+j, r] for A[i,j]
    # V[j*p+k, r] for B[j,k]
    # W[k*n+i, r] for C[i,k]
    #
    # For a given term with X = sum_{(i,j) in Sx} c_{ij} A[i,j] and Y = sum_{(j',k) in Sy} d_{j'k} B[j',k],
    # the contribution to C[i_out, k_out] is:
    # sum_{(i,j) in Sx} sum_{(j',k) in Sy} c_{ij} * d_{j'k} * A[i,j] * B[j',k] * (contribution to C[i_out,k_out])
    #
    # For this to be a valid rank-1 term in the decomposition, we need:
    # U[i*m+j, r] = c_{ij} for (i,j) in Sx, 0 otherwise
    # V[j'*p+k, r] = d_{j'k} for (j',k) in Sy, 0 otherwise
    # W[k_out*n+i_out, r] = (contribution coefficient) for the relevant output positions
    #
    # But the issue is that X*Y = sum_a sum_b X[a] Y[a,b] e_b
    # So the output is a linear combination of 3 basis vectors (the 3 columns of the result).
    # This means we need 3 separate rank-1 terms, one for each output position b.
    #
    # For output position b (0, 1, 2):
    # Contribution = sum_a X[a] * Y[a,b]
    # This is a product of a linear form in A (X[a] for a=0,1) and a linear form in B (Y[a,b] for a=0,1).
    # So it's a rank-1 term in the bilinear map.
    #
    # So for each Strassen term, we get 3 rank-1 contributions (one per output column).
    # Total: 7 terms * 3 = 21? But wait, that gives rank 21, not 42.
    #
    # I think I'm confusing myself. Let me reconsider.
    #
    # The full matrix multiplication tensor for 2x4 * 4x6 has rank at most 48 (schoolbook).
    # Each Strassen term replaces 4 block multiplications with 7, but each block multiplication here
    # is (1x2)*(2x3) which has schoolbook rank 6.
    # So 7 terms * 6 = 42.
    #
    # But when I said "3 rank-1 contributions per term", that's only 21. The discrepancy is because
    # each (1x2)*(2x3) product is NOT a single rank-1 term; it's a bilinear map of rank 6.
    #
    # Let me be more careful. X*Y where X is 1x2 and Y is 2x3:
    # result[b] = X[0]*Y[0,b] + X[1]*Y[1,b]
    # This is a bilinear map from (X, Y) to result. The tensor rank of this map is 6 (schoolbook:
    # 2*3 individual scalar products).
    #
    # So each Strassen term has rank 6, giving total rank 42.
    #
    # To decompose each (1x2)*(2x3) product into rank-1 terms:
    # result[b] = X[0]*Y[0,b] + X[1]*Y[1,b]
    # We can write this as:
    # For b=0: X[0]*Y[0,0] + X[1]*Y[1,0]
    # For b=1: X[0]*Y[0,1] + X[1]*Y[1,1]
    # For b=2: X[0]*Y[0,2] + X[1]*Y[1,2]
    #
    # Each of these is a sum of 2 products, so the rank-1 decomposition has 2 terms per output,
    # total 6 terms per Strassen term.
    #
    # So for term with X coefficients c0, c1 on A entries, and Y coefficients d00,d01,d02 on B row 0
    # and d10,d11,d12 on B row 1:
    # 
    # Output b=0:
    #   Term 1: U has c0 on A entry for X[0], V has d00 on B[0,0], W has 1 on C[i_out, 0]
    #   Term 2: U has c1 on A entry for X[1], V has d10 on B[1,0], W has 1 on C[i_out, 0]
    #
    # Output b=1:
    #   Term 3: U has c0, V has d01 on B[0,1], W has 1 on C[i_out, 1]
    #   Term 4: U has c1, V has d11 on B[1,1], W has 1 on C[i_out, 1]
    #
    # Output b=2:
    #   Term 5: U has c0, V has d02 on B[0,2], W has 1 on C[i_out, 2]
    #   Term 6: U has c1, V has d12 on B[1,2], W has 1 on C[i_out, 2]
    #
    # But the W coefficients depend on which output position (i_out, k_out) this contributes to.
    #
    # This is getting complex. Let me just enumerate all 42 rank-1 terms directly.
    
    # Define the 7 Strassen terms with their X and Y components.
    # X: list of (a_index, coeff) where a_index is (i, j) in A
    # Y: list of (b_index, coeff) where b_index is (j, k) in B, but Y is 2x3 so
    #    Y[a, b] where a is 0 or 1 (row of B-block) and b is 0,1,2 (col)
    
    # Let me define each term by:
    # x_terms: list of (i, j, coeff) for A[i,j]
    # y_terms: list of (j, k, coeff) for B[j,k]
    # w_terms: list of (i_out, k_out, coeff) for C[i_out, k_out]
    
    terms = []
    
    # M1 = (A00 + A11) * (B00 + B11)
    # X: A[0,0]+1, A[0,1]+1, A[1,2]+1, A[1,3]+1
    # Y: B[0,0]+1, B[0,1]+1, B[0,2]+1, B[2,3]+1, B[2,4]+1, B[2,5]+1
    # C: C[0,0]+1, C[0,1]+1, C[0,2]+1, C[1,3]+1, C[1,4]+1, C[1,5]+1
    # But this is NOT a single rank-1 term! It's a product of two linear forms.
    # The rank-1 decomposition requires that for each rank-1 term, U, V, W are such that
    # U[i*m+j, r] * V[j'*p+k, r] * W[k_out*n+i_out, r] picks out exactly one A entry, one B entry,
    # and one C entry.
    #
    # So I need to expand each Strassen term into rank-1 terms.
    #
    # For M1 = (A[0,0]+A[0,1]+A[1,2]+A[1,3]) * (B[0,0]+B[0,1]+B[0,2]+B[2,3]+B[2,4]+B[2,5])
    # This is a product of a linear form in A and a linear form in B.
    # The product expands to 4*6 = 24 terms, but many cancel or combine.
    # Actually no, in the tensor decomposition, we need to represent the bilinear map.
    #
    # I think the correct approach is:
    # Each Strassen term Mi = Xi * Yi where Xi is a linear form on A and Yi is a linear form on B.
    # The contribution to C is a linear combination of the products.
    # To decompose Mi into rank-1 terms of the full tensor, we note that:
    # Xi * Yi = (sum_{(i,j) in Sx} c_{ij} A[i,j]) * (sum_{(j',k) in Sy} d_{j'k} B[j',k])
    # = sum_{(i,j) in Sx} sum_{(j',k) in Sy} c_{ij} d_{j'k} A[i,j] B[j',k]
    # Each such product A[i,j] B[j',k] contributes to C[i, k] (if j==j', otherwise it's 0 in the
    # actual multiplication, but in the Strassen decomposition, the intermediate products are
    # abstract).
    #
    # Wait, I think I need to reconsider the whole approach. The Strassen algorithm works on
    # block matrices, and the "multiplications" are of blocks. In the tensor decomposition,
    # each block multiplication (1x2)*(2x3) is itself a bilinear map that needs to be decomposed.
    #
    # Let me just directly construct the 42 rank-1 terms by thinking of the Strassen algorithm
    # as replacing the 4 block multiplications with 7, and each block multiplication (1x2)*(2x3)
    # is decomposed into 6 rank-1 terms (schoolbook for 2x3).
    
    # Schoolbook for (1x2) * (2x3) -> (1x3):
    # result[0] = A[0]*B[0,0] + A[1]*B[1,0]
    # result[1] = A[0]*B[0,1] + A[1]*B[1,1]
    # result[2] = A[0]*B[0,2] + A[1]*B[1,2]
    # This has 6 rank-1 terms:
    # (A[0], B[0,0], C[0]), (A[1], B[1,0], C[0]),
    # (A[0], B[0,1], C[1]), (A[1], B[1,1], C[1]),
    # (A[0], B[0,2], C[2]), (A[1], B[1,2], C[2])
    
    # So for each Strassen term, I need to:
    # 1. Express X as a linear combination of 2 specific A entries
    # 2. Express Y as a 2x3 matrix of linear combinations of B entries
    # 3. Expand into 6 rank-1 terms
    
    # Let me define each Strassen term with explicit indices.
    
    # Term 0: M1 = (A00 + A11) * (B00 + B11)
    # X = A[0,0] + A[0,1] + A[1,2] + A[1,3]  (but this is 4 terms, not 2)
    # Hmm, A00 is 1x2, A11 is 1x2, so X = A00 + A11 is 1x2 with entries:
    # X[0] = A[0,0] + A[1,2], X[1] = A[0,1] + A[1,3]
    # Y = B00 + B11 is 2x3 with entries:
    # Y[0,0] = B[0,0] + B[2,3], Y[0,1] = B[0,1] + B[2,4], Y[0,2] = B[0,2] + B[2,5]
    # Y[1,0] = B[1,0] + B[3,3], Y[1,1] = B[1,1] + B[3,4], Y[1,2] = B[1,2] + B[3,5]
    
    # So X has 2 entries, each a sum of 2 A entries.
    # Y has 6 entries, each a sum of 2 B entries.
    
    # The product X*Y:
    # result[b] = X[0]*Y[0,b] + X[1]*Y[1,b]
    
    # This is a bilinear map. To decompose into rank-1 terms of the full tensor:
    # Each rank-1 term picks one A entry, one B entry, and one C entry.
    
    # X[0] = A[0,0] + A[1,2]
    # X[1] = A[0,1] + A[1,3]
    # Y[0,b] = B[0,b] + B[2,b+3]
    # Y[1,b] = B[1,b] + B[3,b+3]
    
    # result[b] = (A[0,0] + A[1,2])*(B[0,b] + B[2,b+3]) + (A[0,1] + A[1,3])*(B[1,b] + B[3,b+3])
    # = A[0,0]*B[0,b] + A[0,0]*B[2,b+3] + A[1,2]*B[0,b] + A[1,2]*B[2,b+3]
    #   + A[0,1]*B[1,b] + A[0,1]*B[3,b+3] + A[1,3]*B[1,b] + A[1,3]*B[3,b+3]
    
    # Each of these 8 products (for each b) is a rank-1 term in the bilinear map from (A,B) to result[b].
    # But result[b] contributes to specific C entries.
    
    # For M1, result[b] contributes to C[0,b] and C[1,b+3] (both with +1).
    
    # So for each b in 0,1,2, we have 8 rank-1 terms, each contributing to C[0,b] and C[1,b+3].
    # That's 24 rank-1 terms for M1 alone, which is too many.
    
    # I think the issue is that I'm overcomplicating this. Let me look at it differently.
    
    # The key insight: in the tensor decomposition U, V, W, each rank-1 term is:
    # U[i*m+j, r] * V[j'*p+k, r] * W[k_out*n+i_out, r]
    # where i,j is the A index, j',k is the B index, and i_out,k_out is the C index.
    # For this to be valid, we need j == j' (the inner dimension must match).
    
    # So a rank-1 term is: A[i,j] * B[j,k] * C[i,k] (with coefficients).
    
    # The Strassen algorithm doesn't directly give rank-1 terms; it gives a way to compute the
    # product using fewer multiplications. To convert to a tensor decomposition, we need to
    # express each "multiplication" in the Strassen algorithm as a combination of rank-1 terms.
    
    # Each Strassen multiplication is of the form (linear form in A) * (linear form in B),
    # and the result is a linear combination of C entries.
    
    # For the tensor decomposition, we can use the fact that:
    # (sum c_i A_i) * (sum d_j B_j) = sum_i sum_j c_i d_j A_i B_j
    # and each A_i B_j contributes to specific C entries.
    
    # But this would give a high rank. The whole point of Strassen is to REDUCE the rank.
    
    # I think the correct interpretation is:
    # The Strassen algorithm defines a tensor decomposition of rank 42 directly.
    # Each of the 7 Strassen terms corresponds to 6 rank-1 terms (since each block product
    # (1x2)*(2x3) has schoolbook rank 6).
    
    # Let me just implement it and verify numerically.
    
    # For each Strassen term, I'll define:
    # - The 2 A-indices for X (with coefficients)
    # - The 2 B-row-indices and 3 B-col-indices for Y (with coefficients)
    # - The 2 C-row-indices and 3 C-col-indices for the output (with coefficients)
    
    # Then for each combination, I'll add rank-1 terms.
    
    # Actually, let me just hardcode the 42 terms based on the Strassen formulas.
    
    # I'll use a systematic approach:
    # For each of the 7 Strassen terms, the block product (1x2)*(2x3) is decomposed into
    # 6 rank-1 terms using schoolbook. The X vector has 2 entries, each being a linear
    # combination of A entries. The Y matrix is 2x3, each entry a linear combination of B entries.
    
    # For term with X = [x0, x1] and Y = [[y00, y01, y02], [y10, y11, y12]]:
    # result[b] = x0*y0b + x1*y1b
    
    # This is a bilinear map. The rank-1 decomposition has 6 terms:
    # (x0, y00, out[0]), (x1, y10, out[0]),
    # (x0, y01, out[1]), (x1, y11, out[1]),
    # (x0, y02, out[2]), (x1, y12, out[2])
    
    # Where "out[b]" is the C entry that this contributes to.
    
    # But x0, x1 are linear forms in A, and y0b, y1b are linear forms in B.
    # So each "term" like (x0, y00, out[0]) is itself a product of two linear forms,
    # which needs to be expanded.
    
    # I think the cleanest way is to just verify numerically. Let me construct U, V, W
    # using the Strassen formulas and check if they work.
    
    # Given the complexity, let me just return the schoolbook decomposition for now
    # and add the Strassen optimization in a follow-up.
    
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
