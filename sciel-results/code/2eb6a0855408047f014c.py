"""Matrix multiplication 2x2 by 2x6 — rank-minimal exact decomposition."""
import time

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 6
    rank = 15
    # U: (n*m, rank) = (4, 15)
    # V: (m*p, rank) = (12, 15)
    # W: (p*n, rank) = (12, 15)

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # We decompose C = A * B where A is 2x2, B is 2x6.
    # C is 2x6.
    #
    # Let's think of B as two 2x3 blocks: B_top (rows 0, cols 0-5) and B_bot (row 1, cols 0-5).
    # Actually B is 2x6: row 0 has 6 elements, row 1 has 6 elements.
    #
    # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:]
    # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:]
    #
    # We can use Strassen-like decomposition for the 2x2 matrix multiplication,
    # but since B has 6 columns, each "product" is a row vector of length 6.
    #
    # Standard Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # For our case, B00, B01, B10, B11 are each 1x3 vectors (since B is 2x6,
    # split into 2x3 blocks: B00 = B[0,0:3], B01 = B[0,3:6], B10 = B[1,0:3], B11 = B[1,3:6])
    #
    # Wait, let me reconsider the indexing. B is 2x6.
    # B[j, k] for j in {0,1}, k in {0,...,5}.
    # 
    # Let's split B into two 2x3 matrices: B_left = B[:, 0:3], B_right = B[:, 3:6].
    # Then C[:, 0:3] = A * B_left and C[:, 3:6] = A * B_right.
    #
    # So we need to compute two 2x2 * 2x3 multiplications.
    # For each 2x2 * 2x3 multiplication, we can use a rank-7 decomposition (Strassen).
    #
    # Actually, let me think more carefully. The tensor is for multiplying a 2x2 matrix
    # by a 2x6 matrix. The "tensor" T has indices (i, j, k) where i is the output row,
    # j is the input matrix element index, k is the B element index.
    #
    # Actually the standard formulation: C[i,k] = sum_j A[i,j] * B[j,k].
    # Here i in {0,1}, j in {0,1}, k in {0,...,5}.
    #
    # The tensor is of format (2, 2, 6) — we want to decompose it into rank-r terms.
    #
    # Let me just use a direct approach. For the 2x2 * 2xN case, the minimal rank
    # is known. For 2x2 * 2x6, we can think of it as 3 copies of 2x2 * 2x2 (each of
    # which has rank 7), giving rank 21. But we can do better by sharing structure.
    #
    # Actually, let's think of it differently. The multiplication A*B where A is 2x2
    # and B is 2x6 can be seen as: for each column k of B, we multiply A by the 2x1
    # column B[:,k]. That's 6 independent 2x2*2x1 multiplications, each of rank 3.
    # Total rank = 18.
    #
    # But we can do better. Let's use the fact that we can share some products.
    #
    # For a 2x2 matrix A = [[a, b], [c, d]] and a 2x6 matrix B with rows r0 = [b0,...,b5]
    # and r1 = [c0,...,c5], the product is:
    # Row 0 of C: [a*b0 + b*c0, a*b1 + b*c1, ..., a*b5 + b*c5]
    # Row 1 of C: [c*b0 + d*c0, c*b1 + d*c1, ..., c*b5 + d*c5]
    #
    # Let's define:
    # p1 = a + b  (scalar)
    # p2 = a - b  (scalar)
    # p3 = c + d  (scalar)
    # p4 = c - d  (scalar)
    #
    # Hmm, let me just try a rank-12 construction.
    #
    # Actually, the known result for 2x2 * 2xn matrix multiplication:
    # The tensor rank of [2,2,n] is 2n for n >= 1? No, that's the bilinear complexity.
    #
    # Let me just construct a valid decomposition. I'll use rank 12.
    #
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # Let me define 12 rank-1 tensors:
    # For k=0..5, we need terms that produce C[0,k] and C[1,k].
    #
    # Term 2k:   A[0,0] * B[0,k] contributes to C[0,k]
    #   U: index (0*2+0)=0 gets 1, others 0
    #   V: index (0*6+k)=k gets 1, others 0
    #   W: index (k*2+0)=2k gets 1, others 0
    #
    # Term 2k+1: A[0,1] * B[1,k] contributes to C[0,k]
    #   U: index (0*2+1)=1 gets 1
    #   V: index (1*6+k)=6+k gets 1
    #   W: index 2k gets 1
    #
    # Term 2k+2... wait, I'm using too many. Let me re-index.
    #
    # With rank 12, I have 12 terms. Let me assign:
    # Terms 0-5: for k=0..5, A[0,0]*B[0,k] -> C[0,k]
    # Terms 6-11: for k=0..5, A[1,0]*B[0,k] -> C[1,k]
    # But I'm missing A[0,1]*B[1,k] and A[1,1]*B[1,k].
    #
    # I need rank 12 minimum: 4 * 3 = 12.
    # Terms: for each (i,j) pair in A and each k, we have A[i,j]*B[j,k].
    # That's 2*2*6 = 24 terms (schoolbook).
    #
    # To reduce, we can combine. For fixed k:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This is a 2x2 * 2x1 multiplication, which has rank 3.
    # So for 6 values of k, we get rank 18.
    #
    # But wait — can we share across k? The A factors are the same for all k.
    # Let me think of it as: we have 6 copies of the 2x2*2x1 tensor, all sharing
    # the same A-factor indices but different B-factor indices.
    #
    # Actually, the tensor [2,2,6] can be seen as [2,2,2] ⊗ I_3 (three copies).
    # The rank of [2,2,2] is 7 (Strassen). So rank of [2,2,6] <= 7*3 = 21.
    # But this is not tight.
    #
    # Let me try a different approach. Let me just use rank 12 with a clever construction.
    #
    # Define for r = 0..11:
    # We'll use the following 12 terms:
    #
    # Let me index A elements as A00, A01, A10, A11 (row-major, index = i*2+j).
    # Let me index B elements as B[j,k] (row-major, index = j*6+k).
    # Let me index C elements as C[i,k] (row-major, index = i*6+k). Wait, the task says
    # W[k*n+i, r] so C index = k*2 + i.
    #
    # U index = i*m + j = i*2 + j (for A[i,j])
    # V index = j*p + k = j*6 + k (for B[j,k])
    # W index = k*n + i = k*2 + i (for C[i,k])
    #
    # So U has 4 rows, V has 12 rows, W has 12 rows.
    #
    # Let me try rank 12:
    # For each k in 0..5, I'll use 2 terms:
    # Term 2k: (A00+A01) * B[0,k] * ... no, this doesn't work directly.
    #
    # OK let me just go with rank 18 (6 copies of rank-3 2x2*2x1):
    rank = 18
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(12)]
    W = [[0.0] * rank for _ in range(12)]
    
    col = 0
    for k in range(6):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0][col] = 1.0
        V[k][col] = 1.0       # B[0,k] has index 0*6+k = k
        W[k*2+0][col] = 1.0   # C[0,k] has index k*2+0
        col += 1
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[1][col] = 1.0
        V[6+k][col] = 1.0     # B[1,k] has index 1*6+k = 6+k
        W[k*2+0][col] = 1.0
        col += 1
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[2][col] = 1.0
        V[k][col] = 1.0
        W[k*2+1][col] = 1.0   # C[1,k] has index k*2+1
        col += 1
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[3][col] = 1.0
        V[6+k][col] = 1.0
        W[k*2+1][col] = 1.0
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
