"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact for the matrix multiplication tensor."""
    rank = len(V[0]) if V else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check by building the full tensor and comparing with schoolbook
    # But that's expensive. Instead, check a few random inputs.
    # Actually, for exactness, we need to verify the tensor identity.
    # Let's just check with random small inputs.
    np.random.seed(42)
    for _ in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            # U_col is (n*m), V_col is (m*p), W_col is (p*n)
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            A_r = u_col.reshape(n, m)
            B_r = v_col.reshape(m, p)
            C_r = w_col.reshape(p, n)
            # The contribution is outer product structure:
            # C += sum over r of (A_r * B_r) * C_r? No.
            # The tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # C_{i,k} = sum_j T_{i,j,k} = sum_j sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
            # = sum_r W_{k,i,r} * sum_j (U_{i,j,r} V_{j,k,r})
            # Let M_r = diag? No.
            # sum_j U_{i,j,r} V_{j,k,r} = (u_row_i @ v_col_k) where u_row_i is (m,) and v_col_k is (m,)?
            # u_col is length m. v_col is length p? No, V is (m*p, rank).
            # V[j*p+k, r]. So for fixed j,k, it's V[j*p+k, r].
            # Let's just compute directly for small sizes.
            pass
    # Simpler verification: build the multiplication tensor T of size (n*m*p)
    # and check if it can be expressed as sum of rank-1 tensors.
    # T[i, j, k] = delta_{i,k} ? No, T[i,j,k] = 1 if we are computing C[i,k] from A[i,j]B[j,k].
    # The tensor for matrix multiplication M_{n,m,p} has entries:
    # M[i, j, k] = 1 if we are multiplying the (i,j) entry of A and (j,k) entry of B to contribute to (i,k) of C.
    # Wait, the standard tensor T is defined such that C_{ik} = sum_{j} T_{ijk} A_{ij} B_{jk}.
    # So T_{ijk} = 1 if the indices match the multiplication pattern?
    # Actually, the tensor is T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # No, that's not right. The tensor M_{n,m,p} is such that for any A (n x m) and B (m x p),
    # the flattened C (p x n) is obtained by contracting the tensor with flattened A and B.
    # Specifically, if we flatten A to a_{i*m+j}, B to b_{j*p+k}, and C to c_{k*n+i},
    # then c_{k*n+i} = sum_{j} a_{i*m+j} b_{j*p+k}.
    # So the tensor T has indices (i, j, k) and entry T[i, j, k] = 1.
    # Wait, if T[i,j,k] = 1, then sum_{j} T[i,j,k] a_{i*m+j} b_{j*p+k} = sum_{j} a_{i*m+j} b_{j*p+k} = c_{k*n+i}.
    # Yes! The tensor is simply all ones? No, that can't be right for the decomposition.
    # The tensor is T[i, j, k] where i indexes rows of A/C, j indexes cols of A/rows of B, k indexes cols of B/C.
    # The entry is 1.
    # So we need to decompose the all-ones tensor of size (n, m, p) into sum of rank-1 tensors?
    # No, the "tensor" for matrix multiplication is not all ones.
    # The tensor M_{n,m,p} is defined by its action on vectors.
    # Let's stick to the definition:
    # We need U, V, W such that for all i, j, k:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is a valid multiplication term, 0 otherwise.
    # The valid terms are exactly all (i, j, k) with i in [0,n), j in [0,m), k in [0,p).
    # Because C[i,k] = sum_j A[i,j] B[j,k].
    # So the tensor T[i, j, k] is 1 for all valid indices.
    # Wait, if T[i,j,k] is 1 for all i,j,k, then the rank of the all-ones tensor is 1?
    # No, the tensor is not all ones. The indices are (i, j, k).
    # The value is 1.
    # But the decomposition is U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # For a fixed r, the tensor is U_r (n*m) outer V_r (m*p) outer W_r (p*n).
    # The indices are coupled: i is in U and W, j is in U and V, k is in V and W.
    # So it's not a simple outer product of 3 vectors. It's a 3-mode tensor.
    # Let's just verify by checking if C = A @ B can be computed.
    
    # Efficient check:
    # C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's compute this for random A, B and compare with A @ B.
    for _ in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_true = A @ B
        C_test = np.zeros((n, p))
        # We can rearrange the sum:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] V[j*p+k, r])
        # Let S_r[i, k] = sum_j U[i*m+j, r] V[j*p+k, r]
        # This is like a matrix multiplication for each r?
        # U_r is n x m. V_r is m x p.
        # S_r = U_r @ V_r. (n x p)
        # Then C[i,k] = sum_r S_r[i,k] * W[k*n+i, r]
        # Note: W is indexed by (k, i) -> W[k*n+i, r].
        # So W_r is p x n.
        # C = sum_r (U_r @ V_r) * W_r  (element-wise product, then sum)
        for r in range(rank):
            U_r = U[r].reshape(n, m) if U.ndim == 2 and U.shape[0] == rank else U[:, r].reshape(n, m)
            # Wait, U is (n*m, rank). So U[:, r] is (n*m,). Reshape to (n, m).
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            S_r = U_r @ V_r  # (n, p)
            # W_r is (p, n). We need to multiply element-wise with S_r (n, p).
            # S_r[i, k] * W_r[k, i].
            # So we need to transpose W_r to (n, p) and flip? No.
            # W_r[k, i] corresponds to S_r[i, k].
            # So W_r.T[i, k] = W_r[k, i].
            C_test += S_r * W_r.T
        if not np.allclose(C_test, C_true, atol=1e-9):
            return False
    return True

def _strassen_2x2(A, B):
    """
    Compute A @ B for 2x2 matrices using Strassen's algorithm.
    Returns C = A @ B.
    """
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    e, f = B[0, 0], B[0, 1]
    g, h = B[1, 0], B[1, 1]
    
    m1 = (a + d) * (e + h)
    m2 = (c + d) * e
    m3 = a * (f - h)
    m4 = d * (g - e)
    m5 = (a + b) * h
    m6 = (c - a) * (e + f)
    m7 = (b - d) * (g + h)
    
    C = np.array([
        [m1 + m4 - m5 + m7, m3 + m5],
        [m2 + m4, m1 - m2 + m3 + m6]
    ])
    return C

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # We will construct a decomposition using Strassen for 2x2 blocks.
    # The matrix multiplication is A (3x4) * B (4x6) = C (3x6).
    
    # Strategy:
    # Split A into A1 (2x4) and A2 (1x4).
    # Split B into B1 (4x4) and B2 (4x2).
    # C1 = A1 * B1 (2x4).
    # C2 = A1 * B2 (2x2).
    # C3 = A2 * B1 (1x4).
    # C4 = A2 * B2 (1x2).
    # C = [C1 C2; C3 C4] (if we view C as 2 blocks of rows? No, C is 3x6.
    # C[0:2, 0:4] = C1, C[0:2, 4:6] = C2
    # C[2:3, 0:4] = C3, C[2:3, 4:6] = C4
    
    # Let's decompose each part.
    # Part 1: C1 = A1 (2x4) * B1 (4x4).
    # Split A1 into A11 (2x2) and A12 (2x2).
    # Split B1 into B11 (2x2) and B12 (2x2).
    # C1 = A11 B11 + A12 B12. (2x2 + 2x2)
    # A11 B11 is 2x2x2. Use Strassen (rank 7).
    # A12 B12 is 2x2x2. Use Strassen (rank 7).
    # Total rank for C1: 14.
    
    # Part 2: C2 = A1 (2x4) * B2 (4x2).
    # Split A1 into A11 (2x2) and A12 (2x2).
    # Split B2 into B21 (2x2) and B22 (2x2).
    # C2 = A11 B21 + A12 B22.
    # Each is 2x2x2. Rank 7 each. Total 14.
    
    # Part 3: C3 = A2 (1x4) * B1 (4x4).
    # This is (1x4) * (4x4) = 1x4.
    # No 2x2 blocks possible for A2 (only 1 row).
    # Schoolbook: 1*4*4 = 16.
    # Can we do better?
    # R_{1,4,4} = 4? No, R_{1,4,4} is the rank of the tensor for 1x4x4.
    # The tensor is T[i, j, k] where i=0.
    # T[0, j, k] = 1.
    # This is a 1x4x4 tensor.
    # It can be decomposed into 4 rank-1 terms?
    # T[0, j, k] = sum_r U[r] V[r] W[r]?
    # No, the indices are (j, k).
    # C[0, k] = sum_j A[0, j] B[j, k].
    # This is just a dot product of A[0, :] and B[:, k].
    # The tensor has entries T[j, k] = 1.
    # The rank of the 4x4 all-ones matrix is 1.
    # So R_{1,4,4} = 4? No.
    # The tensor is 3D: (1, 4, 4).
    # Entry T[0, j, k] = 1.
    # We need U[0*m+j, r], V[j*p+k, r], W[k*n+0, r].
    # n=1, m=4, p=4.
    # U[j, r], V[j*4+k, r], W[k, r].
    # Sum_r U[j, r] V[j*4+k, r] W[k, r] = 1.
    # This is a 4x4 tensor T_{j,k} = 1.
    # This is a rank-1 tensor?
    # If we set U[j, 1] = 1, V[j*4+k, 1] = 1, W[k, 1] = 1, then sum is 1.
    # So rank is 1?
    # Wait, the indices of V are coupled to j and k.
    # V is a vector of length m*p = 16.
    # If we use rank 1, we have vectors u (length 4), v (length 16), w (length 4).
    # T[j, k] = u[j] * v[j*4+k] * w[k] = 1 for all j, k.
    # This requires u[j] != 0, w[k] != 0.
    # Let u[j] = 1, w[k] = 1. Then v[j*4+k] must be 1.
    # So v is all ones.
    # This works!
    # So R_{1,4,4} = 1?
    # Let's check the definition.
    # C[0, k] = sum_j A[0, j] B[j, k].
    # Using the decomposition:
    # C[0, k] = sum_r W[k*1+0, r] * (sum_j U[0*4+j, r] V[j*4+k, r])
    # For r=1: W[k, 1] * (sum_j U[j, 1] V[j*4+k, 1])
    # = 1 * (sum_j 1 * 1) = 4.
    # But we want C[0, k] = sum_j A[0, j] B[j, k].
    # The decomposition computes sum_j (U[j, r] A[0, j]) (V[j*4+k, r] B[j, k])?
    # No, the decomposition is for the *tensor*, not the computation directly.
    # The tensor T_{j,k} = 1 means that the term A[0, j] B[j, k] contributes 1 to C[0, k].
    # So the sum over j of T_{j,k} A[0, j] B[j, k] = sum_j A[0, j] B[j, k].
    # If we use the rank-1 decomposition of T:
    # T_{j,k} = U[j, 1] V[j*4+k, 1] W[k, 1].
    # Then the contribution is sum_j (U[j, 1] A[0, j]) (V[j*4+k, 1] B[j, k]) W[k, 1].
    # This is NOT sum_j A[0, j] B[j, k] unless U[j,1]=1, V[j*4+k,1]=1, W[k,1]=1.
    # In that case, the sum is sum_j (1 * A[0, j]) (1 * B[j, k]) * 1 = sum_j A[0, j] B[j, k].
    # Yes! It works.
    # So R_{1,4,4} = 1?
    # Wait, the "rank" of the tensor is the number of rank-1 tensors needed to sum to T.
    # T is the all-ones tensor of size (1, 4, 4).
    # Its rank is indeed 1, because it is a rank-1 tensor itself.
    # So C3 can be computed with rank 1?
    # But that seems too good to be true.
    # Let's re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix by a 4 by 6 matrix."
    # The tensor is M_{3,4,6}.
    # The rank is the minimum number of rank-1 tensors to sum to M_{3,4,6}.
    # For M_{1,4,4}, the tensor is T[i, j, k] = 1.
    # It is a rank-1 tensor. So R_{1,4,4} = 1.
    # Similarly, R_{1,4,2} = 1.
    
    # Let's re-evaluate the total rank.
    # C1 (2x4x4): R_{2,4,4} <= R_{2,2,2} + R_{2,2,2} = 7 + 7 = 14.
    # C2 (2x4x2): R_{2,4,2} <= R_{2,2,2} + R_{2,2,2} = 7 + 7 = 14.
    # C3 (1x4x4): R_{1,4,4} = 1.
    # C4 (1x4x2): R_{1,4,2} = 1.
    # Total rank: 14 + 14 + 1 + 1 = 30.
    
    # Wait, is R_{2,4,4} really 14?
    # M_{2,4,4} = M_{2,2,2} (for A[0:2, 0:2] B[0:2, 0:2]) + M_{2,2,2} (for A[0:2, 2:4] B[2:4, 0:4])?
    # No, B is 4x4.
    # A (2x4) * B (4x4) = C (2x4).
    # C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2:4] B[2:4, 0:2].
    # C[0:2, 2:4] = A[0:2, 0:2] B[0:2, 2:4] + A[0:2, 2:4] B[2:4, 2:4].
    # So it's 4 sub-multiplications of 2x2x2.
    # Each has rank 7.
    # Total 28.
    # But we can share terms?
    # No, the indices are disjoint.
    # So R_{2,4,4} <= 28.
    # Can we do better?
    # R_{2,4,4} is known to be 24? Or 28?
    # Let's assume 28 for now.
    
    # Let's re-calculate with 28.
    # C1: 28.
    # C2: R_{2,4,2}.
    # A (2x4) * B (4x2) = C (2x2).
    # C[0:2, 0:2] = A[0:2, 0:2] B[0:2, 0:2] + A[0:2, 2:4] B[2:4, 0:2].
    # Two 2x2x2 multiplications. Rank 7 each. Total 14.
    # C3: 1.
    # C4: 1.
    # Total: 28 + 14 + 1 + 1 = 44.
    
    # This is much better than 72.
    
    # Let's implement this.
    # We need to construct the U, V, W matrices.
    
    # Total rank R = 44.
    R = 44
    U = np.zeros((n * m, R))  # 12 x 44
    V = np.zeros((m * p, R))  # 24 x 44
    W = np.zeros((p * n, R))  # 18 x 44
    
    # Helper to add a 2x2x2 Strassen block.
    # The block computes C_block = A_block (2x2) * B_block (2x2).
    # It contributes to C at a specific offset.
    # A_block is A[i_off:i_off+2, j_off:j_off+2].
    # B_block is B[j_off:j_off+2, k_off:k_off+2].
    # C_block is C[i_off:i_off+2, k_off:k_off+2].
    
    def add_strassen_block(i_off, j_off, k_off, col_start):
        # Strassen's 7 multiplications.
        # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
        # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
        # m1 = (a+d)(e+h) -> U: (1,1), (1,1) for a,d. V: (1,1), (1,1) for e,h. W: (1,1), (1,1) for c[0,0], c[1,1].
        # Actually, we need to map to the global indices.
        
        # m1 = (a+d)(e+h)
        # U[i_off, j_off] += 1, U[i_off+1, j_off+1] += 1
        # V[j_off, k_off] += 1, V[j_off+1, k_off+1] += 1
        # W[k_off, i_off] += 1, W[k_off+1, i_off+1] += 1
        
        # m2 = (c+d)e
        # U[i_off+1, j_off] += 1, U[i_off+1, j_off+1] += 1
        # V[j_off, k_off] += 1
        # W[k_off, i_off+1] += 1
        
        # m3 = a(f-h)
        # U[i_off, j_off] += 1
        # V[j_off, k_off+1] += 1, V[j_off+1, k_off+1] += -1
        # W[k_off+1, i_off] += 1
        
        # m4 = d(g-e)
        # U[i_off+1, j_off+1] += 1
        # V[j_off+1, k_off] += 1, V[j_off, k_off] += -1
        # W[k_off, i_off+1] += 1
        
        # m5 = (a+b)h
        # U[i_off, j_off] += 1, U[i_off, j_off+1] += 1
        # V[j_off+1, k_off+1] += 1
        # W[k_off+1, i_off] += 1
        
        # m6 = (c-a)(e+f)
        # U[i_off+1, j_off] += 1, U[i_off, j_off] += -1
        # V[j_off, k_off] += 1, V[j_off, k_off+1] += 1
        # W[k_off, i_off+1] += 1, W[k_off+1, i_off+1] += 1
        
        # m7 = (b-d)(g+h)
        # U[i_off, j_off+1] += 1, U[i_off+1, j_off+1] += -1
        # V[j_off+1, k_off] += 1, V[j_off+1, k_off+1] += 1
        # W[k_off, i_off] += 1, W[k_off+1, i_off] += 1
        
        # Let's write this more carefully.
        # U indices: (i, j) -> U[i*m + j, col]
        # V indices: (j, k) -> V[j*p + k, col]
        # W indices: (k, i) -> W[k*n + i, col]
        
        # m1
        U[i_off * m + j_off, col_start] += 1
        U[(i_off + 1) * m + (j_off + 1), col_start] += 1
        V[j_off * p + k_off, col_start] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start] += 1
        W[k_off * n + i_off, col_start] += 1
        W[(k_off + 1) * n + (i_off + 1), col_start] += 1
        
        # m2
        U[(i_off + 1) * m + j_off, col_start + 1] += 1
        U[(i_off + 1) * m + (j_off + 1), col_start + 1] += 1
        V[j_off * p + k_off, col_start + 1] += 1
        W[k_off * n + (i_off + 1), col_start + 1] += 1
        
        # m3
        U[i_off * m + j_off, col_start + 2] += 1
        V[j_off * p + (k_off + 1), col_start + 2] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start + 2] -= 1
        W[(k_off + 1) * n + i_off, col_start + 2] += 1
        
        # m4
        U[(i_off + 1) * m + (j_off + 1), col_start + 3] += 1
        V[(j_off + 1) * p + k_off, col_start + 3] += 1
        V[j_off * p + k_off, col_start + 3] -= 1
        W[k_off * n + (i_off + 1), col_start + 3] += 1
        
        # m5
        U[i_off * m + j_off, col_start + 4] += 1
        U[i_off * m + (j_off + 1), col_start + 4] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start + 4] += 1
        W[(k_off + 1) * n + i_off, col_start + 4] += 1
        
        # m6
        U[(i_off + 1) * m + j_off, col_start + 5] += 1
        U[i_off * m + j_off, col_start + 5] -= 1
        V[j_off * p + k_off, col_start + 5] += 1
        V[j_off * p + (k_off + 1), col_start + 5] += 1
        W[k_off * n + (i_off + 1), col_start + 5] += 1
        W[(k_off + 1) * n + (i_off + 1), col_start + 5] += 1
        
        # m7
        U[i_off * m + (j_off + 1), col_start + 6] += 1
        U[(i_off + 1) * m + (j_off + 1), col_start + 6] -= 1
        V[(j_off + 1) * p + k_off, col_start + 6] += 1
        V[(j_off + 1) * p + (k_off + 1), col_start + 6] += 1
        W[k_off * n + i_off, col_start + 6] += 1
        W[(k_off + 1) * n + i_off, col_start + 6] += 1
        
        return col_start + 7

    # Add blocks for C1 (2x4x4)
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 0, 0, 0)
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 0, 2, col)
    # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen_block(0, 2, 0, col)
    # A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_strassen_block(0, 2, 2, col)
    
    # Add blocks for C2 (2x4x2)
    # A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    col = add_strassen_block(0, 0, 4, col)
    # A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    col = add_strassen_block(0, 2, 4, col)
    
    # Add rank-1 terms for C3 (1x4x4)
    # A[2, 0:4] * B[0:4, 0:4] -> C[2, 0:4]
    # T[i=2, j, k] = 1.
    # U[2*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + 2, col] = 1
    for j in range(4):
        for k in range(4):
            U[2 * m + j, col] += 1
            V[j * p + k, col] += 1
            W[k * n + 2, col] += 1
    # This is not rank 1. This is rank 16 (schoolbook).
    # To get rank 1, we need:
    # U[2*m + j, col] = u_j
    # V[j*p + k, col] = v_{j,k}
    # W[k*n + 2, col] = w_k
    # Such that sum_j u_j v_{j,k} w_k = 1 for all k?
    # No, the sum is over j.
    # C[2, k] = sum_j U[2*m+j, r] V[j*p+k, r] W[k*n+2, r]
    # We want this to be sum_j A[2, j] B[j, k].
    # So we need the tensor T[2, j, k] = 1 to be decomposed.
    # T[2, j, k] = U[2*m+j, r] V[j*p+k, r] W[k*n+2, r]
    # For rank 1, we need U[2*m+j, 1] V[j*p+k, 1] W[k*n+2, 1] = 1.
    # This implies U[2*m+j, 1] = 1, V[j*p+k, 1] = 1, W[k*n+2, 1] = 1.
    # Let's try that.
    col -= 16 # Reset col to before the loop
    U[2 * m + 0:2 * m + 4, col] = 1
    V[0:4 * p, col] = 1 # j in 0..3, k in 0..3 -> j*p+k in 0..15.
    W[0:4, col] = 1 # k in 0..3, i=2 -> k*3+2 in 2,5,8,11.
    # Wait, W indices are k*n+i. n=3, i=2.
    # k=0 -> 2, k=1 -> 5, k=2 -> 8, k=3 -> 11.
    W[2, col] = 1
    W[5, col] = 1
    W[8, col] = 1
    W[11, col] = 1
    # V indices are j*p+k. p=6.
    # j=0, k=0..3 -> 0,1,2,3
    # j=1, k=0..3 -> 6,7,8,9
    # j=2, k=0..3 -> 12,13,14,15
    # j=3, k=0..3 -> 18,19,20,21
    # So V[0:4, col] = 1, V[6:10, col] = 1, V[12:16, col] = 1, V[18:22, col] = 1.
    V[0:4, col] = 1
    V[6:10, col] = 1
    V[12:16, col] = 1
    V[18:22, col] = 1
    col += 1
    
    # Add rank-1 term for C4 (1x4x2)
    # A[2, 0:4] * B[0:4, 4:6] -> C[2, 4:6]
    # T[2, j, k] = 1 for k in 4,5.
    # U[2*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+2, col] = 1
    U[2 * m + 0:2 * m + 4, col] = 1
    # j=0, k=4,5 -> 24,25
    # j=1, k=4,5 -> 30,31
    # j=2, k=4,5 -> 36,37
    # j=3, k=4,5 -> 42,43
    V[24:26, col] = 1
    V[30:32, col] = 1
    V[36:38, col] = 1
    V[42:44, col] = 1
    # k=4 -> 14, k=5 -> 17
    W[14, col] = 1
    W[17, col] = 1
    col += 1
    
    # Total rank used: 28 (C1) + 14 (C2) + 1 (C3) + 1 (C4) = 44.
    # col should be 44.
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school))
        V_school = np.zeros((m * p, rank_school))
        W_school = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j, c] = 1.0
                    V_school[j * p + k, c] = 1.0
                    W_school[k * n + i, c] = 1.0
                    c += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
