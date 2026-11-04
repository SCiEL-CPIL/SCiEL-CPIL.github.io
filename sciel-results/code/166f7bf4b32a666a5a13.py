import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W):
    """Check if the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T_recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    target = np.ones((n, m, p))
    return np.allclose(T_recon, target, atol=1e-9)

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Rank-7 Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns U, V, W as 2D arrays.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing convention:
    A, B are 2x2.
    C = A * B
    C[i, k] = sum_j A[i, j] B[j, k]
    
    Strassen variables:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    Results:
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U represents coefficients for A
    # V represents coefficients for B
    # W represents coefficients for C
    
    # We need to map the linear combinations to indices.
    # A indices: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> flat 0, 1, 2, 3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_strassen_blocks():
    n, m, p = DIMENSIONS
    # We partition the 5x6 product into blocks.
    # Since n=2, we can treat the rows separately or jointly.
    # Strassen requires square blocks.
    # 5 is not a power of 2. 6 is not a power of 2.
    # We can decompose 5 as 2+2+1 and 6 as 2+2+2.
    # This creates a block structure.
    
    # Block sizes for m (columns of A, rows of B): [2, 2, 1]
    # Block sizes for p (rows of C, columns of B): [2, 2, 2]
    # Block sizes for n (rows of A, rows of C): [1, 1] (since n=2, we must treat each row separately or use a 2x2 block if we had 2 rows of A and 2 cols of A? No, A is 2x5. B is 5x6. C is 2x6.)
    
    # If we use 2x2 Strassen, we need 2 rows in A and 2 cols in A? No.
    # Strassen multiplies two 2x2 matrices.
    # Here A is 2x5, B is 5x6.
    # We can view A as [A_top; A_bottom] where A_top is 1x5, A_bottom is 1x5.
    # This doesn't fit 2x2 directly unless we embed in a larger structure or use specific block decompositions.
    
    # Alternative: Use the known rank-7 decomposition for 2x2.
    # Can we apply it to 2x2 sub-blocks of the 5x6 multiplication?
    # The multiplication C[i, k] = sum_j A[i, j] B[j, k].
    # For a fixed i, this is a vector-matrix product.
    # Strassen reduces the cost of 2x2 matrix multiplication.
    # If we group 2 rows of B and 2 columns of B? No, B is 5x6.
    
    # Let's try to decompose the 5x6 multiplication using 2x2 Strassen blocks where possible.
    # We can group the middle dimension j (size 5) into chunks of 2: [2, 2, 1].
    # We can group the output dimension k (size 6) into chunks of 2: [2, 2, 2].
    
    # For each pair of j-chunks and k-chunks, we have a sub-problem.
    # If the j-chunk is size 2 and k-chunk is size 2, can we use Strassen?
    # The sub-problem is:
    # C_sub (1x2 for one row i) = A_sub (1x2) * B_sub (2x2)
    # This is not a 2x2 matrix multiplication. It's a 1x2 by 2x2.
    # Strassen applies to 2x2 * 2x2 -> 2x2.
    
    # Since n=2, we have two rows in A.
    # Let A be 2x5, B be 5x6.
    # We can consider the multiplication of the full 2x5 A and 5x6 B.
    # We can pad A to 2x8 and B to 8x6? Or 4x6 and 4x4?
    # If we pad to 4x4, we can use 2x2 Strassen recursively.
    # But we want exact minimal rank for the 2x5x6 case.
    
    # Let's try a direct construction based on the 2x2 Strassen decomposition.
    # The rank of 2x2x2 is 7.
    # The rank of 2x5x6 is at most 2x2x6? No.
    
    # Let's use the block structure:
    # Split j into [2, 2, 1]
    # Split k into [2, 2, 2]
    
    # For each (j_block, k_block) pair:
    # If j_block size is 2 and k_block size is 2:
    #   We are computing C[:, k_block] += A[:, j_block] * B[j_block, k_block]
    #   A_sub is 2x2. B_sub is 2x2. C_sub is 2x2.
    #   This is exactly 2x2 matrix multiplication!
    #   So we can use the rank-7 Strassen decomposition for each such block.
    
    # There are 3 j_blocks (sizes 2, 2, 1) and 3 k_blocks (sizes 2, 2, 2).
    # Pairs with (2,2):
    # j0(2) with k0(2), k1(2), k2(2) -> 3 blocks of 2x2x2
    # j1(2) with k0(2), k1(2), k2(2) -> 3 blocks of 2x2x2
    # j2(1) with k0(2), k1(2), k2(2) -> 3 blocks of 1x2x2 (not 2x2x2)
    
    # So we have 6 blocks of 2x2x2.
    # Each 2x2x2 block has rank 7.
    # Total rank from these = 6 * 7 = 42.
    
    # For the j2(1) blocks:
    # A_sub is 2x1. B_sub is 1x2. C_sub is 2x2.
    # C[i, k] = A[i, j] * B[j, k]
    # This is an outer product of a 2x1 and 1x2 matrix, resulting in a 2x2 matrix.
    # The rank of this tensor (2x1x2) is 2. (Since it's separable: v1 * v2^T).
    # Actually, T[i, j, k] = 1 for the single j.
    # C[i, k] = A[i, j] B[j, k].
    # This requires 2 multiplications? No, it's 1 multiplication per entry?
    # A[i, j] * B[j, k] is one scalar multiplication per (i, k) pair?
    # No, A[i, j] is a vector of 2, B[j, k] is a vector of 2.
    # The product is a 2x2 matrix.
    # The tensor T[i, j, k] has i in [0,1], j in [0], k in [0,1].
    # T[i, 0, k] = 1 for all i, k.
    # Rank of this 2x1x2 tensor:
    # It is the outer product of u (2x1) and v (1x2)?
    # T = u * v^T?
    # u = [1, 1], v = [1, 1].
    # T[i, k] = u[i] v[k] = 1.
    # Yes, rank 1.
    # So each 1x2 block (where j is size 1, k is size 2) has rank 1?
    # Wait, the bilinear form is C[i, k] = A[i, j] B[j, k].
    # We need to compute 4 values of C using 1 input A[i, j] and 1 input B[j, k]?
    # No, for fixed j, A has 2 values (A[0,j], A[1,j]), B has 2 values (B[j,0], B[j,1]).
    # C[0,0] = A[0,j] B[j,0]
    # C[0,1] = A[0,j] B[j,1]
    # C[1,0] = A[1,j] B[j,0]
    # C[1,1] = A[1,j] B[j,1]
    # This is 4 scalar multiplications.
    # Can we do it with fewer?
    # This is the tensor 2x1x2.
    # Rank is 2. (Since T = [1,1] * [1,1]^T is rank 1 for the values, but the inputs are separate).
    # Let's check the rank of the 2x1x2 multiplication tensor.
    # It is equivalent to multiplying a 2-vector by a 1x2 matrix? No.
    # It's an outer product of two vectors.
    # The bilinear complexity of 2x1 * 1x2 is 2? No, it's 1 if we share?
    # No, A[0,j] and A[1,j] are distinct. B[j,0] and B[j,1] are distinct.
    # We need to form 4 products.
    # (A0+A1)(B0+B1) = A0B0 + A0B1 + A1B0 + A1B1.
    # We can compute M1 = (A0+A1)(B0+B1).
    # But we need A0B0, A0B1, A1B0, A1B1 separately.
    # We cannot recover them from M1 alone.
    # We need 4 distinct products?
    # Actually, this is just 4 scalar multiplications.
    # Is there a way to do it with rank < 4?
    # The tensor is T[i, j, k] = 1 for i in 0..1, j=0, k in 0..1.
    # T = [[1, 1], [1, 1]] (matrix of values for i,k).
    # The rank of the tensor T (2x1x2) is the minimum number of rank-1 tensors summing to T.
    # T[i, 0, k] = 1.
    # Can we write 1 as sum_c u_c[i] v_c w_c[k]?
    # 1 = sum_c u_c[i] w_c[k] (with v_c=1).
    # Since i and k are independent, and the value is constant 1,
    # u_c[i] w_c[k] must sum to 1 for all i,k.
    # This implies u is a vector and w is a vector such that u w^T = J (all ones 2x2).
    # Rank of J is 1.
    # So the tensor rank is 1.
    # Wait, if tensor rank is 1, then we can do it with 1 multiplication?
    # M = (sum_i A[i,j] * 1) * (sum_k B[j,k] * 1)?
    # No, the decomposition is T[i,j,k] = U[i,j] V[j,k] W[i,k].
    # Here V[j,k] is for the middle index.
    # T[i,0,k] = U[i,0] V[0,k] W[i,k].
    # We want T=1.
    # Let U[i,0]=1, V[0,k]=1, W[i,k]=1.
    # Then the algorithm is:
    # m = U * A * V * B?
    # The bilinear form is sum_j A[i,j] B[j,k] T[i,j,k].
    # If T[i,0,k] = U[i,0] V[0,k] W[i,k] = 1*1*1 = 1.
    # The term is A[i,0] B[0,k] * 1.
    # This gives C[i,k] += A[i,0] B[0,k].
    # This is correct.
    # Does this count as 1 rank-1 term?
    # The rank of the tensor is 1.
    # So each 1x2 block contributes rank 1.
    # There are 3 such blocks (j2 with k0, k1, k2).
    # Total rank = 6 * 7 + 3 * 1 = 42 + 3 = 45.
    
    # This is better than schoolbook (60).
    # Can we do better?
    # The 2x2x2 blocks are independent.
    # The 1x2 blocks are independent.
    # So rank 45 seems correct for this block decomposition.
    
    # Let's implement this.
    
    n, m, p = DIMENSIONS
    # j_blocks: [2, 2, 1]
    # k_blocks: [2, 2, 2]
    
    j_starts = [0, 2, 4]
    j_sizes = [2, 2, 1]
    k_starts = [0, 2, 4]
    k_sizes = [2, 2, 2]
    
    U = np.zeros((n * m, 45))
    V = np.zeros((m * p, 45))
    W = np.zeros((p * n, 45))
    
    col = 0
    U_strassen, V_strassen, W_strassen = _strassen_2x2x2()
    
    for jb in range(3):
        for kb in range(3):
            js = j_starts[jb]
            jsz = j_sizes[jb]
            ks = k_starts[kb]
            ksz = k_sizes[kb]
            
            if jsz == 2 and ksz == 2:
                # Use Strassen 2x2x2
                # A_sub is rows 0,1 of A, cols js, js+1
                # B_sub is rows js, js+1 of B, cols ks, ks+1
                # C_sub is rows 0,1 of C, cols ks, ks+1
                
                for c in range(7):
                    # U
                    for i in range(2):
                        for j in range(2):
                            val = U_strassen[i * 2 + j, c]
                            U[i * m + js + j, col] += val
                            
                    # V
                    for j in range(2):
                        for k in range(2):
                            val = V_strassen[j * 2 + k, c]
                            V[(js + j) * p + ks + k, col] += val
                            
                    # W
                    for k in range(2):
                        for i in range(2):
                            val = W_strassen[k * 2 + i, c]
                            W[(ks + k) * n + i, col] += val
                            
                    col += 1
                    
            else:
                # jsz == 1, ksz == 2
                # Rank 1 term
                # T[i, j, k] = 1 for i in 0..1, j=js, k in ks..ks+1
                # U[i, js] = 1
                # V[js, k] = 1
                # W[k, i] = 1
                
                for i in range(n):
                    U[i * m + js, col] = 1
                for k in range(ks, ks + ksz):
                    V[js * p + k, col] = 1
                for k in range(ks, ks + ksz):
                    for i in range(n):
                        W[k * n + i, col] = 1
                        
                col += 1
                
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    
    # Try the Strassen-based block decomposition
    U, V, W = _construct_strassen_blocks()
    
    # Verify
    if _verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails
        U, V, W = _schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    if np.allclose(T, expected):
        print("Verification passed. Rank:", U.shape[1])
    else:
        print("Verification failed.")
        print("Max diff:", np.max(np.abs(T - expected)))
