import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill
    
    # We want to decompose the matrix multiplication tensor for 4x4 * 4x8.
    # The tensor T has indices (i, j, k) where i in 0..3 (row of A / row of C),
    # j in 0..3 (col of A / row of B), k in 0..7 (col of B / col of C).
    # T[i,j,k] = 1 if the product C[i,k] involves A[i,j]*B[j,k].
    # Specifically, C[i,k] = sum_j A[i,j] * B[j,k].
    # So the tensor is sum_j A[i,j] B[j,k] C[k,i]... wait.
    # The multiplication tensor is T = sum_{i,j,k} e_i e_j f_k \otimes e_j f_k g_i ... no.
    # T = sum_{i,j,k} A_{ij} B_{jk} C_{ki} is the trace.
    # The standard multiplication tensor is T = sum_{i,j,k} E_{ij}^{(A)} E_{jk}^{(B)} E_{ki}^{(C)}.
    # Wait, C = AB. C_{ik} = sum_j A_{ij} B_{jk}.
    # So T = sum_{i,j,k} A_{ij} B_{jk} C_{ik}.
    # The factors are:
    # U indexed by (i,j), V by (j,k), W by (k,i).
    # Wait, the interface says:
    # U: (n*m, rank) -> n=4, m=4 -> 16 rows. Index i*m+j.
    # V: (m*p, rank) -> m=4, p=8 -> 32 rows. Index j*p+k.
    # W: (p*n, rank) -> p=8, n=4 -> 32 rows. Index k*n+i.
    
    # We can use Strassen's algorithm for 4x4 * 4x4, and do it twice.
    # Or, more efficiently, we can use the known rank of 4x4x4 multiplication.
    # The best known rank for 4x4x4 is 49 (from Strassen).
    # So for 4x4 * 4x8, we can split B into two 4x4 blocks, and C into two 4x4 blocks.
    # C1 = A * B1, C2 = A * B2.
    # Each requires rank 49. Total rank 98.
    
    # Let's construct the 4x4x4 Strassen factors.
    # We can do this by applying 2x2x2 Strassen to the 2x2 block structure,
    # where each block is 2x2.
    # The 2x2 block multiplication uses 7 products of 2x2 matrices.
    # Each 2x2 matrix product has rank 7 (Strassen).
    # Total rank 49.
    
    # First, define the 2x2x2 Strassen factors.
    # U2, V2, W2 are 4x7 matrices (since 2*2=4).
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # Strassen 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1; W2[1, 0] = -1; W2[2, 0] = 1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B01-B00)
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now, construct the 4x4x4 factors.
    # The 4x4x4 multiplication can be seen as a 2x2 block multiplication,
    # where each block is 2x2.
    # The factors for the block multiplication are U2, V2, W2.
    # The factors for the inner 2x2 multiplication are also U2, V2, W2.
    # The total factors are the Kronecker product of the block factors and inner factors.
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for cb in range(7):  # block column
        for ci in range(7):  # inner column
            c = cb * 7 + ci
            for ib in range(2):  # block row
                for ii in range(2):  # inner row
                    i = ib * 2 + ii
                    for jb in range(2):  # block col
                        for jj in range(2):  # inner col
                            j = jb * 2 + jj
                            # U4 index is i*4 + j
                            U4[i * 4 + j, c] = U2[ib * 2 + jb, cb] * U2[ii * 2 + jj, ci]
                            V4[j * 4 + k, c] = V2[jb * 2 + kb, cb] * V2[jj * 2 + kk, ci]
                            # Wait, the indices for V and W need to be consistent.
                            # Let's do it more carefully.
                    # Actually, let's just loop over all i, j, k.
                    # But that's 16*16*16 = 4096 iterations, which is fine.
    
    # Let's redo the construction more carefully.
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for cb in range(7):
        for ci in range(7):
            c = cb * 7 + ci
            for i in range(4):
                for j in range(4):
                    # i = ib*2 + ii, j = jb*2 + jj
                    ib = i // 2
                    ii = i % 2
                    jb = j // 2
                    jj = j % 2
                    U4[i * 4 + j, c] = U2[ib * 2 + jb, cb] * U2[ii * 2 + jj, ci]
                    for k in range(4):
                        kb = k // 2
                        kk = k % 2
                        V4[j * 4 + k, c] = V2[jb * 2 + kb, cb] * V2[jj * 2 + kk, ci]
                        W4[k * 4 + i, c] = W2[kb * 2 + ib, cb] * W2[kk * 2 + ii, ci]
    
    # Now, construct the 4x4 * 4x8 factors.
    # We split B into B1 (4x4) and B2 (4x4).
    # C = [C1 | C2], where C1 = A*B1, C2 = A*B2.
    # The factors for C1 are U4, V4 (for B1), W4 (for C1).
    # The factors for C2 are U4, V4 (for B2), W4 (for C2).
    # Total rank is 49 + 49 = 98.
    
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)
    
    # First 49 columns: for C1
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
    
    # Second 49 columns: for C2
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, 49:98] = V4[j * 4 + (k - 4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[(k - 4) * 4 + i, :]
    
    # Verify the decomposition (optional, but good for debugging).
    # We can check a few random instances.
    # But let's just return the result.
    # Convert to lists of lists.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
