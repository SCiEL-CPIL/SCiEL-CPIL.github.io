"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
n, m, p = DIMENSIONS

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, tol=1e-9):
    """Check that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    # Build the multiplication tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For a 2x4 * 4x4 multiplication:
    # A is 2x4, B is 4x4, C = A*B is 2x4
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T[i, j, k] should be 1 if this is the multiplication tensor
    # Actually, the tensor for matrix multiplication M_{n,m,p} has:
    # T[i, j, k] = 1 for all i,j,k (it's the standard multiplication tensor)
    # The decomposition says: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where U is indexed by (i,j) with i in [0,n), j in [0,m)
    # V is indexed by (j,k) with j in [0,m), k in [0,p)
    # W is indexed by (k,i) with k in [0,p), i in [0,n)
    
    # Let's verify by checking all (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, tol=1e-9):
    """Faster verification using numpy."""
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    
    # For each (i, j, k): sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # We can do this as: for each c, the outer product contributes
    # T[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    
    # Build T as zeros
    T = np.zeros((n, m, p))
    for c in range(rank):
        # Ur[:, :, c] is (n, m), Vr[:, :, c] is (m, p), Wr[:, :, c] is (p, n)
        # We need to multiply element-wise in a specific pattern
        # T[i,j,k] += Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # This is a sum over the shared indices
        # Let's do it as: for each i, j: T[i,j,:] += Ur[i,j,c] * Vr[j,:,c] * Wr[:,i,c]
        for i in range(n):
            for j in range(m):
                T[i, j, :] += Ur[i, j, c] * Vr[j, :, c] * Wr[:, i, c]
    
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True


def _try_strassen_variant_2x4x4():
    """
    Try to find a low-rank decomposition for 2x4 * 4x4.
    
    The schoolbook rank is 2*4*4 = 32.
    
    We can think of this as:
    A (2x4) * B (4x4) = C (2x4)
    
    One approach: use Strassen-like ideas.
    
    Let's try to find a decomposition with rank < 32.
    
    Key insight: We can split B into blocks.
    B = [B1 B2] where B1, B2 are 4x2.
    Then C = A * B = [A*B1, A*B2].
    
    Each A*Bi is 2x4 * 4x2 = 2x2.
    For 2x2 * 2x2, Strassen gives rank 7 (but that's for square).
    For 2x4 * 4x2, the schoolbook is 2*4*2 = 16.
    
    Actually, let's think differently.
    
    A is 2x4, B is 4x4.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    We can write this as:
    C = A * B
    
    Let's try to use the fact that we can factor through smaller matrices.
    
    One known result: the rank of the 2x4 * 4x4 tensor is at most 24 (by some bounds).
    Let me try to construct a rank-24 or lower decomposition.
    
    Actually, let me try a direct approach: use bilinear algorithms.
    
    For C[i,k] = sum_j A[i,j] B[j,k], we have 8 output entries (2*4=8).
    
    Let's try rank 24 first, then try to reduce.
    """
    # Let's try to construct a decomposition using block structure.
    # Split A into two 2x2 blocks: A = [A1 A2] where A1, A2 are 2x2.
    # Split B into four 2x2 blocks: B = [[B11 B12], [B21 B22]]
    # Then C = [A1 B11 + A2 B21, A1 B12 + A2 B22]
    # C is 2x4, so C = [C1 C2] where C1, C2 are 2x2.
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # Each term like A1 B11 is a 2x2 * 2x2 multiplication, which has rank 7 (Strassen).
    # But we have 4 such terms. If we could share some computations...
    
    # Actually, let's just try to find a low-rank decomposition numerically.
    # For now, return the schoolbook and try to improve.
    pass


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    deadline = time.time() + 140  # Leave 10s margin
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition
    # For 2x4 * 4x4, let's try various ranks
    
    # The multiplication tensor M_{2,4,4} has dimension 2*4 + 4*4 + 4*2 = 8 + 16 + 8 = 32
    # Wait, the tensor lives in A* ⊗ B* ⊗ C* where dim A* = 2*4=8, dim B* = 4*4=16, dim C* = 2*4=8
    # So the tensor is in R^8 ⊗ R^16 ⊗ R^8
    
    # The schoolbook rank is 32.
    # Let's try to find a decomposition with rank < 32.
    
    # One approach: use the fact that we can combine some terms.
    # For example, if we have terms that differ only in one index, we might be able to merge them.
    
    # Let's try a constructive approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = A[i,0] B[0,k] + A[i,1] B[1,k] + A[i,2] B[2,k] + A[i,3] B[3,k]
    
    # We can write this as:
    # Let X0 = A[:, 0:2], X1 = A[:, 2:4]  (each 2x2)
    # Let Y0 = B[0:2, :], Y1 = B[2:4, :]  (each 2x4)
    # Then C = X0 Y0 + X1 Y1
    
    # X0 Y0 is 2x2 * 2x4, rank at most 2*2*4 = 16 (schoolbook)
    # X1 Y1 is 2x2 * 2x4, rank at most 16
    # Total: 32 (same as schoolbook)
    
    # Let's try Strassen on the 2x2 * 2x2 parts:
    # X0 Y0 = [x00 x01; x10 x11] * [y00 y01 y02 y03; y10 y11 y12 y13]
    # = [x00*y00 + x01*y10, x00*y01 + x01*y11, x00*y02 + x01*y12, x00*y03 + x01*y13;
    #    x10*y00 + x11*y10, x10*y01 + x11*y11, x10*y02 + x11*y12, x10*y03 + x11*y13]
    
    # For 2x2 * 2x4, the schoolbook is 2*2*4 = 16.
    # Can we do better? The tensor is in R^4 ⊗ R^8 ⊗ R^8.
    # I believe the rank is 16 for 2x2*2x4 (no known improvement).
    
    # Let me try a different approach: use random initialization and optimization.
    # But that might not converge to exact integer/half-integer solutions.
    
    # For now, let's try a few known constructions.
    
    # Construction: use the fact that M_{2,4,4} can be embedded in M_{4,4,4}
    # M_{4,4,4} has rank at most 49 (Strassen) or 48 (Lader).
    # But we need M_{2,4,4}, which is a sub-tensor.
    
    # Actually, let's just return the schoolbook for now and try to improve.
    # The key is to find a valid decomposition with rank < 32.
    
    # Let's try rank 28, 24, 20, 16...
    
    # For each candidate rank r, try to find U, V, W with integer/half-integer entries.
    # This is a hard problem, so let's try specific constructions.
    
    # Construction 1: Use 2x2 Strassen blocks
    # A = [A1 A2], B = [[B11 B12], [B21 B22]] where each block is 2x2
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # For C1 = A1 B11 + A2 B21:
    # A1, B11, A2, B21 are all 2x2.
    # We can use Strassen to compute A1 B11 with 7 mults and A2 B21 with 7 mults.
    # But we can share some: M1 = A1 B11, M2 = A2 B21, C1 = M1 + M2.
    # That's 14 mults for C1.
    # Similarly, 14 mults for C2.
    # Total: 28 mults.
    
    # Let's construct this explicitly.
    # Strassen for 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For C1 = A1 B11 + A2 B21:
    # Let's compute A1 B11 with Strassen (7 mults) and A2 B21 with Strassen (7 mults).
    # But the intermediate sums are different, so we can't easily share.
    
    # Actually, the rank of the sum of two 2x2*2x2 tensors is at most 14 (7+7).
    # So C1 has rank at most 14, C2 has rank at most 14, total 28.
    
    # Let's construct this.
    U28 = []
    V28 = []
    W28 = []
    
    # Helper: add a Strassen decomposition for X (2x2) * Y (2x2) -> Z (2x2)
    # X[i,j] -> U indices: i*2+j (in the 2x4 A matrix, first 2 columns for A1)
    # Actually, let's be more careful with indexing.
    
    # A is 2x4: A[i, j] for i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j, k] for j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i, k] for i in {0,1}, k in {0,1,2,3}
    
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C1 = A1 B11 + A2 B21 (2x2)
    # C2 = A1 B12 + A2 B22 (2x2)
    
    # For A1 B11:
    # A1[i,j] = A[i, j] for i in {0,1}, j in {0,1}
    # B11[j,k] = B[j, k] for j in {0,1}, k in {0,1}
    # Result: R1[i,k] for i in {0,1}, k in {0,1}
    
    # For A2 B21:
    # A2[i,j] = A[i, j+2] for i in {0,1}, j in {0,1}
    # B21[j,k] = B[j+2, k] for j in {0,1}, k in {0,1}
    # Result: R2[i,k] for i in {0,1}, k in {0,1}
    
    # C1[i,k] = R1[i,k] + R2[i,k]
    
    # Strassen for A1 B11:
    # M1_11 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M2_11 = (A[1,0]+A[1,1]) * B[0,0]
    # M3_11 = A[0,0] * (B[0,1]-B[1,1])
    # M4_11 = A[1,1] * (B[1,0]-B[0,0])
    # M5_11 = (A[0,0]+A[0,1]) * B[1,1]
    # M6_11 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # M7_11 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # R1[0,0] = M1_11 + M4_11 - M5_11 + M7_11
    # R1[0,1] = M3_11 + M5_11
    # R1[1,0] = M2_11 + M4_11
    # R1[1,1] = M1_11 - M2_11 + M3_11 + M6_11
    
    # Strassen for A2 B21:
    # M1_21 = (A[0,2]+A[1,3]) * (B[2,0]+B[3,1])
    # M2_21 = (A[1,2]+A[1,3]) * B[2,0]
    # M3_21 = A[0,2] * (B[2,1]-B[3,1])
    # M4_21 = A[1,3] * (B[3,0]-B[2,0])
    # M5_21 = (A[0,2]+A[0,3]) * B[3,1]
    # M6_21 = (A[1,2]-A[0,2]) * (B[2,0]+B[2,1])
    # M7_21 = (A[0,3]-A[1,3]) * (B[3,0]+B[3,1])
    # R2[0,0] = M1_21 + M4_21 - M5_21 + M7_21
    # R2[0,1] = M3_21 + M5_21
    # R2[1,0] = M2_21 + M4_21
    # R2[1,1] = M1_21 - M2_21 + M3_21 + M6_21
    
    # C1[0,0] = R1[0,0] + R2[0,0]
    # C1[0,1] = R1[0,1] + R2[0,1]
    # C1[1,0] = R1[1,0] + R2[1,0]
    # C1[1,1] = R1[1,1] + R2[1,1]
    
    # Similarly for C2 = A1 B12 + A2 B22.
    
    # This gives rank 28. Let's construct it.
    
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (16, rank)
    # W has shape (p*n, rank) = (8, rank)
    
    # U[i*m + j, c] corresponds to A[i, j]
    # V[j*p + k, c] corresponds to B[j, k]
    # W[k*n + i, c] corresponds to C[k, i] ... wait, the trace convention is W[k*n+i, r]
    # So W index is k*n + i, where k is the column of B / row of C, and i is the row of A / column of C.
    # Actually, for C[i,k], the W index should be k*n + i.
    
    # Let's build the 28-rank decomposition.
    rank28 = 28
    U28 = np.zeros((8, rank28), dtype=float)
    V28 = np.zeros((16, rank28), dtype=float)
    W28 = np.zeros((8, rank28), dtype=float)
    
    # Strassen for A1 B11 (7 mults)
    # A1[i,j] = A[i, j], B11[j,k] = B[j, k]
    # U index for A[i,j] is i*4 + j (since m=4)
    # V index for B[j,k] is j*4 + k (since p=4)
    # W index for C[i,k] is k*2 + i (since n=2)
    
    def add_strassen(A_row, A_col_offset, B_row_offset, B_col_offset, C_row, C_col_offset, start_col):
        """
        Add a Strassen decomposition for A[A_row, A_col_offset:A_col_offset+2] * 
        B[B_row_offset:B_row_offset+2, B_col_offset:B_col_offset+2] -> C[C_row, C_col_offset:C_col_offset+2]
        
        A_row: row of A (0 or 1)
        A_col_offset: column offset in A (0 or 2)
        B_row_offset: row offset in B (0 or 2)
        B_col_offset: col offset in B (0 or 2)
        C_row: row of C (0 or 1)
        C_col_offset: col offset in C (0 or 2)
        start_col: starting column in U, V, W
        """
        a00 = A_row * 4 + A_col_offset
        a01 = A_row * 4 + A_col_offset + 1
        a10 = (A_row + 1) * 4 + A_col_offset
        a11 = (A_row + 1) * 4 + A_col_offset + 1
        
        b00 = B_row_offset * 4 + B_col_offset
        b01 = B_row_offset * 4 + B_col_offset + 1
        b10 = (B_row_offset + 1) * 4 + B_col_offset
        b11 = (B_row_offset + 1) * 4 + B_col_offset + 1
        
        c00 = C_col_offset * 2 + C_row
        c01 = (C_col_offset + 1) * 2 + C_row
        c10 = C_col_offset * 2 + (C_row + 1)
        c11 = (C_col_offset + 1) * 2 + (C_row + 1)
        
        # M1 = (a00+a11)(b00+b11)
        col = start_col
        U28[a00, col] = 1
        U28[a11, col] = 1
        V28[b00, col] = 1
        V28[b11, col] = 1
        W28[c00, col] = 1
        W28[c01, col] = 1
        W28[c10, col] = 0
        W28[c11, col] = 1
        col += 1
        
        # M2 = (a10+a11)b00
        U28[a10, col] = 1
        U28[a11, col] = 1
        V28[b00, col] = 1
        W28[c00, col] = 0
        W28[c01, col] = 0
        W28[c10, col] = 1
        W28[c11, col] = -1
        col += 1
        
        # M3 = a00(b01-b11)
        U28[a00, col] = 1
        V28[b01, col] = 1
        V28[b11, col] = -1
        W28[c00, col] = 0
        W28[c01, col] = 1
        W28[c10, col] = 0
        W28[c11, col] = 1
        col += 1
        
        # M4 = a11(b10-b00)
        U28[a11, col] = 1
        V28[b10, col] = 1
        V28[b00, col] = -1
        W28[c00, col] = 1
        W28[c01, col] = 0
        W28[c10, col] = 1
        W28[c11, col] = 0
        col += 1
        
        # M5 = (a00+a01)b11
        U28[a00, col] = 1
        U28[a01, col] = 1
        V28[b11, col] = 1
        W28[c00, col] = -1
        W28[c01, col] = 1
        W28[c10, col] = 0
        W28[c11, col] = 0
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U28[a10, col] = 1
        U28[a00, col] = -1
        V28[b00, col] = 1
        V28[b01, col] = 1
        W28[c00, col] = 0
        W28[c01, col] = 0
        W28[c10, col] = 0
        W28[c11, col] = 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U28[a01, col] = 1
        U28[a11, col] = -1
        V28[b10, col] = 1
        V28[b11, col] = 1
        W28[c00, col] = 1
        W28[c01, col] = 0
        W28[c10, col] = 0
        W28[c11, col] = 0
        col += 1
        
        return col
    
    # C1 = A1 B11 + A2 B21
    # A1 B11: A_row=0, A_col_offset=0, B_row_offset=0, B_col_offset=0, C_row=0, C_col_offset=0
    col = add_strassen(0, 0, 0, 0, 0, 0, 0)
    
    # A2 B21: A_row=0, A_col_offset=2, B_row_offset=2, B_col_offset=0, C_row=0, C_col_offset=0
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    
    # C2 = A1 B12 + A2 B22
    # A1 B12: A_row=0, A_col_offset=0, B_row_offset=0, B_col_offset=2, C_row=0, C_col_offset=2
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    
    # A2 B22: A_row=0, A_col_offset=2, B_row_offset=2, B_col_offset=2, C_row=0, C_col_offset=2
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    
    # Verify
    if col == rank28 and _verify_fast(U28, V28, W28):
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # If verification failed, fall back to schoolbook
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
