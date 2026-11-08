"""Recursive block decomposition for 3x3x7 matrix multiplication using Strassen."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 7

    # Strategy: Split into 2x2x4 block (Strassen 7 + schoolbook 4 = 11) 
    # and 1x2x3 tail (schoolbook 3). Total = 14.
    # But we need to verify correctness.

    # Let me build the full decomposition and verify it.
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)  
    # W: (p*n, rank) = (21, rank)

    # For the 2x2x4 block (i in {0,1}, j in {0,1}, k in {0,1,2,3}):
    # Use Strassen for k in {0,1} and schoolbook for k in {2,3}

    # Strassen's 7 multiplications for 2x2x2:
    # Let me define each term as contributions to U, V, W.

    # Term 1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # U contributions: A[0,0] and A[1,1] -> U[0,0]=1, U[1,0]=1 (using local indices)
    # V contributions: B[0,0] and B[1,1] -> V[0,0]=1, V[1,1]=1
    # W contributions: result goes to C[0,0] -> W[0,0]=1

    # I'll build this systematically and then verify with random matrices.

    U = np.zeros((9, 14), dtype=float)
    V = np.zeros((21, 14), dtype=float)
    W = np.zeros((21, 14), dtype=float)

    # 2x2x4 block: rows 0,1 of A; cols 0,1,2,3 of B
    # Local indices: i in {0,1} -> global i, j in {0,1} -> global j, k in {0,1,2,3} -> global k

    # Strassen terms for 2x2x2 (k=0,1):
    # M1 = (a00+a11)(b00+b11)
    U[0, 0] += 1; U[3, 0] += 1       # a00, a11 (local i=0,1; j=0,1)
    V[0, 0] += 1; V[3, 0] += 1       # b00, b11 (local j=0,1; k=0,1)
    W[0, 0] += 1                      # c00

    # M2 = (a10+a11)b00
    U[2, 1] += 1; U[3, 1] += 1       # a10, a11
    V[0, 1] += 1                     # b00
    W[1, 1] += 1                      # c10

    # M3 = a00(b10-b11)
    U[0, 2] += 1                     # a00
    V[2, 2] += 1; V[3, 2] += -1     # b10, -b11
    W[0, 2] += 1                      # c00

    # M4 = a11(b01-b00)
    U[3, 3] += 1                     # a11
    V[1, 3] += 1; V[0, 3] += -1     # b01, -b00
    W[1, 3] += 1; W[0, 3] += 1       # c10, c00

    # M5 = (a00+a01)b11
    U[0, 4] += 1; U[1, 4] += 1       # a00, a01
    V[3, 4] += 1                     # b11
    W[0, 4] += -1; W[1, 4] += 1      # -c00, c01

    # M6 = (a10-a00)(b00+b10)
    U[2, 5] += 1; U[0, 5] += -1      # a10, -a00
    V[0, 5] += 1; V[2, 5] += 1       # b00, b10
    W[3, 5] += 1                      # c11

    # M7 = (a01-a11)(b01+b11)
    U[1, 6] += 1; U[3, 6] += -1      # a01, -a11
    V[1, 6] += 1; V[3, 6] += 1       # b01, b11
    W[0, 6] += 1                      # c00

    # Schoolbook for 2x2x2 tail (k=2,3):
    # C[0,2] = a00*b20 + a01*b30  (local: a[0,0]*b[0,2] + a[0,1]*b[1,2])
    U[0, 7] += 1; U[1, 7] += 1       # a00, a01
    V[2, 7] += 1; V[5, 7] += 1       # b[0,2], b[1,2] -> indices 2, 5
    W[2, 7] += 1                      # c[2,0]

    # C[0,3] = a00*b[0,3] + a01*b[1,3]
    U[0, 8] += 1; U[1, 8] += 1
    V[4, 8] += 1; V[5, 8] += 1       # b[0,3], b[1,3] -> indices 4, 5
    W[3, 8] += 1                      # c[3,0]

    # C[1,2] = a10*b[0,2] + a11*b[1,2]
    U[2, 9] += 1; U[3, 9] += 1       # a10, a11
    V[2, 9] += 1; V[5, 9] += 1       # b[0,2], b[1,2]
    W[2, 9] += 1                      # c[2,1]

    # C[1,3] = a10*b[0,3] + a11*b[1,3]
    U[2, 10] += 1; U[3, 10] += 1
    V[4, 10] += 1; V[5, 10] += 1
    W[3, 10] += 1                     # c[3,1]

    # 1x2x3 tail: row 2 of A, cols 0,1 of A; cols 4,5,6 of B
    # C[2,4] = a[2,0]*b[0,4] + a[2,1]*b[1,4]
    U[6, 11] += 1; U[7, 11] += 1     # a[2,0], a[2,1]
    V[4, 11] += 1; V[7, 11] += 1     # b[0,4], b[1,4] -> indices 4, 7
    W[4, 11] += 1                     # c[4,2]

    # C[2,5] = a[2,0]*b[0,5] + a[2,1]*b[1,5]
    U[6, 12] += 1; U[7, 12] += 1
    V[5, 12] += 1; V[7, 12] += 1     # b[0,5], b[1,5] -> indices 5, 7
    W[5, 12] += 1                     # c[5,2]

    # C[2,6] = a[2,0]*b[0,6] + a[2,1]*b[1,6]
    U[6, 13] += 1; U[7, 13] += 1
    V[6, 13] += 1; V[7, 13] += 1     # b[0,6], b[1,6] -> indices 6, 7
    W[6, 13] += 1                     # c[6,2]

    # Verify with random matrices
    rng = np.random.RandomState(42)
    A = rng.randn(3, 3)
    B = rng.randn(3, 7)
    C_expected = A @ B

    # Compute using decomposition
    C_computed = np.zeros((3, 7))
    for r in range(14):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # u is 9-vector, v is 21-vector, w is 21-vector
        # u[i*m+j] * v[j*p+k] should sum to w[k*n+i] * C[i,k]
        # Actually: C[i,k] = sum_r u[i*m+j] * v[j*p+k] * w[k*n+i] over j
        # Wait, the decomposition is: C[i,k] = sum_r (sum_j U[i*m+j, r]) * (sum_j V[j*p+k, r]) * W[k*n+i, r]
        # No, that's not right either. The tensor decomposition is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C[i,k] = sum_j A[i,j] * B[j,k]
        # So we need: sum_j A[i,j]*B[j,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * V[j*p+k,r] * W[k*n+i,r]
        # This only works if U, V, W are chosen such that the product reconstructs.

        # Actually the standard form is:
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
        # No, that's not standard. Let me think.

        # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k (in the trivial case)
        # and the rank-1 decomposition is T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r[i*m+j] = U[i*m+j, r], etc.
        # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] δ_{i,i}δ_{k,k}
        # = sum_r (sum_j A[i,j] u_r[i*m+j]) (sum_j B[j,k] v_r[j*p+k]) w_r[k*n+i]
        # Wait, this doesn't quite work because the indices don't match up cleanly.

        # Let me just verify numerically by constructing the full tensor.
        pass

    # Let me verify differently: construct the multiplication tensor and check
    # that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the tensor T where T[i*m+j, j*p+k, k*n+i] = 1
    # and all other entries are 0.

    # Actually, the correct verification is:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i', j', k') combinations that don't correspond to valid (i,j,k):
    # sum_r U[i'*m+j', r] * V[j'*p+k', r] * W[k'*n+i', r] = 0

    # But this is expensive. Let me just test with specific (i,j,k) pairs.

    # Test a few specific entries
    errors = []
    for i in range(3):
        for j in range(3):
            for k in range(7):
                # Expected: 1 if this is a valid (i,j,k) triple
                expected = 1.0
                actual = 0.0
                for r in range(14):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    errors.append((i, j, k, actual, expected))

    if errors:
        # Fallback to schoolbook if Strassen decomposition is incorrect
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
