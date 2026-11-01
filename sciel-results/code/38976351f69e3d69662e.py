"""Matrix multiplication 2x2x9 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100
    n, m, p = DIMENSIONS  # 2, 2, 9

    # Strategy: split the 9 columns into 4 groups of 2 + 1 group of 1.
    # For each 2-column group, use Strassen's 7-term algorithm for 2x2 matrix mult.
    # For the 1-column group, use schoolbook (4 terms).
    # Total rank = 4*7 + 4 = 32.
    #
    # Strassen for computing C = A * B where A is 2x2, B is 2x2:
    #   A = [[a, b], [c, d]],  B = [[e, f], [g, h]]
    #   M1 = (a+d)(e+h)
    #   M2 = (c+d)e
    #   M3 = a(g+h)
    #   M4 = d(h-f)
    #   M5 = (a+b)h
    #   M6 = (c-a)(e+f)
    #   M7 = (b-d)(g+h)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # Tensor convention:
    #   U[i*m+j, r] — coefficient on A[i,j]
    #   V[j*p+k, r] — coefficient on B[j,k]
    #   W[k*n+i, r] — coefficient on output C[i,k]
    #
    # We need: for each output C[i,k], sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = 1
    # and for all other (i',j',k') combinations the sum is 0.
    #
    # Actually the tensor T[i,j,k] = 1 for all valid (i,j,k) means:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the decomposition must satisfy:
    #   sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1  for all i,j,k
    # Wait, that's not quite right. Let me think again.
    #
    # The matrix multiplication tensor T_{n,m,p} is defined as:
    #   T[i,j,k] = 1  (for all i in [0,n), j in [0,m), k in [0,p))
    #
    # A rank-r decomposition writes:
    #   T[i,j,k] = sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # So we need: for every (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    #
    # Let's build the Strassen decomposition carefully.
    #
    # For a 2x2x2 block covering A columns j0,j1 and B rows j0,j1, B columns k0,k1:
    #   a = A[0,j0], b = A[0,j1], c = A[1,j0], d = A[1,j1]
    #   e = B[j0,k0], f = B[j0,k1], g = B[j1,k0], h = B[j1,k1]
    #
    #   C[0,k0] = M1 + M4 - M5 + M7
    #   C[0,k1] = M3 + M5
    #   C[1,k0] = M2 + M4
    #   C[1,k1] = M1 - M2 + M3 + M6
    #
    # In tensor terms, each M_r contributes to multiple (i,j,k) triples.
    # Let's expand:
    #
    # M1 = (a+d)(e+h) = (A[0,j0]+A[1,j1])(B[j0,k0]+B[j1,k1])
    #   = A[0,j0]*B[j0,k0] + A[0,j0]*B[j1,k1] + A[1,j1]*B[j0,k0] + A[1,j1]*B[j1,k1]
    #
    # M2 = (c+d)*e = (A[1,j0]+A[1,j1])*B[j0,k0]
    #   = A[1,j0]*B[j0,k0] + A[1,j1]*B[j0,k0]
    #
    # M3 = a*(g+h) = A[0,j0]*(B[j1,k0]+B[j1,k1])
    #   = A[0,j0]*B[j1,k0] + A[0,j0]*B[j1,k1]
    #
    # M4 = d*(h-f) = A[1,j1]*(B[j1,k1]-B[j0,k1])
    #   = A[1,j1]*B[j1,k1] - A[1,j1]*B[j0,k1]
    #
    # M5 = (a+b)*h = (A[0,j0]+A[0,j1])*B[j1,k1]
    #   = A[0,j0]*B[j1,k1] + A[0,j1]*B[j1,k1]
    #
    # M6 = (c-a)*(e+f) = (A[1,j0]-A[0,j0])*(B[j0,k0]+B[j0,k1])
    #   = A[1,j0]*B[j0,k0] + A[1,j0]*B[j0,k1] - A[0,j0]*B[j0,k0] - A[0,j0]*B[j0,k1]
    #
    # M7 = (b-d)*(g+h) = (A[0,j1]-A[1,j1])*(B[j1,k0]+B[j1,k1])
    #   = A[0,j1]*B[j1,k0] + A[0,j1]*B[j1,k1] - A[1,j1]*B[j1,k0] - A[1,j1]*B[j1,k1]
    #
    # Now C[0,k0] = M1 + M4 - M5 + M7:
    #   M1: A[0,j0]B[j0,k0] + A[0,j0]B[j1,k1] + A[1,j1]B[j0,k0] + A[1,j1]B[j1,k1]
    #   M4: A[1,j1]B[j1,k1] - A[1,j1]B[j0,k1]
    #  -M5: -A[0,j0]B[j1,k1] - A[0,j1]B[j1,k1]
    #   M7: A[0,j1]B[j1,k0] + A[0,j1]B[j1,k1] - A[1,j1]B[j1,k0] - A[1,j1]B[j1,k1]
    #   Sum: A[0,j0]B[j0,k0] + [A[0,j0]B[j1,k1]-A[0,j0]B[j1,k1]] + A[1,j1]B[j0,k0]
    #        + [A[1,j1]B[j1,k1]+A[1,j1]B[j1,k1]-A[1,j1]B[j1,k1]-A[1,j1]B[j1,k1]]
    #        - A[1,j1]B[j0,k1] - A[0,j1]B[j1,k1] + A[0,j1]B[j1,k0] + A[0,j1]B[j1,k1]
    #        - A[1,j1]B[j1,k0]
    #   = A[0,j0]B[j0,k0] + A[1,j1]B[j0,k0] - A[1,j1]B[j0,k1] - A[1,j1]B[j1,k0]
    #     + A[0,j1]B[j1,k0]
    #   Hmm, this is getting messy. Let me just verify with a numerical check at the end.
    #
    # For the tensor decomposition, each term r gives:
    #   For each (i,j,k) that appears in the expansion of M_r with coefficient coef:
    #     U[i*m+j, r] gets a contribution from the A-factor
    #     V[j*p+k, r] gets a contribution from the B-factor
    #     W[k*n+i, r] gets a contribution from the C-factor
    #
    # But actually, each M_r is a product of one linear form in A, one in B, and one scalar
    # that distributes to outputs. So:
    #   M_r = (sum_{i,j} u_{i,j}^{(r)} A[i,j]) * (sum_{j,k} v_{j,k}^{(r)} B[j,k])
    #   and then C[i,k] += w_{i,k}^{(r)} * M_r
    #
    # So U[i*m+j, r] = u_{i,j}^{(r)}, V[j*p+k, r] = v_{j,k}^{(r)}, W[k*n+i, r] = w_{i,k}^{(r)}
    #
    # Let me build each Strassen term explicitly.

    rank = 32
    U = np.zeros((n*m, rank), dtype=float)  # (4, 32)
    V = np.zeros((m*p, rank), dtype=float)  # (18, 32)
    W = np.zeros((p*n, rank), dtype=float)  # (18, 32)

    r = 0  # current rank index

    # Define Strassen coefficients for a 2x2x2 block
    # A indices: (0,j0),(0,j1),(1,j0),(1,j1) -> flat: 0*m+j0, 0*m+j1, 1*m+j0, 1*m+j1
    # B indices: (j0,k0),(j0,k1),(j1,k0),(j1,k1) -> flat: j0*p+k0, j0*p+k1, j1*p+k0, j1*p+k1
    # C indices: (0,k0),(0,k1),(1,k0),(1,k1) -> flat: k0*n+0, k1*n+0, k0*n+1, k1*n+1

    def strassen_block(j0, j1, k0, k1, r_offset):
        """Fill U, V, W for a Strassen 2x2x2 block. Returns 7 (one rank slot per term)."""
        # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
        #   C[0,k0] += M1, C[1,k1] += M1
        U[0*m+j0, r_offset+0] += 1
        U[1*m+j1, r_offset+0] += 1
        V[j0*p+k0, r_offset+0] += 1
        V[j1*p+k1, r_offset+0] += 1
        W[k0*n+0, r_offset+0] += 1   # C[0,k0]
        W[k1*n+1, r_offset+0] += 1   # C[1,k1]

        # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
        #   C[1,k0] += M2
        U[1*m+j0, r_offset+1] += 1
        U[1*m+j1, r_offset+1] += 1
        V[j0*p+k0, r_offset+1] += 1
        W[k0*n+1, r_offset+1] += 1   # C[1,k0]

        # M3 = A[0,j0] * (B[j1,k0]+B[j1,k1])
        #   C[0,k1] += M3, C[1,k1] += M3
        U[0*m+j0, r_offset+2] += 1
        V[j1*p+k0, r_offset+2] += 1
        V[j1*p+k1, r_offset+2] += 1
        W[k1*n+0, r_offset+2] += 1   # C[0,k1]
        W[k1*n+1, r_offset+2] += 1   # C[1,k1]

        # M4 = A[1,j1] * (B[j1,k1]-B[j0,k1])
        #   C[0,k0] += M4, C[1,k0] += M4
        U[1*m+j1, r_offset+3] += 1
        V[j1*p+k1, r_offset+3] += 1
        V[j0*p+k1, r_offset+3] -= 1
        W[k0*n+0, r_offset+3] += 1   # C[0,k0]
        W[k0*n+1, r_offset+3] += 1   # C[1,k0]

        # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
        #   C[0,k0] -= M5, C[0,k1] += M5
        U[0*m+j0, r_offset+4] += 1
        U[0*m+j1, r_offset+4] += 1
        V[j1*p+k1, r_offset+4] += 1
        W[k0*n+0, r_offset+4] -= 1   # C[0,k0] gets -M5
        W[k1*n+0, r_offset+4] += 1   # C[0,k1] gets +M5

        # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
        #   C[1,k1] += M6
        U[1*m+j0, r_offset+5] += 1
        U[0*m+j0, r_offset+5] -= 1
        V[j0*p+k0, r_offset+5] += 1
        V[j0*p+k1, r_offset+5] += 1
        W[k1*n+1, r_offset+5] += 1   # C[1,k1]

        # M7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
        #   C[0,k0] += M7
        U[0*m+j1, r_offset+6] += 1
        U[1*m+j1, r_offset+6] -= 1
        V[j1*p+k0, r_offset+6] += 1
        V[j1*p+k1, r_offset+6] += 1
        W[k0*n+0, r_offset+6] += 1   # C[0,k0]

        return 7

    # Define schoolbook for 2x2x1 block (A is 2x2, B is 2x1)
    # C[0,k] = A[0,j0]*B[j0,k] + A[0,j1]*B[j1,k]
    # C[1,k] = A[1,j0]*B[j0,k] + A[1,j1]*B[j1,k]
    def schoolbook_2x2x1(j0, j1, k0, r_offset):
        """Fill U, V, W for a schoolbook 2x2x1 block. Uses 4 rank slots."""
        # Term 0: A[0,j0]*B[j0,k0] -> C[0,k0]
        U[0*m+j0, r_offset+0] += 1
        V[j0*p+k0, r_offset+0] += 1
        W[k0*n+0, r_offset+0] += 1

        # Term 1: A[0,j1]*B[j1,k0] -> C[0,k0]
        U[0*m+j1, r_offset+1] += 1
        V[j1*p+k0, r_offset+1] += 1
        W[k0*n+0, r_offset+1] += 1

        # Term 2: A[1,j0]*B[j0,k0] -> C[1,k0]
        U[1*m+j0, r_offset+2] += 1
        V[j0*p+k0, r_offset+2] += 1
        W[k0*n+1, r_offset+2] += 1

        # Term 3: A[1,j1]*B[j1,k0] -> C[1,k0]
        U[1*m+j1, r_offset+3] += 1
        V[j1*p+k0, r_offset+3] += 1
        W[k0*n+1, r_offset+3] += 1

        return 4

    # Now assemble: 4 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7)
    # and 1 schoolbook block for column 8
    r = strassen_block(0, 1, 0, 1, r)
    r = strassen_block(0, 1, 2, 3, r)
    r = strassen_block(0, 1, 4, 5, r)
    r = strassen_block(0, 1, 6, 7, r)
    r = schoolbook_2x2x1(0, 1, 8, r)

    # r should be 4*7 + 4 = 32
    assert r == 32, f"Expected rank 32, got {r}"

    # Verify the decomposition numerically
    # For each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] should equal 1
    tensor = np.einsum('ir,jr,kr->ijk', U, V, W)
    # tensor[i,j,k] should be 1 for all i in [0,2), j in [0,2), k in [0,9)
    expected = np.ones((n, m, p))
    if not np.allclose(tensor, expected, atol=1e-9):
        # Fallback to schoolbook rank 36
        rank36 = n * m * p
        U36 = np.zeros((n*m, rank36))
        V36 = np.zeros((m*p, rank36))
        W36 = np.zeros((p*n, rank36))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    U36[i*m+j, col] = 1
                    V36[j*p+k, col] = 1
                    W36[k*n+i, col] = 1
        return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
