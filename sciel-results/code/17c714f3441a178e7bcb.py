"""Structural block-partition decomposition: Strassen on inner 4=2+2, n=2 rows."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Index conventions (0-based):
    #   U[a*m + j, r]   : a in [0,n), j in [0,m)
    #   V[j*p + k, r]   : j in [0,m), k in [0,p)
    #   W[k*n + i, r]   : k in [0,p), i in [0,n)
    # Target: sum_r U[a*m+j,r] V[j*p+k,r] W[k*n+i,r]
    #          = 1 if (a==i and j==k) else 0

    # Split inner dimension m = 2 + 2.
    # For a fixed output row i (0 or 1) and output col k (0..6):
    #   C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # Split into two 2x2 "blocks": (j=0,1) and (j=2,3).
    # Strassen on each 2x2 block (2 A-entries x 2 B-entries -> 2 C-entries):
    #   M1 = (A0+A1)(B0+B1)
    #   M2 = (A0+A1)B1
    #   M3 = A0(B0-B1)
    #   M4 = A0B0
    #   M5 = (A0-A1)B1
    #   M6 = A1(B0-B1)
    #   M7 = (A1+A0)B0  -- note M7 = M1 - M2 - M3? Let's use standard 7.
    # Standard Strassen 2x2 (A,B 2x2, C=A*B):
    #   M1=(A11+A22)(B11+B22)
    #   M2=(A21+A22)B11
    #   M3=A11(B21-B22)
    #   M4=A22(B11-B21)
    #   M5=(A11+A12)B22
    #   M6=(A21-A11)(B11+B12)
    #   M7=(A12-A22)(B21+B22)
    #   C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6
    # Here each "block" is 2 (A-entries) x 2 (B-entries) producing 2 C-entries
    # (the two output rows i=0,1 share the same j's and k's).
    # So A_block is 2x1 (two i-rows, one j), B_block is 1x2 (one j, two k's).
    # That's a 2x1 * 1x2 = 2x2 product. Strassen applies to 2x2*2x2, but here
    # it's a rank-1 structured product: C[i,k]=sum_j A[i,j]B[j,k] for 2 i, 2 j, 2 k.
    # This is exactly a 2x2 * 2x2 matrix product where A is 2x2 (rows i, cols j)
    # and B is 2x2 (rows j, cols k). Yes! For a fixed (i-pair, j-pair, k-pair).

    # So for each i-pair (only one: {0,1}), j-pair {0,1} or {2,3}, k-pair {0,1},...,{4,5} or {6,?}
    # p=7 is odd. Split p=7 into pairs: (0,1),(2,3),(4,5), and single (6).
    # For each full k-pair: 7 multiplications. For the single k=6: it's a 2x1 * 1x1 = 2x1,
    # needs 2 multiplications (schoolbook).

    # Total per i-pair:
    #   j-pair {0,1}: k-pairs (0,1),(2,3),(4,5) -> 3*7=21, k=6 -> 2
    #   j-pair {2,3}: k-pairs (0,1),(2,3),(4,5) -> 3*7=21, k=6 -> 2
    # Total = 21+2+21+2 = 46.

    # Let me build this explicitly.
    # Each multiplication M corresponds to a rank-1 term: U entry, V entry, W entry.
    # M = (linear combo of A-entries) * (linear combo of B-entries)
    # contributes to specific C[i,k] entries with coefficients.

    # I'll represent each "multiplication" as:
    #   u_coef: dict {(i,j): coeff}  (the A-side linear form)
    #   v_coef: dict {(j,k): coeff}  (the B-side linear form)
    #   w_coef: dict {(i,k): coeff}  (how it contributes to C)
    # Then U[i*m+j, r] = u_coef[(i,j)], V[j*p+k, r] = v_coef[(j,k)], W[k*n+i, r] = w_coef[(i,k)]

    multiplications = []

    def add_mult(u_coef, v_coef, w_coef):
        multiplications.append((u_coef, v_coef, w_coef))

    # Strassen for 2x2 * 2x2 where A[i,j], B[j,k], C[i,k], i in {0,1}, j in {0,1}, k in {0,1}
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    def strassen_block(j0, k0):
        # j0: first j index (0 or 2), k0: first k index (0,2,4)
        # A-entries: (i, j0), (i, j0+1) for i in 0,1
        # B-entries: (j0, k0), (j0+1, k0), (j0, k0+1), (j0+1, k0+1)
        # C-entries: (i, k0), (i, k0+1) for i in 0,1
        j1 = j0 + 1
        k1 = k0 + 1
        # A00 = (0,j0), A01=(0,j1), A10=(1,j0), A11=(1,j1)
        # B00=(j0,k0), B01=(j0,k1), B10=(j1,k0), B11=(j1,k1)
        # C00=(0,k0), C01=(0,k1), C10=(1,k0), C11=(1,k1)

        # M1 = (A00+A11)(B00+B11)
        add_mult(
            {(0, j0): 1, (1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {(0, k0): 1, (1, k1): 1}
        )
        # M2 = (A10+A11)B00
        add_mult(
            {(1, j0): 1, (1, j1): 1},
            {(j0, k0): 1},
            {(1, k0): 1, (1, k1): 1}
        )
        # M3 = A00(B10-B11)
        add_mult(
            {(0, j0): 1},
            {(j1, k0): 1, (j1, k1): -1},
            {(0, k1): 1, (1, k1): 1}
        )
        # M4 = A11(B00-B10)
        add_mult(
            {(1, j1): 1},
            {(j0, k0): 1, (j1, k0): -1},
            {(0, k0): 1, (1, k0): 1}
        )
        # M5 = (A00+A01)B11
        add_mult(
            {(0, j0): 1, (0, j1): 1},
            {(j1, k1): 1},
            {(0, k0): -1, (0, k1): 1}
        )
        # M6 = (A10-A00)(B00+B01)
        add_mult(
            {(1, j0): 1, (0, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {(1, k1): 1}
        )
        # M7 = (A01-A11)(B10+B11)
        add_mult(
            {(0, j1): 1, (1, j1): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {(0, k0): 1}
        )

    # Apply Strassen to each (j-pair, k-pair)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            strassen_block(j0, k0)

    # Handle k=6 (single) with schoolbook: 2 multiplications per j-pair
    for j0 in [0, 2]:
        for i in [0, 1]:
            # C[i,6] += A[i, j0] * B[j0, 6]
            add_mult(
                {(i, j0): 1},
                {(j0, 6): 1},
                {(i, 6): 1}
            )
            # C[i,6] += A[i, j0+1] * B[j0+1, 6]
            add_mult(
                {(i, j0 + 1): 1},
                {(j0 + 1, 6): 1},
                {(i, 6): 1}
            )

    rank = len(multiplications)

    # Build U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for r, (u_coef, v_coef, w_coef) in enumerate(multiplications):
        for (i, j), val in u_coef.items():
            U[i * m + j, r] = val
        for (j, k), val in v_coef.items():
            V[j * p + k, r] = val
        for (i, k), val in w_coef.items():
            W[k * n + i, r] = val

    # Verify
    def verify(U, V, W):
        rank_v = U.shape[1]
        for i in range(n):
            for k in range(p):
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                lhs = 0.0
                                for r in range(rank_v):
                                    lhs += W[k * n + i, r] * U[a * m + b, r] * V[c * p + d, r]
                                rhs = 1.0 if (a == i and b == c and d == k) else 0.0
                                if abs(lhs - rhs) > 1e-9:
                                    return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        r_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, r_idx] = 1
                    V[j * p + k, r_idx] = 1
                    W[k * n + i, r_idx] = 1
                    r_idx += 1

    # Convert to lists
    U_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
    V_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
    W_list = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
