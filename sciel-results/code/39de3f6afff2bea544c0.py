"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Rank 50 decomposition:
    # 4 blocks of 2x2x2 (Strassen, rank 7 each) -> 28
    # 2 blocks of 2x2x2 (Strassen, rank 7 each) -> 14
    # 8 rank-1 terms for 2x4x1 -> 8
    # Total 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a Strassen block
    # A block: rows i0, i0+1; cols j0, j0+1
    # B block: rows j0, j0+1; cols k0, k0+1
    # C block: rows i0, i0+1; cols k0, k0+1
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Local A indices:
        # a00: i0, j0
        # a01: i0, j0+1
        # a10: i0+1, j0
        # a11: i0+1, j0+1

        # Local B indices:
        # b00: j0, k0
        # b01: j0, k0+1
        # b10: j0+1, k0
        # b11: j0+1, k0+1

        # Local C indices (W rows):
        # c00: k0, i0
        # c01: k0+1, i0
        # c10: k0, i0+1
        # c11: k0+1, i0+1

        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M2: (A21+A22)B11
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1

        # M3: A11(B12-B22)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1

        # M5: (A11+A12)B22
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1

        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3)
    # 4 blocks of 2x2x2
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)

    # Part 2: 2x4x2 (k=4..5)
    # 2 blocks of 2x2x2
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # Part 3: 2x4x1 (k=6)
    # 8 rank-1 terms
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    # Now try to reduce rank by finding linear dependencies among the 50 terms
    # Convert to numpy arrays for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build the "joint" matrix: rows indexed by (i,j,k), columns by r
    # Entry: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If columns are linearly dependent, we can reduce rank
    joint = np.zeros((n*m*p, rank), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    joint[idx, r] = U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                idx += 1

    # Find rank of joint matrix and identify independent columns
    try:
        # Use QR with column pivoting to find independent columns
        Q, R, P = np.linalg.qr(joint, mode='complete')
        # The diagonal of R gives us the pivots
        diag_R = np.abs(np.diag(R))
        tol = 1e-10
        # Find pivot columns
        pivot_cols = []
        for i in range(rank):
            if diag_R[i] > tol:
                pivot_cols.append(P[i])
        
        num_independent = len(pivot_cols)
        print(f"Number of independent columns: {num_independent} out of {rank}")
        
        if num_independent < rank:
            # We can reduce to num_independent rank-1 terms
            # Keep only the pivot columns
            U_reduced = U_np[:, pivot_cols]
            V_reduced = V_np[:, pivot_cols]
            W_reduced = W_np[:, pivot_cols]
            
            # Convert back to lists
            U = U_reduced.tolist()
            V = V_reduced.tolist()
            W = W_reduced.tolist()
            rank = num_independent
            print(f"Reduced rank to {rank}")
    except Exception as e:
        print(f"Error in rank reduction: {e}")

    # Verify decomposition
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                if abs(lhs - 1.0) > 1e-9:
                    print(f"Verification failed at i={i},j={j},k={k}: lhs={lhs}")
                    verified = False
                    break
            if not verified:
                break
    if verified:
        print(f"Decomposition verified with rank {rank}")

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        print(f"Time limit approaching: {elapsed}s")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
