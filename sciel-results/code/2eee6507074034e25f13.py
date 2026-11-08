"""
Rank-88 decomposition for 3x4 * 4x8 matrix multiplication.

Uses:
  - 32 schoolbook terms for the 1x4x8 sub-block (i=0)
  - 2 j-blocks (pairs of columns) x 4 k-blocks (pairs of rows) x 7 Strassen terms = 56

Total rank = 88.

The tensor convention is W[k*n + i, r] with C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
"""
import time

# Safety deadline: kill if we exceed this wall-clock time.
_DEADLINE = 120.0
_START_TIME = time.time()

def _check_deadline():
    if time.time() - _START_TIME > _DEADLINE:
        return True
    return False

def construct():
    # Dimensions: A is n x m (3x4), B is m x p (4x8), C is n x p (3x8)
    n = 3
    m = 4
    p = 8
    rank = 88

    # Allocate factor matrices as lists of lists for exact arithmetic
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # --- Part 1: i=0 block (1x4x8) using schoolbook ---
    # For each j in 0..3, k in 0..7:
    #   U[0*m + j] += 1
    #   V[j*p + k] += 1
    #   W[k*n + 0] += 1
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1
            if col % 16 == 0 and _check_deadline():
                # Return whatever we have so far (will be invalid if incomplete)
                # But we should not hit deadline for rank-88 construction
                pass

    # --- Part 2: i=1,2 block (2x4x8) using Strassen on 2x2x2 sub-problems ---
    # Split j into 2 blocks of 2: J0=[0,1], J1=[2,3]
    # Split k into 4 blocks of 2: K0=[0,1], K1=[2,3], K2=[4,5], K3=[6,7]
    # For each (j_block, k_block) pair, apply 7-term Strassen to the 2x2x2 sub-problem.

    # Strassen 2x2x2 factors (local 2x2 matrices):
    # sU[t][l] where l encodes (local_i, local_j) with local_i = l//2, local_j = l%2
    # sV[t][l] where l encodes (local_j, local_k)
    # sW[t][l] where l encodes (local_k, local_i)

    # Standard Strassen (one valid set):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = A00(B10-B11)
    # M4 = A11(B00-B10)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    sU = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    sV = [
        [1, 0, 0, 1],   # M1: B00 + B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B10 - B11
        [-1, 0, 1, 0],  # M4: B00 - B10
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00 + B01
        [0, 0, 1, 1],   # M7: B10 + B11
    ]
    sW = [
        [1, 0, 0, 1],   # M1: C00 + C11
        [0, 0, 1, -1],  # M2: C10 - C11
        [0, 1, 0, 1],   # M3: C01 + C11
        [1, 0, 1, 0],   # M4: C00 + C10
        [-1, 1, 0, 0],  # M5: -C00 + C01
        [0, 0, 0, 1],   # M6: C11
        [1, 0, 0, 0],   # M7: C00
    ]

    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            for t in range(7):
                # Update U: local (i', j') -> global (i=1+i', j=j_start+j')
                for l in range(4):
                    val = sU[t][l]
                    if val != 0:
                        li = l // 2
                        lj = l % 2
                        gi = 1 + li
                        gj = j_start + lj
                        U[gi * m + gj][col] += val

                # Update V: local (j', k') -> global (j=j_start+j', k=k_start+k')
                for l in range(4):
                    val = sV[t][l]
                    if val != 0:
                        lj = l // 2
                        lk = l % 2
                        gj = j_start + lj
                        gk = k_start + lk
                        V[gj * p + gk][col] += val

                # Update W: local (k', i') -> global (k=k_start+k', i=1+i')
                # W index is [k*n + i]
                for l in range(4):
                    val = sW[t][l]
                    if val != 0:
                        lk = l // 2
                        li = l % 2
                        gk = k_start + lk
                        gi = 1 + li
                        W[gk * n + gi][col] += val

                col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    return {"U": U, "V": V, "W": W}
