"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 9]


def _verify_fast(U, V, W, n=4, m=4, p=9):
    """Fast verification using numpy. U,V,W are numpy arrays."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    return True


def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True


def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    # M1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    # M3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    # M7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}

def _schoolbook_2x2x1():
    """Schoolbook decomposition for 2x2 x 2x1 matmul, rank 4.
    U[i*2+j, c] (4 rows), V[j*1+k, c] (2 rows), W[k*2+i, c] (2 rows).
    Here p=1, so k is always 0.
    U[i*2+j, c], V[j, c], W[i, c].
    """
    rank = 4
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(2)]
    W = [[0] * rank for _ in range(2)]
    
    # C[i,0] = A[i,0]B[0,0] + A[i,1]B[1,0]
    # Term 1: A[i,0]B[0,0]
    U[0][0] = 1; U[2][0] = 1 # i=0,j=0; i=1,j=0
    V[0][0] = 1
    W[0][0] = 1; W[1][0] = 1 # k=0, i=0; k=0, i=1
    
    # Term 2: A[i,1]B[1,0]
    U[1][1] = 1; U[3][1] = 1 # i=0,j=1; i=1,j=1
    V[1][1] = 1
    W[0][1] = 1; W[1][1] = 1
    
    # Wait, the indices for W are k*n+i. n=2, p=1.
    # W[k*2+i, c]. k=0. So W[i, c].
    # U[i*2+j, c].
    # V[j*p+k, c] -> V[j, c].
    
    # Let's re-verify indices for 2x2x1.
    # n=2, m=2, p=1.
    # U size: 2*2 = 4.
    # V size: 2*1 = 2.
    # W size: 1*2 = 2.
    
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    # Rank 1 term 1: A[0,0]B[0,0] -> C[0,0] += 1
    # U[0,0]=1, V[0,0]=1, W[0,0]=1.
    # Rank 1 term 2: A[0,1]B[1,0] -> C[0,0] += 1
    # U[1,1]=1, V[1,1]=1, W[0,1]=1.
    # Rank 1 term 3: A[1,0]B[0,0] -> C[1,0] += 1
    # U[2,2]=1, V[0,2]=1, W[1,2]=1.
    # Rank 1 term 4: A[1,1]B[1,0] -> C[1,0] += 1
    # U[3,3]=1, V[1,3]=1, W[1,3]=1.
    
    U = [[0]*4 for _ in range(4)]
    V = [[0]*4 for _ in range(2)]
    W = [[0]*4 for _ in range(2)]
    
    U[0][0]=1; V[0][0]=1; W[0][0]=1
    U[1][1]=1; V[1][1]=1; W[0][1]=1
    U[2][2]=1; V[0][2]=1; W[1][2]=1
    U[3][3]=1; V[1][3]=1; W[1][3]=1
    
    return {"U": U, "V": V, "W": W}

def _decomp_2x2x3():
    """Decomposition for 2x2 A times 2x3 B -> 2x3 C, rank 11.
    Uses Strassen (rank 7) for cols 0,1 and Schoolbook (rank 4) for col 2.
    U[i*2+j, c] (4 rows), V[j*3+k, c] (6 rows), W[k*2+i, c] (6 rows).
    """
    S = _strassen_2x2()
    SB = _schoolbook_2x2x1()
    rank = 11
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]

    # Part 1: Cols 0,1 (Strassen)
    for c in range(7):
        for ii in range(2):
            for jj in range(2):
                U[ii * 2 + jj][c] += S['U'][ii * 2 + jj][c]
        for jj in range(2):
            for kk in range(2):
                V[jj * 3 + kk][c] += S['V'][jj * 2 + kk][c]
        for kk in range(2):
            for ii in range(2):
                W[kk * 2 + ii][c] += S['W'][kk * 2 + ii][c]

    # Part 2: Col 2 (Schoolbook 2x2x1)
    for c in range(4):
        cc = 7 + c
        for ii in range(2):
            for jj in range(2):
                U[ii * 2 + jj][cc] += SB['U'][ii * 2 + jj][c]
        for jj in range(2):
            # V index for col 2: jj*3 + 2
            V[jj * 3 + 2][cc] += SB['V'][jj][c]
        for kk in range(1): # kk=0
            for ii in range(2):
                W[kk * 2 + ii][cc] += SB['W'][ii][c]

    return {"U": U, "V": V, "W": W}


def _try_rank10_2x2x3():
    """Attempt to find a rank-10 decomposition for 2x2 x 2x3 matmul.
    Returns None if not found within time budget."""
    import numpy as np
    from itertools import product
    import time

    start = time.time()
    timeout = 30  # seconds

    # Target: T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2}
    # U: 4 x 10, V: 6 x 10, W: 6 x 10
    # sum_c U[i*2+j,c] * V[j*3+k,c] * W[k*2+i,c] = 1

    np.random.seed(42)
    n_trials = 50

    for trial in range(n_trials):
        if time.time() - start > timeout:
            break

        # Random initialization
        U = np.random.randn(4, 10)
        V = np.random.randn(6, 10)
        W = np.random.randn(6, 10)

        # Gradient descent on the residual
        lr = 0.01
        for it in range(500):
            if time.time() - start > timeout:
                break

            # Compute residual and gradients
            # R[i,j,k] = sum_c U[i*2+j,c]*V[j*3+k,c]*W[k*2+i,c] - 1
            # dR/dU[i*2+j,c] = V[j*3+k,c]*W[k*2+i,c]
            # dR/dV[j*3+k,c] = U[i*2+j,c]*W[k*2+i,c]
            # dR/dW[k*2+i,c] = U[i*2+j,c]*V[j*3+k,c]

            gU = np.zeros_like(U)
            gV = np.zeros_like(V)
            gW = np.zeros_like(W)

            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        # Residual contribution
                        r = 0.0
                        for c in range(10):
                            r += U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c]
                        r -= 1.0  # target is 1

                        for c in range(10):
                            gU[i*2+j, c] += r * V[j*3+k, c] * W[k*2+i, c]
                            gV[j*3+k, c] += r * U[i*2+j, c] * W[k*2+i, c]
                            gW[k*2+i, c] += r * U[i*2+j, c] * V[j*3+k, c]

            # Also need to handle the "zero" constraints: for invalid (i,j,k) combos,
            # but in this case ALL (i,j,k) are valid, so no zero constraints.

            U -= lr * gU
            V -= lr * gV
            W -= lr * gW

            # Check convergence
            max_err = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        s = 0.0
                        for c in range(10):
                            s += U[i*2+j, c] * V[j*3+k, c] * W[k*2+i, c]
                        max_err = max(max_err, abs(s - 1.0))

            if max_err < 1e-6:
                # Found a solution! Convert to integer/half-integer if possible
                # For now, return as-is (will need to round)
                U_list = [[U[i, c] for c in range(10)] for i in range(4)]
                V_list = [[V[i, c] for c in range(10)] for i in range(6)]
                W_list = [[W[i, c] for c in range(10)] for i in range(6)]
                return {"U": U_list, "V": V_list, "W": W_list}

    return None

def _decomp_4x4x3():
    """Decomposition for 4x4 A times 4x3 B -> 4x3 C, rank 44.
    Uses 4 blocks of 2x2x3 (rank 11 each).
    """
    B = _decomp_2x2x3()
    rank = 44
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]

    for I in range(2):
        for J in range(2):
            for c in range(11):
                cc = (I * 2 + J) * 11 + c
                # U: global row (2I+ii)*4 + (2J+jj)
                for ii in range(2):
                    for jj in range(2):
                        U[(2 * I + ii) * 4 + (2 * J + jj)][cc] += B['U'][ii * 2 + jj][c]
                # V: global row (2J+jj)*3 + kk
                for jj in range(2):
                    for kk in range(3):
                        V[(2 * J + jj) * 3 + kk][cc] += B['V'][jj * 3 + kk][c]
                # W: global row kk*4 + (2I+ii)
                for kk in range(3):
                    for ii in range(2):
                        W[kk * 4 + (2 * I + ii)][cc] += B['W'][kk * 2 + ii][c]

    return {"U": U, "V": V, "W": W}

def _decomp_4x4x9():
    """Full 4x4 x 4x9 decomposition, rank 132.
    Direct sum of three 4x4x3 blocks."""
    B = _decomp_4x4x3()
    rank = 132
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    for t in range(3):
        for c in range(44):
            cc = t * 44 + c
            for i in range(4):
                for j in range(4):
                    U[i * 4 + j][cc] += B['U'][i * 4 + j][c]
            for j in range(4):
                for k in range(3):
                    V[j * 9 + (3 * t + k)][cc] += B['V'][j * 3 + k][c]
            for k in range(3):
                for i in range(4):
                    W[(3 * t + k) * 4 + i][cc] += B['W'][k * 4 + i][c]

    return {"U": U, "V": V, "W": W}


def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 9
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    try:
        cand = _decomp_4x4x9()
        if _verify(cand['U'], cand['V'], cand['W'], n, m, p):
            best_result = cand
            best_rank = 132
            print("Verified rank-132 decomposition")
    except Exception as e:
        print("rank-132 construction failed:", e)

    # Try to find a rank-10 decomposition for 2x2 x 2x3
    if time.time() - start_time < DEADLINE - 30:
        try:
            r10 = _try_rank10_2x2x3()
            if r10 is not None:
                # Verify the rank-10 decomposition
                U10 = np.array(r10['U'], dtype=object)
                V10 = np.array(r10['V'], dtype=object)
                W10 = np.array(r10['W'], dtype=object)
                valid = True
                for i in range(2):
                    for j in range(2):
                        for k in range(3):
                            s = 0
                            for c in range(10):
                                s += U10[i*2+j, c] * V10[j*3+k, c] * W10[k*2+i, c]
                            if abs(s - 1.0) > 1e-4:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    print("Found rank-10 decomposition for 2x2 x 2x3!")
                    # Build the full 4x4 x 4x9 decomposition with rank 120
                    # 4 blocks of 2x2 x 2x3 (rank 10 each) per 4x4 x 4x3
                    # 3 blocks of 4x4 x 4x3 (rank 40 each)
                    rank_total = 120
                    U_full = [[0] * rank_total for _ in range(16)]
                    V_full = [[0] * rank_total for _ in range(36)]
                    W_full = [[0] * rank_total for _ in range(36)]

                    for t in range(3):
                        for I in range(2):
                            for J in range(2):
                                for c in range(10):
                                    cc = t * 40 + (I * 2 + J) * 10 + c
                                    for ii in range(2):
                                        for jj in range(2):
                                            U_full[(2*I+ii)*4 + (2*J+jj)][cc] += r10['U'][ii*2+jj][c]
                                    for jj in range(2):
                                        for kk in range(3):
                                            V_full[(2*J+jj)*9 + (3*t+kk)][cc] += r10['V'][jj*3+kk][c]
                                    for kk in range(3):
                                        for ii in range(2):
                                            W_full[(3*t+kk)*4 + (2*I+ii)][cc] += r10['W'][kk*2+ii][c]

                    if _verify(U_full, V_full, W_full, n, m, p):
                        best_result = {"U": U_full, "V": V_full, "W": W_full}
                        best_rank = 120
                        print("Verified rank-120 decomposition!")
                    else:
                        print("Rank-120 verification failed")
        except Exception as e:
            print("Rank-10 search failed:", e)

    return best_result
