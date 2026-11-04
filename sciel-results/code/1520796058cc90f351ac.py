"""
Program to find rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Strategy:
  Use the known rank-7 Strassen decomposition for 2x2 matrix multiplication,
  applied to each of the 10 columns of the 3x10 matrix B independently.
  
  For each column k of B (10 columns total), we multiply A (2x3) by B[:,k] (3x1).
  The output is C[:,k] (2x1).
  
  We can view this as: for each k, compute C[:,k] = A @ B[:,k].
  
  Since m=3, we can partition the middle dimension into 2+1.
  - For the first 2 rows of B (B[0:2, k]), multiply by A[:, 0:2] (2x2) -> use Strassen (rank 7)
  - For the last 1 row of B (B[2, k]), multiply by A[:, 2] (2x1) -> schoolbook (rank 2)
  
  Total per column: 7 + 2 = 9
  Total rank: 10 * 9 = 90
  
  But we can do better. Let's use the trivial schoolbook: rank = 2*3*10 = 60.
  
  Actually, let's try a different approach. The current program achieves rank 55.
  Let me verify that and see if we can improve.
  
  The key insight: we can use the fact that for 2x3 * 3x10, we can exploit the structure.
  
  Let me try the direct schoolbook: rank = 2*3*10 = 60.
  The current program gets 55 via Strassen on 2x2 blocks.
  
  Can we do better than 55? Let's think about lower bounds.
  The rank of matrix multiplication tensor M_{n,m,p} is at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
  It's also at least the border rank, which for M_{2,3,10} is at least 2*3*10/3 = 20 (weak bound).
  
  Actually, for M_{2,3,10}, the best known upper bound might be around 40-50.
  
  Let me try to construct a better decomposition.
  
  Idea: Use the rank-7 Strassen for the 2x2 part (rows 0,1 of A and rows 0,1 of B),
  which gives 5 blocks * 7 = 35.
  For the remaining part (row 2 of A, row 2 of B), we need 2*10 = 20 terms.
  Total: 55.
  
  Can we reduce the 20 terms? The sub-problem is 2x1 * 1x10, which has rank exactly 20
  (it's a simple outer product structure, rank = 2*10 = 20).
  
  So 55 seems hard to beat with this partitioning.
  
  Alternative: partition differently. What if we don't split the middle dimension?
  Use a single Strassen-like decomposition for 2x3 * 3x10 directly?
  
  The tensor M_{2,3,10} has 2*3*10 = 60 entries. The trivial rank is 60.
  Strassen gives 55. Can we do better?
  
  Let me try: use the fact that 3 = 2+1, and for the 2x2 part use rank 7 (Strassen),
  and for the cross terms between the 2x2 block and the 1x1 block, use optimized formulas.
  
  Actually, the current approach already does this optimally for this partition.
  
  Let me try a completely different strategy: use the rank-19 decomposition for 3x3x3
  (Strassen's 3x3x3 algorithm) embedded somehow.
  
  Or: since 10 = 2*5, and 3 = 2+1, maybe we can use a Kronecker product structure.
  
  Actually, let me just verify the current rank-55 solution is correct and return it.
  The score is seed/value where seed is the trivial rank 60.
  Score = 60/55 = 1.0909...
  
  To improve, we need rank < 55.
  
  Let me try: can we get rank 50?
  
  Idea: For the 2x1 * 1x10 part, instead of 20 terms, can we share some computation
  with the Strassen part?
  
  In the Strassen part, we compute products involving A[0:2, 0:2] and B[0:2, :].
  The remaining part involves A[:, 2] and B[2, :].
  
  These are independent, so no sharing is possible without changing the partition.
  
  What if we use a different partition of the middle dimension?
  Partition 3 as 1+2 instead of 2+1?
  
  Part 1: row 0 of A (2x1) times row 0 of B (1x10) -> schoolbook rank 20
  Part 2: rows 1,2 of A (2x2) times rows 1,2 of B (2x10) -> 5 Strassen blocks, rank 35
  Total: 55. Same.
  
  What about partitioning the output? The output is 2x10.
  
  I think 55 is the best we can get with these simple strategies.
  
  Let me try to implement a search for a better decomposition using linear algebra.
  We can set up the tensor decomposition as a system of equations and try to find
  a solution with fewer than 55 terms.
  
  But this is computationally expensive. Let me just return the rank-55 solution.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    rank = 55

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (5 blocks of 2x2x2)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        U[0 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        U[1 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        U[0 * 3 + 0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        V[1 * 10 + k1, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        U[1 * 3 + 1, idx] = 1.0
        V[1 * 10 + k0, idx] = 1.0
        V[0 * 10 + k0, idx] = -1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0 * 3 + 0, idx] = 1.0
        U[0 * 3 + 1, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        U[1 * 3 + 0, idx] = 1.0
        U[0 * 3 + 0, idx] = -1.0
        V[0 * 10 + k0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        U[0 * 3 + 1, idx] = 1.0
        U[1 * 3 + 1, idx] = -1.0
        V[1 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    if time.time() - start_time > DEADLINE:
        pass

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r]
        v_r = V[:, r]
        w_r = W[:, r]

        A_weighted = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_weighted[i, j] = u_r[i * m + j]

        B_weighted = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_weighted[j, k] = v_r[j * p + k]

        prod_r = np.sum(A_weighted * A) * np.sum(B_weighted * B)

        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * w_r[k * n + i]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
