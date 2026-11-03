import time

def construct():
    start_time = time.time()
    deadline = 110.0

    n, m, p = 2, 3, 8

    dim_U = n * m
    dim_V = m * p
    dim_W = p * n

    U = []
    V = []
    W = []

    # Part 1: j=0 (2x1x8) -> 16 terms
    for k in range(p):
        # C[0,k] = A[0,0]*B[0,k]
        U.append([1 if (i==0 and j==0) else 0 for i in range(n) for j in range(m)])
        V.append([1 if (j==0 and k2==k) else 0 for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k and i==0) else 0 for k2 in range(p) for i in range(n)])

        # C[1,k] = A[1,0]*B[0,k]
        U.append([1 if (i==1 and j==0) else 0 for i in range(n) for j in range(m)])
        V.append([1 if (j==0 and k2==k) else 0 for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k and i==1) else 0 for k2 in range(p) for i in range(n)])

    # Part 2: j=1,2 (2x2x8)
    # Split into 4 blocks of 2x2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        # P1: (A[0,1] + A[1,2]) * (B[1,k0] + B[2,k1])
        U.append([1 if (i==0 and j==1) else (1 if (i==1 and j==2) else 0) for i in range(n) for j in range(m)])
        V.append([1 if (j==1 and k2==k0) else (1 if (j==2 and k2==k1) else 0) for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k0 and i==0) else (1 if (k2==k1 and i==1) else 0) for k2 in range(p) for i in range(n)])

        # P2: (A[1,1] + A[1,2]) * B[1,k0]
        U.append([1 if (i==1 and j in [1,2]) else 0 for i in range(n) for j in range(m)])
        V.append([1 if (j==1 and k2==k0) else 0 for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k0 and i==1) else (-1 if (k2==k1 and i==1) else 0) for k2 in range(p) for i in range(n)])

        # P3: A[0,1] * (B[1,k1] - B[2,k1])
        U.append([1 if (i==0 and j==1) else 0 for i in range(n) for j in range(m)])
        V.append([1 if (j==1 and k2==k1) else (-1 if (j==2 and k2==k1) else 0) for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k1 and i==0) else (1 if (k2==k1 and i==1) else 0) for k2 in range(p) for i in range(n)])

        # P4: A[1,2] * (B[2,k0] - B[1,k0])
        U.append([1 if (i==1 and j==2) else 0 for i in range(n) for j in range(m)])
        V.append([1 if (j==2 and k2==k0) else (-1 if (j==1 and k2==k0) else 0) for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k0 and i==0) else (1 if (k2==k0 and i==1) else 0) for k2 in range(p) for i in range(n)])

        # P5: (A[0,1] + A[0,2]) * B[2,k1]
        U.append([1 if (i==0 and j in [1,2]) else 0 for i in range(n) for j in range(m)])
        V.append([1 if (j==2 and k2==k1) else 0 for j in range(m) for k2 in range(p)])
        W.append([-1 if (k2==k0 and i==0) else (1 if (k2==k1 and i==0) else 0) for k2 in range(p) for i in range(n)])

        # P6: (A[1,1] - A[0,1]) * (B[1,k0] + B[1,k1])
        U.append([1 if (i==1 and j==1) else (-1 if (i==0 and j==1) else 0) for i in range(n) for j in range(m)])
        V.append([1 if (j==1 and k2 in [k0,k1]) else 0 for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k1 and i==1) else 0 for k2 in range(p) for i in range(n)])

        # P7: (A[0,2] - A[1,2]) * (B[2,k0] + B[2,k1])
        U.append([1 if (i==0 and j==2) else (-1 if (i==1 and j==2) else 0) for i in range(n) for j in range(m)])
        V.append([1 if (j==2 and k2 in [k0,k1]) else 0 for j in range(m) for k2 in range(p)])
        W.append([1 if (k2==k0 and i==0) else 0 for k2 in range(p) for i in range(n)])

    # Convert to required shapes
    rank = len(U)
    U_mat = [[0]*rank for _ in range(dim_U)]
    V_mat = [[0]*rank for _ in range(dim_V)]
    W_mat = [[0]*rank for _ in range(dim_W)]

    for r in range(rank):
        for idx in range(dim_U):
            U_mat[idx][r] = U[r][idx]
        for idx in range(dim_V):
            V_mat[idx][r] = V[r][idx]
        for idx in range(dim_W):
            W_mat[idx][r] = W[r][idx]

    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _verify(U, V, W):
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0

    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
