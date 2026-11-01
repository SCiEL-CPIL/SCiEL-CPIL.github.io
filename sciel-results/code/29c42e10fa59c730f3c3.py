import numpy as np
import time

def construct():
    n, m, p = 2, 2, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin

    def build_rank32():
        rank = 32
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0

        # Verify Strassen numerically first
        np.random.seed(42)
        A_test = np.random.rand(2, 2)
        B_test = np.random.rand(2, 2)
        C_test = A_test @ B_test

        a, b = A_test[0,0], A_test[0,1]
        c, d = A_test[1,0], A_test[1,1]
        x, y = B_test[0,0], B_test[0,1]
        z, w = B_test[1,0], B_test[1,1]

        m1 = (a+d)*(x+w)
        m2 = (c+d)*x
        m3 = (a-b)*(y+w)
        m4 = (c+d)*z
        m5 = (a+b)*x
        m6 = (c-a)*(x+z)
        m7 = (b-d)*(y+w)

        C00 = m1 - m4 + m6 + m7
        C01 = m3 + m4
        C10 = m2 + m5
        C11 = m1 - m3 + m5 - m2

        if not (np.isclose(C00, C_test[0,0]) and np.isclose(C01, C_test[0,1]) and
                np.isclose(C10, C_test[1,0]) and np.isclose(C11, C_test[1,1])):
            return None

        # Build decomposition using verified Strassen
        for pair_idx in range(4):
            k1 = pair_idx * 2
            k2 = pair_idx * 2 + 1

            # m1 = (a+d)(x+w): U:{a:1,d:1}, V:{B[0,k1]:1, B[1,k2]:1}
            # C[0,k1]+=m1, C[1,k1]+=m1, C[0,k2]+=m1, C[1,k2]+=m1
            # Wait: C[0,k1] gets m1 with coeff +1, C[1,k1] gets m1 with coeff... 
            # From formulas: C[0,0]=m1-m4+m6+m7, C[1,1]=m1-m3+m5-m2
            # So m1 appears in C[0,k1] with +1 and C[1,k2] with +1
            U[0, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1; V[1*p+k2, col] = 1
            W[k1*n+0, col] = 1
            W[k2*n+1, col] = 1
            col += 1

            # m2 = (c+d)(x): U:{c:1,d:1}, V:{B[0,k1]:1}
            # C[1,k1]+=m2, C[1,k2]-=m2
            U[2, col] = 1; U[3, col] = 1
            V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1
            W[k2*n+1, col] = -1
            col += 1

            # m3 = (a-b)(y+w): U:{a:1,b:-1}, V:{B[0,k2]:1, B[1,k2]:1}
            # C[0,k2]+=m3, C[1,k2]-=m3
            U[0, col] = 1; U[1, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            W[k2*n+0, col] = 1
            W[k2*n+1, col] = -1
            col += 1

            # m4 = (c+d)(z): U:{c:1,d:1}, V:{B[1,k1]:1}
            # C[0,k1]-=m4, C[0,k2]+=m4
            U[2, col] = 1; U[3, col] = 1
            V[1*p+k1, col] = 1
            W[k1*n+0, col] = -1
            W[k2*n+0, col] = 1
            col += 1

            # m5 = (a+b)(x): U:{a:1,b:1}, V:{B[0,k1]:1}
            # C[1,k1]+=m5, C[1,k2]+=m5
            U[0, col] = 1; U[1, col] = 1
            V[0*p+k1, col] = 1
            W[k1*n+1, col] = 1
            W[k2*n+1, col] = 1
            col += 1

            # m6 = (c-a)(x+z): U:{c:1,a:-1}, V:{B[0,k1]:1, B[1,k1]:1}
            # C[0,k1]+=m6
            U[2, col] = 1; U[0, col] = -1
            V[0*p+k1, col] = 1; V[1*p+k1, col] = 1
            W[k1*n+0, col] = 1
            col += 1

            # m7 = (b-d)(y+w): U:{b:1,d:-1}, V:{B[0,k2]:1, B[1,k2]:1}
            # C[0,k1]+=m7
            U[1, col] = 1; U[3, col] = -1
            V[0*p+k2, col] = 1; V[1*p+k2, col] = 1
            W[k1*n+0, col] = 1
            col += 1

        # Last single column k=8: rank 4
        k = 8
        U[0, col] = 1; V[0*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[1, col] = 1; V[1*p+k, col] = 1; W[k*n+0, col] = 1; col += 1
        U[2, col] = 1; V[0*p+k, col] = 1; W[k*n+1, col] = 1; col += 1
        U[3, col] = 1; V[1*p+k, col] = 1; W[k*n+1, col] = 1; col += 1

        assert col == rank, f"Expected {rank}, got {col}"

        # Full numerical verification
        np.random.seed(123)
        A_rand = np.random.rand(n, m)
        B_rand = np.random.rand(m, p)
        C_expected = A_rand @ B_rand

        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = sum(U[i*m+j, r] * A_rand[i,j] for j in range(m))
                    v_val = sum(V[j*p+k, r] * B_rand[j,k] for j in range(m))
                    C_decomp[i,k] += W[k*n+i, r] * u_val * v_val

        if not np.allclose(C_expected, C_decomp, atol=1e-10):
            return None

        return U.tolist(), V.tolist(), W.tolist()

    result = build_rank32()
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 36
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
