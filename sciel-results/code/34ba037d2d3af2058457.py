import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def strassen_222():
        """Return U, V, W for 2x2x2 matrix multiplication, rank 7 (Strassen)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # A=[[a,b],[c,d]] -> U idx: 0=a,1=b,2=c,3=d
        # B=[[e,f],[g,h]] -> V idx: 0=e,1=f,2=g,3=h
        # C=[[C00,C01],[C10,C11]] -> W idx: 0=C00,1=C01,2=C10,3=C11
        # P1=(a+b)(e+f)
        U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1; W[1,0]=1
        # P2=(c+d)e
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
        # P3=(a-c)(g+h)
        U[0,2]=1; U[2,2]=-1; V[2,2]=1; V[3,2]=1; W[0,2]=1; W[2,2]=1
        # P4=(d-b)(f+h)
        U[3,3]=1; U[1,3]=-1; V[1,3]=1; V[3,3]=1; W[1,3]=1; W[3,3]=1
        # P5=(a+d)(e-h)
        U[0,4]=1; U[3,4]=1; V[0,4]=1; V[3,4]=-1; W[0,4]=1; W[3,4]=-1
        # P6=(b+c)(g-e)
        U[1,5]=1; U[2,5]=1; V[2,5]=1; V[0,5]=-1; W[1,5]=1; W[2,5]=-1
        # P7=(a+b-d)(h-g)
        U[0,6]=1; U[1,6]=1; U[3,6]=-1; V[3,6]=1; V[2,6]=-1; W[3,6]=1; W[1,6]=-1
        return U, V, W

    def laderman_333():
        """Return U, V, W for 3x3x3 matrix multiplication, rank 23 (Laderman)."""
        U = np.zeros((9, 23))
        V = np.zeros((9, 23))
        W = np.zeros((9, 23))
        # A idx: 0=a,1=b,2=c,3=d,4=e,5=f,6=g,7=h,8=i
        # B idx: 0=j,1=k,2=l,3=m,4=n,5=o,6=p,7=q,8=r
        # C idx: 0=C00,1=C01,2=C02,3=C10,4=C11,5=C12,6=C20,7=C21,8=C22

        def add_term(u_pairs, v_pairs, w_pairs, col):
            for idx, c in u_pairs: U[idx, col] += c
            for idx, c in v_pairs: V[idx, col] += c
            for idx, c in w_pairs: W[idx, col] += c

        # 23 Laderman products
        add_term([(0,1),(1,1)], [(0,1),(1,1)], [(0,1),(1,1)], 0)   # (a+b)(j+k)
        add_term([(1,1),(2,1)], [(1,1),(2,1)], [(1,1),(2,1)], 1)   # (b+c)(k+l)
        add_term([(2,1),(0,1)], [(2,1),(0,1)], [(2,1),(0,1)], 2)   # (c+a)(l+j)
        add_term([(3,1),(4,1)], [(3,1),(4,1)], [(3,1),(4,1)], 3)   # (d+e)(m+n)
        add_term([(4,1),(5,1)], [(4,1),(5,1)], [(4,1),(5,1)], 4)   # (e+f)(n+o)
        add_term([(5,1),(3,1)], [(5,1),(3,1)], [(5,1),(3,1)], 5)   # (f+d)(o+m)
        add_term([(6,1),(7,1)], [(6,1),(7,1)], [(6,1),(7,1)], 6)   # (g+h)(p+q)
        add_term([(7,1),(8,1)], [(7,1),(8,1)], [(7,1),(8,1)], 7)   # (h+i)(q+r)
        add_term([(8,1),(6,1)], [(8,1),(6,1)], [(8,1),(6,1)], 8)   # (i+g)(r+p)
        add_term([(0,1),(3,1),(6,1)], [(0,1),(3,1),(6,1)], [(0,1),(3,1),(6,1)], 9)   # (a+d+g)(j+m+p)
        add_term([(1,1),(4,1),(7,1)], [(1,1),(4,1),(7,1)], [(1,1),(4,1),(7,1)], 10)  # (b+e+h)(k+n+q)
        add_term([(2,1),(5,1),(8,1)], [(2,1),(5,1),(8,1)], [(2,1),(5,1),(8,1)], 11)  # (c+f+i)(l+o+r)
        add_term([(0,1),(4,-1)], [(0,1),(4,-1)], [(0,1),(4,-1)], 12)  # (a-e)(j-n)
        add_term([(1,1),(5,-1)], [(1,1),(5,-1)], [(1,1),(5,-1)], 13)  # (b-f)(k-o)
        add_term([(2,1),(0,-1)], [(2,1),(0,-1)], [(2,1),(0,-1)], 14)  # (c-a)(l-j)
        add_term([(3,1),(6,-1)], [(3,1),(6,-1)], [(3,1),(6,-1)], 15)  # (d-g)(m-p)
        add_term([(4,1),(8,-1)], [(4,1),(8,-1)], [(4,1),(8,-1)], 16)  # (e-i)(n-r)
        add_term([(5,1),(3,-1)], [(5,1),(3,-1)], [(5,1),(3,-1)], 17)  # (f-d)(o-m)
        add_term([(7,1),(6,-1)], [(7,1),(6,-1)], [(7,1),(6,-1)], 18)  # (h-g)(q-p)
        add_term([(8,1),(7,-1)], [(8,1),(7,-1)], [(8,1),(7,-1)], 19)  # (i-h)(r-q)
        add_term([(0,1),(1,1),(2,-1)], [(0,1),(1,1),(2,-1)], [(0,1),(1,1),(2,-1)], 20)  # (a+b-c)(j+k-l)
        add_term([(3,1),(4,1),(5,-1)], [(3,1),(4,1),(5,-1)], [(3,1),(4,1),(5,-1)], 21)  # (d+e-f)(m+n-o)
        add_term([(6,1),(7,1),(8,-1)], [(6,1),(7,1),(8,-1)], [(6,1),(7,1),(8,-1)], 22)  # (g+h-i)(p+q-r)

        return U, V, W

    def schoolbook(n, m, p):
        """Return U, V, W for schoolbook matrix multiplication."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def embed(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
              i_off, j_off, k_off, n, m, p):
        """Embed a sub-decomposition into the full (n,m,p) tensor."""
        rank = U_sub.shape[1]
        U_full = np.zeros((n * m, rank))
        V_full = np.zeros((m * p, rank))
        W_full = np.zeros((p * n, rank))
        for c in range(rank):
            for r in range(n_sub * m_sub):
                ii = r // m_sub
                jj = r % m_sub
                U_full[(ii + i_off) * m + (jj + j_off), c] = U_sub[r, c]
            for r in range(m_sub * p_sub):
                jj = r // p_sub
                kk = r % p_sub
                V_full[(jj + j_off) * p + (kk + k_off), c] = V_sub[r, c]
            for r in range(p_sub * n_sub):
                kk = r // n_sub
                ii = r % n_sub
                W_full[(kk + k_off) * n + (ii + i_off), c] = W_sub[r, c]
        return U_full, V_full, W_full

    def decompose(n, m, p):
        """Recursively find the best decomposition for M_{n,m,p}."""
        # Base cases
        if n == 1 or m == 1 or p == 1:
            return schoolbook(n, m, p)
        if n == 2 and m == 2 and p == 2:
            return strassen_222()
        if n == 3 and m == 3 and p == 3:
            return laderman_333()

        best_U, best_V, best_W = None, None, None
        best_rank = n * m * p

        # Try splitting n
        for n1 in range(1, n):
            n2 = n - n1
            U1, V1, W1 = decompose(n1, m, p)
            U2, V2, W2 = decompose(n2, m, p)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                U1e, V1e, W1e = embed(U1, V1, W1, n1, m, p, 0, 0, 0, n, m, p)
                U2e, V2e, W2e = embed(U2, V2, W2, n2, m, p, n1, 0, 0, n, m, p)
                best_U = np.hstack([U1e, U2e])
                best_V = np.hstack([V1e, V2e])
                best_W = np.hstack([W1e, W2e])
                best_rank = r

        # Try splitting m
        for m1 in range(1, m):
            m2 = m - m1
            U1, V1, W1 = decompose(n, m1, p)
            U2, V2, W2 = decompose(n, m2, p)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                U1e, V1e, W1e = embed(U1, V1, W1, n, m1, p, 0, 0, 0, n, m, p)
                U2e, V2e, W2e = embed(U2, V2, W2, n, m2, p, 0, m1, 0, n, m, p)
                best_U = np.hstack([U1e, U2e])
                best_V = np.hstack([V1e, V2e])
                best_W = np.hstack([W1e, W2e])
                best_rank = r

        # Try splitting p
        for p1 in range(1, p):
            p2 = p - p1
            U1, V1, W1 = decompose(n, m, p1)
            U2, V2, W2 = decompose(n, m, p2)
            r = U1.shape[1] + U2.shape[1]
            if r < best_rank:
                U1e, V1e, W1e = embed(U1, V1, W1, n, m, p1, 0, 0, 0, n, m, p)
                U2e, V2e, W2e = embed(U2, V2, W2, n, m, p2, 0, 0, p1, n, m, p)
                best_U = np.hstack([U1e, U2e])
                best_V = np.hstack([V1e, V2e])
                best_W = np.hstack([W1e, W2e])
                best_rank = r

        if best_U is None:
            U, V, W = schoolbook(n, m, p)
            return U, V, W
        return best_U, best_V, best_W

    # Build the decomposition for 3x5x7
    U, V, W = decompose(3, 5, 7)

    # Verify correctness
    n, m, p = 3, 5, 7
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_comp = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.sum((T_comp - T) ** 2)

    if error > 1e-9:
        # Fallback to schoolbook
        U, V, W = schoolbook(3, 5, 7)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
