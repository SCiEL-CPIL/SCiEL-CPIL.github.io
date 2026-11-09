import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    deadline = start_time + 140

    def verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i*m+j, j*p+k, k*n+i] += u[i,j] * v[j,k] * w[k,i]
        
        Texp = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Texp[i*m+j, j*p+k, k*n+i] = 1.0
        
        return np.allclose(T, Texp, atol=1e-9)

    def build_rank_36():
        """4 Strassen 2x2x2 blocks + 8 schoolbook for col 4"""
        strassen_U = [[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
        strassen_V = [[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
        strassen_W = [[1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],[-1,0,1,0],[0,0,0,1],[1,0,0,0]]
        
        rank = 4*7 + 8
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        
        col = 0
        for j0 in (0, 2):
            for k0 in (0, 2):
                for r in range(7):
                    for li in range(2):
                        for lj in range(2):
                            U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                    for lj in range(2):
                        for lk in range(2):
                            V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                    for lk in range(2):
                        for li in range(2):
                            W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                    col += 1
        
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j*p+4][col] = 1
                W[4*n+i][col] = 1
                col += 1
        
        assert col == rank
        return U, V, W

    def build_rank_32():
        """
        Try: 2 blocks of M_{2,2,5} each with rank 16.
        M_{2,2,5} = M_{2,2,2} (Strassen, rank 7) + M_{2,2,3} (rank 9?)
        But M_{2,2,3} = M_{2,2,2} (rank 7) + M_{2,2,1} (rank 4) = 11, so total 18.
        Alternatively: M_{2,2,5} with 2 Strassen + 4 schoolbook = 18.
        2*18 = 36, not 32.
        
        Let me try a different approach: 4 blocks where each handles 2x2x5 but with overlap?
        No, let's try: 2 Strassen for k=0,1 and 2 Strassen for k=2,3, and 4 schoolbook for k=4.
        That's 4*7 + 4 = 32!
        
        Wait, that's what I have: 4 Strassen blocks (for j-blocks and k-blocks 0-1, 2-3) = 28, plus 8 schoolbook = 36.
        
        The issue is the schoolbook part: for k=4, I need 2*4=8 terms.
        Can I reduce those 8 terms?
        
        For k=4: T_{(i,j),(j,4),(4,i)} = 1 for i in {0,1}, j in {0,1,2,3}.
        That's 8 entries. Each has distinct (i,j) and (4,i) and (j,4) indices... 
        Actually (j,4) varies with j, (4,i) varies with i. The 8 terms are:
        (i=0,j=0): e_{(0,0)} ⊗ e_{(0,4)} ⊗ e_{(4,0)}
        (i=0,j=1): e_{(0,1)} ⊗ e_{(1,4)} ⊗ e_{(4,0)}
        (i=0,j=2): e_{(0,2)} ⊗ e_{(2,4)} ⊗ e_{(4,0)}
        (i=0,j=3): e_{(0,3)} ⊗ e_{(3,4)} ⊗ e_{(4,0)}
        (i=1,j=0): e_{(1,0)} ⊗ e_{(0,4)} ⊗ e_{(4,1)}
        (i=1,j=1): e_{(1,1)} ⊗ e_{(1,4)} ⊗ e_{(4,1)}
        (i=1,j=2): e_{(1,2)} ⊗ e_{(2,4)} ⊗ e_{(4,1)}
        (i=1,j=3): e_{(1,3)} ⊗ e_{(3,4)} ⊗ e_{(4,1)}
        
        These 8 terms have linearly independent U, V, W vectors? 
        U vectors: e_{(0,0)}, e_{(0,1)}, e_{(0,2)}, e_{(0,3)}, e_{(1,0)}, e_{(1,1)}, e_{(1,2)}, e_{(1,3)} - all independent.
        So rank of this sub-tensor is 8. Can't reduce.
        
        So 36 seems to be the natural block decomposition. Let me try to find a non-block decomposition.
        """
        return None

    # Build rank-36
    U, V, W = build_rank_36()
    
    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n*m*p
        U_s = [[0]*rank for _ in range(n*m)]
        V_s = [[0]*rank for _ in range(m*p)]
        W_s = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][col] = 1
                    V_s[j*p+k][col] = 1
                    W_s[k*n+i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}

    return {"U": U, "V": V, "W": W}
