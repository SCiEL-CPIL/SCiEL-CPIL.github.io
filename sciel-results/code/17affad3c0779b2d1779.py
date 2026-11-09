"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    deadline = time.time() + 140
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def make_u(entries):
        vec = [0] * (n * m)
        for (i, j), val in entries.items():
            vec[i * m + j] = val
        return vec
    
    def make_v(entries):
        vec = [0] * (m * p)
        for (j, k), val in entries.items():
            vec[j * p + k] = val
        return vec
    
    def make_w(entries):
        vec = [0] * (p * n)
        for (k, i), val in entries.items():
            vec[k * n + i] = val
        return vec
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # C[i0,k0] += M1, C[i1,k1] += M1
        U_cols.append(make_u({(i0,j0): 1, (i1,j1): 1}))
        V_cols.append(make_v({(j0,k0): 1, (j1,k1): 1}))
        W_cols.append(make_w({(k0,i0): 1, (k1,i1): 1}))
        
        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        # C[i0,k1] += M2, C[i1,k0] += M2
        U_cols.append(make_u({(i1,j0): 1, (i1,j1): 1}))
        V_cols.append(make_v({(j0,k0): 1}))
        W_cols.append(make_w({(k1,i0): 1, (k0,i1): 1}))
        
        # M3 = A[i0,j0]*(B[j1,k0]-B[j0,k0])
        # C[i0,k0] -= M3, C[i1,k0] += M3
        U_cols.append(make_u({(i0,j0): 1}))
        V_cols.append(make_v({(j1,k0): 1, (j0,k0): -1}))
        W_cols.append(make_w({(k0,i0): -1, (k0,i1): 1}))
        
        # M4 = A[i1,j1]*(B[j0,k1]-B[j0,k0])
        # C[i0,k0] -= M4, C[i1,k1] += M4
        U_cols.append(make_u({(i1,j1): 1}))
        V_cols.append(make_v({(j0,k1): 1, (j0,k0): -1}))
        W_cols.append(make_w({(k0,i0): -1, (k1,i1): 1}))
        
        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        # C[i0,k1] += M5, C[i1,k1] -= M5
        U_cols.append(make_u({(i0,j0): 1, (i0,j1): 1}))
        V_cols.append(make_v({(j1,k1): 1}))
        W_cols.append(make_w({(k1,i0): 1, (k1,i1): -1}))
        
        # M6 = (A[i1,j0]-A[i0,j0])*(B[j0,k0]+B[j0,k1])
        # C[i0,k0] += M6, C[i1,k0] -= M6
        U_cols.append(make_u({(i1,j0): 1, (i0,j0): -1}))
        V_cols.append(make_v({(j0,k0): 1, (j0,k1): 1}))
        W_cols.append(make_w({(k0,i0): 1, (k0,i1): -1}))
        
        # M7 = (A[i1,j1]-A[i0,j1])*(B[j1,k0]+B[j1,k1])
        # C[i0,k1] -= M7
        # Wait, let me recheck. C01 = M2+M5-M7. So M7 contributes -1 to C[i0,k1].
        # But does M7 contribute to any other C entry? No, only C01.
        U_cols.append(make_u({(i1,j1): 1, (i0,j1): -1}))
        V_cols.append(make_v({(j1,k0): 1, (j1,k1): 1}))
        W_cols.append(make_w({(k1,i0): -1}))
    
    def add_schoolbook(i, j, k):
        U_cols.append(make_u({(i,j): 1}))
        V_cols.append(make_v({(j,k): 1}))
        W_cols.append(make_w({(k,i): 1}))
    
    # Strassen blocks
    strassen_blocks = []
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3)]:
            strassen_blocks.append((0, 1, j0, j1, k0, k1))
    
    covered = set()
    for (i0, i1, j0, j1, k0, k1) in strassen_blocks:
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    covered.add((i, j, k))
    
    for block in strassen_blocks:
        add_strassen_block(*block)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_schoolbook(i, j, k)
    
    rank = len(U_cols)
    
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
