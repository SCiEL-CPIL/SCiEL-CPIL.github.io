"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140

    n, m, p = DIMENSIONS  # 3, 5, 5

    i_parts = [1, 2]
    j_parts = [2, 1, 2]
    k_parts = [2, 1, 2]

    i_offsets = [0]
    for s in i_parts[:-1]:
        i_offsets.append(i_offsets[-1] + s)
    j_offsets = [0]
    for s in j_parts[:-1]:
        j_offsets.append(j_offsets[-1] + s)
    k_offsets = [0]
    for s in k_parts[:-1]:
        k_offsets.append(k_offsets[-1] + s)

    U_cols = []
    V_cols = []
    W_cols = []

    def add_schoolblock(i0, j0, k0, di, dj, dk):
        for a in range(di):
            for b in range(dj):
                for c in range(dk):
                    i = i0 + a
                    j = j0 + b
                    k = k0 + c
                    U_cols.append({(i, j): 1})
                    V_cols.append({(j, k): 1})
                    W_cols.append({(k, i): 1})

    def add_strassen(i0, j0, k0):
        # Local: a in {0,1} (i), b in {0,1} (j), c in {0,1} (k)
        # M1 = (A11+A22)(B11+B22)
        U_cols.append({(i0+0, j0+0): 1, (i0+1, j0+1): 1})
        V_cols.append({(j0+0, k0+0): 1, (j0+1, k0+1): 1})
        W_cols.append({(k0+0, i0+0): 1, (k0+1, i0+1): -1, (k0+1, i0+0): -1, (k0+0, i0+1): 1})
        
        # M2 = (A21+A22)B11
        U_cols.append({(i0+1, j0+0): 1, (i0+1, j0+1): 1})
        V_cols.append({(j0+0, k0+0): 1})
        W_cols.append({(k0+0, i0+1): 1, (k0+1, i0+1): -1})
        
        # M3 = A11(B12-B22)
        U_cols.append({(i0+0, j0+0): 1})
        V_cols.append({(j0+0, k0+1): 1, (j0+1, k0+1): -1})
        W_cols.append({(k0+1, i0+0): 1, (k0+1, i0+1): 1})
        
        # M4 = A22(B21-B11)
        U_cols.append({(i0+1, j0+1): 1})
        V_cols.append({(j0+1, k0+0): 1, (j0+0, k0+0): -1})
        W_cols.append({(k0+0, i0+1): 1, (k0+1, i0+1): 1})
        
        # M5 = (A11+A12)B22
        U_cols.append({(i0+0, j0+0): 1, (i0+0, j0+1): 1})
        V_cols.append({(j0+1, k0+1): 1})
        W_cols.append({(k0+1, i0+0): 1, (k0+0, i0+1): -1})
        
        # M6 = (A21-A11)(B11+B12)
        U_cols.append({(i0+1, j0+0): 1, (i0+0, j0+0): -1})
        V_cols.append({(j0+0, k0+0): 1, (j0+0, k0+1): 1})
        W_cols.append({(k0+0, i0+1): 1, (k0+1, i0+1): -1})
        
        # M7 = (A12-A22)(B21+B22)
        U_cols.append({(i0+0, j0+1): 1, (i0+1, j0+1): -1})
        V_cols.append({(j0+1, k0+0): 1, (j0+1, k0+1): 1})
        W_cols.append({(k0+0, i0+0): 1, (k0+1, i0+0): -1})

    for ii in range(len(i_parts)):
        for jj in range(len(j_parts)):
            for kk in range(len(k_parts)):
                di = i_parts[ii]
                dj = j_parts[jj]
                dk = k_parts[kk]
                i0 = i_offsets[ii]
                j0 = j_offsets[jj]
                k0 = k_offsets[kk]
                if di == 2 and dj == 2 and dk == 2:
                    add_strassen(i0, j0, k0)
                else:
                    add_schoolblock(i0, j0, k0, di, dj, dk)

    rank = len(U_cols)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for (i, j), val in U_cols[col].items():
            U[i * m + j][col] = val
        for (j, k), val in V_cols[col].items():
            V[j * p + k][col] = val
        for (k, i), val in W_cols[col].items():
            W[k * n + i][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
