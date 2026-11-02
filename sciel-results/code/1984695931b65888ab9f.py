"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=5, p=5
    
    # Partition indices for Strassen blocks
    # i (rows of A): {0,1} | {2}
    # j (cols of A / rows of B): {0,1} | {2,3} | {4}
    # k (cols of B): {0,1} | {2,3} | {4}
    
    strassen_blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3]),
    ]
    
    # Collect all (i,j,k) triples in Strassen blocks
    strassen_triples = set()
    for i_set, j_set, k_set in strassen_blocks:
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    strassen_triples.add((i, j, k))
    
    # Schoolbook triples
    schoolbook_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p) 
                          if (i, j, k) not in strassen_triples]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i_set, j_set, k_set in strassen_blocks:
        i0, i1 = i_set
        j0, j1 = j_set
        k0, k1 = k_set
        
        # M1 = (a_i0j0 + a_i1j1)(b_j0k0 + b_j1k1)
        U_cols.append({(i0, j0): 1, (i1, j1): 1})
        V_cols.append({(j0, k0): 1, (j1, k1): 1})
        W_cols.append({(k0, i0): 1, (k1, i1): 1, (k0, i1): -1, (k1, i0): 1})
        
        # M2 = (a_i1j0 + a_i1j1) b_j0k0
        U_cols.append({(i1, j0): 1, (i1, j1): 1})
        V_cols.append({(j0, k0): 1})
        W_cols.append({(k0, i1): 1})
        
        # M3 = a_i0j0 (b_j0k1 - b_j1k1)
        U_cols.append({(i0, j0): 1})
        V_cols.append({(j0, k1): 1, (j1, k1): -1})
        W_cols.append({(k1, i0): 1})
        
        # M4 = a_i1j1 (b_j1k0 - b_j0k0)
        U_cols.append({(i1, j1): 1})
        V_cols.append({(j1, k0): 1, (j0, k0): -1})
        W_cols.append({(k0, i1): 1, (k0, i0): 1})
        
        # M5 = (a_i0j0 + a_i0j1) b_j1k1
        U_cols.append({(i0, j0): 1, (i0, j1): 1})
        V_cols.append({(j1, k1): 1})
        W_cols.append({(k1, i0): 1, (k0, i0): -1})
        
        # M6 = (a_i1j0 - a_i0j0)(b_j0k0 + b_j0k1)
        U_cols.append({(i1, j0): 1, (i0, j0): -1})
        V_cols.append({(j0, k0): 1, (j0, k1): 1})
        W_cols.append({(k1, i1): 1})
        
        # M7 = (a_i0j1 - a_i1j1)(b_j1k0 + b_j1k1)
        U_cols.append({(i0, j1): 1, (i1, j1): -1})
        V_cols.append({(j1, k0): 1, (j1, k1): 1})
        W_cols.append({(k1, i0): 1, (k0, i1): -1})
        
        if time.time() > deadline:
            break
    
    for (i, j, k) in schoolbook_triples:
        U_cols.append({(i, j): 1})
        V_cols.append({(j, k): 1})
        W_cols.append({(k, i): 1})
    
    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for (i, j), val in U_cols[c].items():
            U[i * m + j][c] = val
        for (j, k), val in V_cols[c].items():
            V[j * p + k][c] = val
        for (k, i), val in W_cols[c].items():
            W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
