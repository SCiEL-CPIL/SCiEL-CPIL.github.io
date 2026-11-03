"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = 145.0

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_indices, v_indices, w_indices, u_vals, v_vals, w_vals):
        U_cols.append(dict(zip(u_indices, u_vals)))
        V_cols.append(dict(zip(v_indices, v_vals)))
        W_cols.append(dict(zip(w_indices, w_vals)))
    
    # Strassen blocks: i∈{0,1}, j∈{0,1} or {2,3}, k∈{0,1} or {2,3} or {4,5}
    i_blocks = [(0, 1)]
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    strassen_triples = set()
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            strassen_triples.add((i, j, k))
    
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    schoolbook_triples = [t for t in all_triples if t not in strassen_triples]
    
    # Add Strassen blocks
    for (i1, i2) in i_blocks:
        for (j1, j2) in j_blocks:
            for (k1, k2) in k_blocks:
                # 7 Strassen terms
                add_term([i1*m+j1, i2*m+j2], [j1*p+k1, j2*p+k2], [k1*n+i1, k2*n+i2], [1,1], [1,1], [1,1])
                add_term([i2*m+j1, i2*m+j2], [j1*p+k1], [k1*n+i2], [1,1], [1], [1])
                add_term([i1*m+j1], [j2*p+k1, j2*p+k2], [k1*n+i2], [1], [1,-1], [1])
                add_term([i2*m+j2], [j1*p+k2, j1*p+k1], [k2*n+i1], [1], [1,-1], [1])
                add_term([i1*m+j1, i1*m+j2], [j2*p+k2], [k2*n+i1], [1,1], [1], [1])
                add_term([i2*m+j1, i1*m+j1], [j1*p+k1, j1*p+k2], [k2*n+i2], [1,-1], [1,1], [1])
                add_term([i1*m+j2, i2*m+j2], [j2*p+k1, j2*p+k2], [k1*n+i1], [1,-1], [1,1], [1])
    
    # Add schoolbook terms
    for (i, j, k) in schoolbook_triples:
        add_term([i*m+j], [j*p+k], [k*n+i], [1], [1], [1])
    
    rank = len(U_cols)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

