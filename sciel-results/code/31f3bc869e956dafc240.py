"""Matrix multiplication 3,5,7 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 7
    # U: (n*m, rank) = (15, rank)
    # V: (m*p, rank) = (35, rank)
    # W: (p*n, rank) = (21, rank)
    # W index convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    terms = []  # (u_idx, v_idx, w_idx, coeff)
    
    def add_term(u, v, w, c=1):
        if c != 0:
            terms.append((u, v, w, c))
    
    # Split m=5 into m1=2 and m2=3
    # A (3x5) = [A1 (3x2), A2 (3x3)]
    # B (5x7) = [B1 (2x7); B2 (3x7)]
    # C = A1*B1 + A2*B2
    
    # --- Part 1: A1 (3x2) * B1 (2x7) ---
    # Split p=7 into 3+2+2
    # Sub 1.1: 3x2x3 schoolbook (rank 18)
    for i in range(3):
        for j in range(2):
            for k in range(3):
                u = i * 5 + j
                v = j * 7 + k
                w = k * 3 + i
                add_term(u, v, w, 1)
    
    # Sub 1.2: 3x2x2 schoolbook (rank 6)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u = i * 5 + j
                v = j * 7 + (3 + k)
                w = (3 + k) * 3 + i
                add_term(u, v, w, 1)
    
    # Sub 1.3: 3x2x2 schoolbook (rank 6)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                u = i * 5 + j
                v = j * 7 + (5 + k)
                w = (5 + k) * 3 + i
                add_term(u, v, w, 1)
    
    # --- Part 2: A2 (3x3) * B2 (3x7) ---
    # Split p=7 into 3+2+2
    # Sub 2.1: 3x3x3 schoolbook (rank 27)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = i * 5 + (2 + j)
                v = (2 + j) * 7 + k
                w = k * 3 + i
                add_term(u, v, w, 1)
    
    # Sub 2.2: 3x3x2 schoolbook (rank 18)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                u = i * 5 + (2 + j)
                v = (2 + j) * 7 + (3 + k)
                w = (3 + k) * 3 + i
                add_term(u, v, w, 1)
    
    # Sub 2.3: 3x3x2 schoolbook (rank 18)
    for i in range(3):
        for j in range(3):
            for k in range(2):
                u = i * 5 + (2 + j)
                v = (2 + j) * 7 + (5 + k)
                w = (5 + k) * 3 + i
                add_term(u, v, w, 1)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u, v, w, coeff = terms[c]
        U[u][c] = coeff
        V[v][c] = coeff
        W[w][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
