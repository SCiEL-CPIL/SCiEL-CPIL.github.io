"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Splits for n, m, p
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    n_starts = [0, 2]
    m_starts = [0, 2, 4]
    p_starts = [0, 2, 4]
    
    # Calculate total rank
    total_rank = 0
    for i in range(len(n_splits)):
        for j in range(len(m_splits)):
            for k in range(len(p_splits)):
                ns = n_splits[i]
                ms = m_splits[j]
                ps = p_splits[k]
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7
                else:
                    total_rank += ns * ms * ps
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i in range(len(n_splits)):
        n0 = n_starts[i]
        ns = n_splits[i]
        for j in range(len(m_splits)):
            m0 = m_starts[j]
            ms = m_splits[j]
            for k in range(len(p_splits)):
                p0 = p_starts[k]
                ps = p_splits[k]
                
                if ns == 2 and ms == 2 and ps == 2:
                    # Strassen 2x2x2
                    # Define local to global mapping helpers
                    def u_row(a, b): return (n0 + a) * m + (m0 + b)
                    def v_row(b, c): return (m0 + b) * p + (p0 + c)
                    def w_row(c, a): return (p0 + c) * n + (n0 + a)
                    
                    # M1 = (A00 + A11)(B00 + B11)
                    U[u_row(0,0)][col] += 1
                    U[u_row(1,1)][col] += 1
                    V[v_row(0,0)][col] += 1
                    V[v_row(1,1)][col] += 1
                    W[w_row(0,0)][col] += 1
                    W[w_row(1,1)][col] += 1
                    col += 1
                    
                    # M2 = (A10 + A11)B00
                    U[u_row(1,0)][col] += 1
                    U[u_row(1,1)][col] += 1
                    V[v_row(0,0)][col] += 1
                    W[w_row(0,1)][col] += 1
                    W[w_row(1,1)][col] += -1
                    col += 1
                    
                    # M3 = A00(B01 - B11)
                    U[u_row(0,0)][col] += 1
                    V[v_row(0,1)][col] += 1
                    V[v_row(1,1)][col] += -1
                    W[w_row(1,0)][col] += 1
                    W[w_row(1,1)][col] += 1
                    col += 1
                    
                    # M4 = A11(B10 - B00)
                    U[u_row(1,1)][col] += 1
                    V[v_row(1,0)][col] += 1
                    V[v_row(0,0)][col] += -1
                    W[w_row(0,0)][col] += 1
                    W[w_row(1,1)][col] += 1
                    col += 1
                    
                    # M5 = (A00 + A10)B11
                    U[u_row(0,0)][col] += 1
                    U[u_row(1,0)][col] += 1
                    V[v_row(1,1)][col] += 1
                    W[w_row(0,0)][col] += -1
                    W[w_row(1,0)][col] += 1
                    col += 1
                    
                    # M6 = (A10 - A11)(B00 + B10)
                    U[u_row(1,0)][col] += 1
                    U[u_row(1,1)][col] += -1
                    V[v_row(0,0)][col] += 1
                    V[v_row(1,0)][col] += 1
                    W[w_row(1,1)][col] += 1
                    col += 1
                    
                    # M7 = (A01 - A11)(B10 + B11)
                    U[u_row(0,1)][col] += 1
                    U[u_row(1,1)][col] += -1
                    V[v_row(1,0)][col] += 1
                    V[v_row(1,1)][col] += 1
                    W[w_row(0,0)][col] += 1
                    W[w_row(1,0)][col] += -1
                    col += 1
                    
                else:
                    # Schoolbook
                    for a in range(ns):
                        for b in range(ms):
                            for c in range(ps):
                                I = n0 + a
                                J = m0 + b
                                K = p0 + c
                                U[I * m + J][col] = 1
                                V[J * p + K][col] = 1
                                W[K * n + I][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
