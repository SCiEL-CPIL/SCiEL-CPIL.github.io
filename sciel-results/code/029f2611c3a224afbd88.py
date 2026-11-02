"""Strassen-based decomposition for 2x5x5 matrix multiplication."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-49 decomposition using Strassen on 2x2x2 sub-blocks."""
    n, m, p = 2, 5, 5
    
    # We'll use 49 terms: 7 (Strassen) + 12 + 12 + 18 (schoolbook)
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]  # 10 x 49
    V = [[0] * rank for _ in range(m * p)]  # 25 x 49
    W = [[0] * rank for _ in range(p * n)]  # 10 x 49
    
    col = 0
    
    def add_term_u(u_idx, u_sign):
        U[u_idx][col] += u_sign
    
    def add_term_v(v_idx, v_sign):
        V[v_idx][col] += v_sign
    
    def add_term_w(w_idx, w_sign):
        W[w_idx][col] += w_sign
    
    def commit():
        nonlocal col
        col += 1
    
    # Strassen 2x2x2: j in {0,1}, k in {0,1}, i in {0,1}
    # U index: i*m+j = i*5+j
    # V index: j*p+k = j*5+k
    # W index: k*n+i = k*2+i
    
    # M1 = (A00+A11)*(B00+B11)
    # U: A00(0*5+0=0), A11(1*5+1=6)
    # V: B00(0*5+0=0), B11(1*5+1=6)
    # W: C00(0*2+0=0), C01(1*2+0=2), C10(0*2+1=1), C11(1*2+1=3)
    add_term_u(0, 1); add_term_u(6, 1)
    add_term_v(0, 1); add_term_v(6, 1)
    add_term_w(0, 1); add_term_w(2, 1); add_term_w(1, 1); add_term_w(3, 1)
    commit()
    
    # M2 = (A10+A11)*B00
    # U: A10(1*5+0=5), A11(1*5+1=6)
    # V: B00(0*5+0=0)
    # W: C10(0*2+1=1), C11(1*2+1=3)
    add_term_u(5, 1); add_term_u(6, 1)
    add_term_v(0, 1)
    add_term_w(1, 1); add_term_w(3, 1)
    commit()
    
    # M3 = A00*(B01-B11)
    # U: A00(0)
    # V: B01(0*5+1=1), B11(1*5+1=6) with sign -1
    # W: C01(1*2+0=2), C11(1*2+1=3)
    add_term_u(0, 1)
    add_term_v(1, 1); add_term_v(6, -1)
    add_term_w(2, 1); add_term_w(3, 1)
    commit()
    
    # M4 = A11*(B10-B00)
    # U: A11(6)
    # V: B10(1*5+0=5), B00(0) with sign -1
    # W: C00(0), C10(1)
    add_term_u(6, 1)
    add_term_v(5, 1); add_term_v(0, -1)
    add_term_w(0, 1); add_term_w(1, 1)
    commit()
    
    # M5 = (A00+A01)*B11
    # U: A00(0), A01(0*5+1=1)
    # V: B11(6)
    # W: C00(0), C10(1) with sign -1
    add_term_u(0, 1); add_term_u(1, 1)
    add_term_v(6, 1)
    add_term_w(0, -1); add_term_w(1, -1)
    commit()
    
    # M6 = (A10-A00)*(B00+B01)
    # U: A10(5), A00(0) with sign -1
    # V: B00(0), B01(1)
    # W: C11(3)
    add_term_u(5, 1); add_term_u(0, -1)
    add_term_v(0, 1); add_term_v(1, 1)
    add_term_w(3, 1)
    commit()
    
    # M7 = (A01-A11)*(B10+B11)
    # U: A01(1), A11(6) with sign -1
    # V: B10(5), B11(6)
    # W: C00(0), C11(3)
    add_term_u(1, 1); add_term_u(6, -1)
    add_term_v(5, 1); add_term_v(6, 1)
    add_term_w(0, 1); add_term_w(3, 1)
    commit()
    
    # Schoolbook: j in {0,1}, k in {2,3,4}, i in {0,1}
    # 2*2*3 = 12 terms
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_term_u(i*5+j, 1)
                add_term_v(j*5+k, 1)
                add_term_w(k*2+i, 1)
                commit()
    
    # Schoolbook: j in {2,3,4}, k in {0,1}, i in {0,1}
    # 2*3*2 = 12 terms
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                add_term_u(i*5+j, 1)
                add_term_v(j*5+k, 1)
                add_term_w(k*2+i, 1)
                commit()
    
    # Schoolbook: j in {2,3,4}, k in {2,3,4}, i in {0,1}
    # 2*3*3 = 18 terms
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                add_term_u(i*5+j, 1)
                add_term_v(j*5+k, 1)
                add_term_w(k*2+i, 1)
                commit()
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

