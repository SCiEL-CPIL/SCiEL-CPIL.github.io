import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# Memoization for rank calculation
_rank_cache = {}

def get_rank(n, m, p):
    if (n, m, p) in _rank_cache:
        return _rank_cache[(n, m, p)]
    
    # Base case: Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        _rank_cache[(n, m, p)] = 7
        return 7
        
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_rank = schoolbook_rank
    
    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        r1 = get_rank(n, m1, p)
        r2 = get_rank(n, m2, p)
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            
    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        r1 = get_rank(n, m, p1)
        r2 = get_rank(n, m, p2)
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            
    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        r1 = get_rank(n1, m, p)
        r2 = get_rank(n2, m, p)
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            
    _rank_cache[(n, m, p)] = best_rank
    return best_rank

def strassen_decomp():
    # Returns U, V, W for 2x2x2
    # Indexing:
    # U[i*2+j, r]
    # V[j*2+k, r]
    # W[k*2+i, r]
    
    # Strassen M1..M7
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # U rows: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V rows: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # W rows: (0,0)=C[0,0], (0,1)=C[0,1], (1,0)=C[1,0], (1,1)=C[1,1]
    # Note: W[k*n+i]. n=2.
    # k=0,i=0 -> C[0,0]
    # k=0,i=1 -> C[0,1]
    # k=1,i=0 -> C[1,0]
    # k=1,i=1 -> C[1,1]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a+d)(e+h)
    # U: a(1), d(1) -> rows 0, 3
    # V: e(1), h(1) -> rows 0, 3
    # W: C00(1), C01(-1), C10(-1), C11(1) -> rows 0,1,2,3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = -1; W[2, 0] = -1; W[3, 0] = 1
    
    # M2: (c+d)e
    # U: c(1), d(1) -> rows 2, 3
    # V: e(1) -> row 0
    # W: C10(1), C11(-1) -> rows 2, 3
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: a(f-h)
    # U: a(1) -> row 0
    # V: f(1), h(-1) -> rows 1, 3
    # W: C01(1), C11(1) -> rows 1, 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: d(g-f)
    # U: d(1) -> row 3
    # V: g(1), f(-1) -> rows 2, 1
    # W: C00(1), C10(1) -> rows 0, 2
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (a+b)h
    # U: a(1), b(1) -> rows 0, 1
    # V: h(1) -> row 3
    # W: C00(-1), C01(1) -> rows 0, 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: (c-a)(e+f)
    # U: c(1), a(-1) -> rows 2, 0
    # V: e(1), f(1) -> rows 0, 1
    # W: C11(1) -> row 3
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (b-d)(g+h)
    # U: b(1), d(-1) -> rows 1, 3
    # V: g(1), h(1) -> rows 2, 3
    # W: C00(1), C10(-1) -> rows 0, 2
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[2, 6] = -1
    
    return U, V, W

def schoolbook_decomp(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def construct_decomp(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return strassen_decomp()
    
    best_rank = n * m * p
    best_split = None
    
    # Check m split
    for m1 in range(1, m):
        m2 = m - m1
        r = get_rank(n, m1, p) + get_rank(n, m2, p)
        if r < best_rank:
            best_rank = r
            best_split = ('m', m1, m2)
            
    # Check p split
    for p1 in range(1, p):
        p2 = p - p1
        r = get_rank(n, m, p1) + get_rank(n, m, p2)
        if r < best_rank:
            best_rank = r
            best_split = ('p', p1, p2)
            
    # Check n split
    for n1 in range(1, n):
        n2 = n - n1
        r = get_rank(n1, m, p) + get_rank(n2, m, p)
        if r < best_rank:
            best_rank = r
            best_split = ('n', n1, n2)
            
    if best_split is None:
        return schoolbook_decomp(n, m, p)
        
    split_type, d1, d2 = best_split
    
    if split_type == 'm':
        U1, V1, W1 = construct_decomp(n, d1, p)
        U2, V2, W2 = construct_decomp(n, d2, p)
        
        # Combine
        r1, r2 = U1.shape[1], U2.shape[1]
        U = np.zeros((n * (d1 + d2), r1 + r2))
        V = np.zeros(((d1 + d2) * p, r1 + r2))
        W = np.zeros((p * n, r1 + r2))
        
        # U1 covers first n*d1 rows
        U[:n*d1, :r1] = U1
        # U2 covers next n*d2 rows
        U[n*d1:, r1:] = U2
        
        # V1 covers first d1*p rows
        V[:d1*p, :r1] = V1
        # V2 covers next d2*p rows
        V[d1*p:, r1:] = V2
        
        # W is horizontal concat
        W[:, :r1] = W1
        W[:, r1:] = W2
        
        return U, V, W
        
    elif split_type == 'p':
        U1, V1, W1 = construct_decomp(n, m, d1)
        U2, V2, W2 = construct_decomp(n, m, d2)
        
        r1, r2 = U1.shape[1], U2.shape[1]
        U = np.zeros((n * m, r1 + r2))
        V = np.zeros((m * (d1 + d2), r1 + r2))
        W = np.zeros(((d1 + d2) * n, r1 + r2))
        
        # U is horizontal concat
        U[:, :r1] = U1
        U[:, r1:] = U2
        
        # V1 covers first m*d1 rows
        V[:m*d1, :r1] = V1
        # V2 covers next m*d2 rows
        V[m*d1:, r1:] = V2
        
        # W1 covers first d1*n rows
        W[:d1*n, :r1] = W1
        # W2 covers next d2*n rows
        W[d1*n:, r1:] = W2
        
        return U, V, W
        
    elif split_type == 'n':
        U1, V1, W1 = construct_decomp(d1, m, p)
        U2, V2, W2 = construct_decomp(d2, m, p)
        
        r1, r2 = U1.shape[1], U2.shape[1]
        U = np.zeros(((d1 + d2) * m, r1 + r2))
        V = np.zeros((m * p, r1 + r2))
        W = np.zeros((p * (d1 + d2), r1 + r2))
        
        # U1 covers first d1*m rows
        U[:d1*m, :r1] = U1
        # U2 covers next d2*m rows
        U[d1*m:, r1:] = U2
        
        # V is horizontal concat
        V[:, :r1] = V1
        V[:, r1:] = V2
        
        # W1 covers first p*d1 rows
        W[:p*d1, :r1] = W1
        # W2 covers next p*d2 rows
        W[p*d1:, r1:] = W2
        
        return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Calculate rank first to populate cache
    target_rank = get_rank(n, m, p)
    
    # Construct
    U, V, W = construct_decomp(n, m, p)
    
    # Convert to lists of lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
