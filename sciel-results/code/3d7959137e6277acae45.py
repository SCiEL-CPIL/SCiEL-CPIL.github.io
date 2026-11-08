import sys
import numpy as np

# Increase recursion limit if necessary
sys.setrecursionlimit(10000)

DIMENSIONS = [3, 4, 7]

# Memoization for rank calculation
rank_cache = {}

def get_rank(n, m, p):
    if (n, m, p) in rank_cache:
        return rank_cache[(n, m, p)]
    
    # Base cases
    if n == 0 or m == 0 or p == 0:
        rank_cache[(n, m, p)] = 0
        return 0
        
    if n == 1:
        r = m * p
        rank_cache[(n, m, p)] = r
        return r
    if m == 1:
        r = n * p
        rank_cache[(n, m, p)] = r
        return r
    if p == 1:
        r = n * m
        rank_cache[(n, m, p)] = r
        return r
        
    if n == 2 and m == 2 and p == 2:
        rank_cache[(n, m, p)] = 7
        return 7

    best = float('inf')
    
    # Try splitting n
    if n >= 2:
        r1 = get_rank(n-2, m, p) + get_rank(2, m, p)
        if r1 < best: best = r1
    if n >= 1:
        r2 = get_rank(n-1, m, p) + get_rank(1, m, p)
        if r2 < best: best = r2
        
    # Try splitting m
    if m >= 2:
        r3 = get_rank(n, m-2, p) + get_rank(n, 2, p)
        if r3 < best: best = r3
    if m >= 1:
        r4 = get_rank(n, m-1, p) + get_rank(n, 1, p)
        if r4 < best: best = r4
        
    # Try splitting p
    if p >= 2:
        r5 = get_rank(n, m, p-2) + get_rank(n, m, 2)
        if r5 < best: best = r5
    if p >= 1:
        r6 = get_rank(n, m, p-1) + get_rank(n, m, 1)
        if r6 < best: best = r6
        
    rank_cache[(n, m, p)] = best
    return best

# We also need to store the split decision to construct the matrices
split_cache = {}

def get_split(n, m, p):
    if (n, m, p) in split_cache:
        return split_cache[(n, m, p)]
        
    if n == 0 or m == 0 or p == 0:
        split_cache[(n, m, p)] = None
        return None
        
    if n == 1:
        split_cache[(n, m, p)] = ('base_n1', m, p)
        return split_cache[(n, m, p)]
    if m == 1:
        split_cache[(n, m, p)] = ('base_m1', n, p)
        return split_cache[(n, m, p)]
    if p == 1:
        split_cache[(n, m, p)] = ('base_p1', n, m)
        return split_cache[(n, m, p)]
        
    if n == 2 and m == 2 and p == 2:
        split_cache[(n, m, p)] = ('strassen', 2, 2, 2)
        return split_cache[(n, m, p)]

    best = float('inf')
    best_split = None
    
    if n >= 2:
        r = get_rank(n-2, m, p) + get_rank(2, m, p)
        if r < best:
            best = r
            best_split = ('n2', n-2, m, p, 2, m, p)
    if n >= 1:
        r = get_rank(n-1, m, p) + get_rank(1, m, p)
        if r < best:
            best = r
            best_split = ('n1', n-1, m, p, 1, m, p)
            
    if m >= 2:
        r = get_rank(n, m-2, p) + get_rank(n, 2, p)
        if r < best:
            best = r
            best_split = ('m2', n, m-2, p, n, 2, p)
    if m >= 1:
        r = get_rank(n, m-1, p) + get_rank(n, 1, p)
        if r < best:
            best = r
            best_split = ('m1', n, m-1, p, n, 1, p)
            
    if p >= 2:
        r = get_rank(n, m, p-2) + get_rank(n, m, 2)
        if r < best:
            best = r
            best_split = ('p2', n, m, p-2, n, m, 2)
    if p >= 1:
        r = get_rank(n, m, p-1) + get_rank(n, m, 1)
        if r < best:
            best = r
            best_split = ('p1', n, m, p-1, n, m, 1)
            
    split_cache[(n, m, p)] = best_split
    return best_split

# Helper to construct base cases
def construct_base_n1(m, p):
    # Rank m*p
    # U: (1*m, m*p), V: (m*p, m*p), W: (p*1, m*p)
    # U[i*m+j, c] = 1 if c == i*m+j
    # V[j*p+k, c] = 1 if c == j*p+k
    # W[k*n+i, c] = 1 if c == k*n+i (n=1)
    rank = m * p
    U = np.zeros((m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p, rank))
    
    for j in range(m):
        for k in range(p):
            c = j * p + k
            U[j, c] = 1
            V[j * p + k, c] = 1
            W[k, c] = 1
    return U, V, W

def construct_base_m1(n, p):
    # Rank n*p
    rank = n * p
    U = np.zeros((n, rank))
    V = np.zeros((p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            U[i, c] = 1
            V[k, c] = 1
            W[k * n + i, c] = 1
    return U, V, W

def construct_base_p1(n, m):
    # Rank n*m
    rank = n * m
    U = np.zeros((n * m, rank))
    V = np.zeros((m, rank))
    W = np.zeros((n, rank))
    
    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j, c] = 1
            V[j, c] = 1
            W[i, c] = 1
    return U, V, W

def construct_strassen():
    # 2x2x2 Strassen
    # Standard Strassen decomposition
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # U indices: A11, A12, A21, A22
    # V indices: B11, B12, B21, B22
    # W indices: C11, C12, C21, C22
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C11, C22
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1 # C21
    W[3, 1] = -1 # C22
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1 # C12
    W[3, 2] = 1 # C22
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1 # C11
    W[2, 3] = 1 # C21
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1 # C11
    W[1, 4] = 1 # C12
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C22
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C11
    
    return U, V, W

def construct(n, m, p):
    # Get the split
    split = get_split(n, m, p)
    
    if split is None:
        # Empty
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
        
    if split[0] == 'base_n1':
        return construct_base_n1(split[1], split[2])
    elif split[0] == 'base_m1':
        return construct_base_m1(split[1], split[2])
    elif split[0] == 'base_p1':
        return construct_base_p1(split[1], split[2])
    elif split[0] == 'strassen':
        return construct_strassen()
        
    # Recursive splits
    # Format: ('n2', n1, m1, p1, n2, m2, p2)
    type = split[0]
    
    if type == 'n2':
        n1, m1, p1 = split[1], split[2], split[3]
        n2, m2, p2 = split[4], split[5], split[6]
        U1, V1, W1 = construct(n1, m1, p1)
        U2, V2, W2 = construct(n2, m2, p2)
        
        # Combine
        # U: (n*m, rank)
        # U1 is (n1*m, rank1), U2 is (n2*m, rank2)
        # We need to place U1 and U2 in the correct rows.
        # Rows 0..n1*m-1 come from U1. Rows n1*m..n*m-1 come from U2.
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # U
        U[:n1*m, :rank1] = U1
        U[n1*m:, rank1:] = U2
        
        # V
        V[:, :rank1] = V1
        V[:, rank1:] = V2
        
        # W
        # W1 is (p*n1, rank1), W2 is (p*n2, rank2)
        # W rows: 0..p*n1-1 from W1, p*n1..p*n-1 from W2
        W[:p*n1, :rank1] = W1
        W[p*n1:, rank1:] = W2
        
        return U, V, W
        
    elif type == 'n1':
        n1, m1, p1 = split[1], split[2], split[3]
        n2, m2, p2 = split[4], split[5], split[6]
        U1, V1, W1 = construct(n1, m1, p1)
        U2, V2, W2 = construct(n2, m2, p2)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        U[:n1*m, :rank1] = U1
        U[n1*m:, rank1:] = U2
        
        V[:, :rank1] = V1
        V[:, rank1:] = V2
        
        W[:p*n1, :rank1] = W1
        W[p*n1:, rank1:] = W2
        
        return U, V, W
        
    elif type == 'm2':
        n1, m1, p1 = split[1], split[2], split[3]
        n2, m2, p2 = split[4], split[5], split[6]
        U1, V1, W1 = construct(n1, m1, p1)
        U2, V2, W2 = construct(n2, m2, p2)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # U: cols 0..m1-1 from U1, m1..m-1 from U2
        U[:, :m1] = U1
        U[:, m1:] = U2
        
        # V: rows 0..m1*p-1 from V1, m1*p..m*p-1 from V2
        V[:m1*p, :rank1] = V1
        V[m1*p:, rank1:] = V2
        
        # W
        W[:, :rank1] = W1
        W[:, rank1:] = W2
        
        return U, V, W
        
    elif type == 'm1':
        n1, m1, p1 = split[1], split[2], split[3]
        n2, m2, p2 = split[4], split[5], split[6]
        U1, V1, W1 = construct(n1, m1, p1)
        U2, V2, W2 = construct(n2, m2, p2)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        U[:, :m1] = U1
        U[:, m1:] = U2
        
        V[:m1*p, :rank1] = V1
        V[m1*p:, rank1:] = V2
        
        W[:, :rank1] = W1
        W[:, rank1:] = W2
        
        return U, V, W
        
    elif type == 'p2':
        n1, m1, p1 = split[1], split[2], split[3]
        n2, m2, p2 = split[4], split[5], split[6]
        U1, V1, W1 = construct(n1, m1, p1)
        U2, V2, W2 = construct(n2, m2, p2)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # U
        U[:, :rank1] = U1
        U[:, rank1:] = U2
        
        # V: cols 0..p1-1 from V1, p1..p-1 from V2
        V[:, :p1] = V1
        V[:, p1:] = V2
        
        # W: rows 0..p1*n-1 from W1, p1*n..p*n-1 from W2
        W[:p1*n, :rank1] = W1
        W[p1*n:, rank1:] = W2
        
        return U, V, W
        
    elif type == 'p1':
        n1, m1, p1 = split[1], split[2], split[3]
        n2, m2, p2 = split[4], split[5], split[6]
        U1, V1, W1 = construct(n1, m1, p1)
        U2, V2, W2 = construct(n2, m2, p2)
        
        rank1 = U1.shape[1]
        rank2 = U2.shape[1]
        total_rank = rank1 + rank2
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        U[:, :rank1] = U1
        U[:, rank1:] = U2
        
        V[:, :p1] = V1
        V[:, p1:] = V2
        
        W[:p1*n, :rank1] = W1
        W[p1*n:, rank1:] = W2
        
        return U, V, W

    return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))

# Main entry point
def construct():
    n, m, p = DIMENSIONS
    U, V, W = construct(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
