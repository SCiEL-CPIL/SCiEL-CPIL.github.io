import time

# Set a deadline to ensure we don't timeout, although this construction is O(1)
# We use a margin to be safe.
_DEADLINE = time.time() + 145

def construct():
    """
    Returns a rank-35 exact tensor decomposition for multiplying a 2x2 matrix 
    by a 2x10 matrix.
    
    The decomposition is constructed by applying Strassen's rank-7 algorithm 
    to each of the five 2x2 column blocks of the 2x10 matrix independently.
    
    Dimensions:
    n = 2 (rows of A, cols of C)
    m = 2 (cols of A, rows of B)
    p = 10 (cols of B, cols of C)
    
    U shape: (n*m, rank) = (4, 35)
    V shape: (m*p, rank) = (20, 35)
    W shape: (p*n, rank) = (20, 35)
    
    Indices:
    A is indexed by i (0..1), j (0..1). U[i*m + j, k]
    B is indexed by j (0..1), l (0..9). V[j*p + l, k]
    C is indexed by i (0..1), l (0..9). W[l*n + i, k]
    
    Note on W indexing: The problem statement says "output factor follows the 
    official trace convention W[k*n+i,r]". 
    Wait, let's re-read carefully: "W[k*n+i,r]".
    Usually, C has dimensions n x p. So index is (row, col) = (i, l).
    If the convention is W[col*n + row, rank] or W[row*p + col, rank]?
    The prompt says: "W[k*n+i,r]". This implies the first index is `k` and second is `i`?
    No, standard Python indexing is W[index1, index2].
    If W has shape (p*n, rank), the first dimension is p*n = 20.
    The indices for C are i in {0,1} and l in {0,..,9}.
    If the convention is W[l*n + i, r], then the index is l*2 + i.
    Let's check the seed code:
    W[4*b][col] = 1  -> b=0, index 0. l=0, i=0. 0*2+0=0.
    W[4*b+3][col] = 1 -> b=0, index 3. l=1, i=1? 1*2+1=3.
    So the seed uses W[l*n + i, r] where l is the block index (0..4) and within block, i is row.
    Actually, let's look at the seed code's W indices:
    For block b (0..4):
    Output columns are 2*b and 2*b+1.
    Row indices for C are 0 and 1.
    Seed:
    W[4*b][col] -> Corresponds to C[0, 2*b]?
    W[4*b+1][col] -> Corresponds to C[1, 2*b]?
    W[4*b+2][col] -> Corresponds to C[0, 2*b+1]?
    W[4*b+3][col] -> Corresponds to C[1, 2*b+1]?
    
    Let's verify the "official trace convention W[k*n+i,r]".
    If k is the column of C (0..9) and i is the row of C (0..1), then index = k*2 + i.
    For column 0 (2*b), row 0 (i=0): index 0*2+0 = 0. Matches W[4*b].
    For column 0 (2*b), row 1 (i=1): index 0*2+1 = 1. Matches W[4*b+1].
    For column 1 (2*b+1), row 0 (i=0): index 1*2+0 = 2. Matches W[4*b+2].
    For column 1 (2*b+1), row 1 (i=1): index 1*2+1 = 3. Matches W[4*b+3].
    
    So the mapping is:
    C[i, l] -> W[l*n + i, r]
    
    Strassen's Algorithm for A (2x2) * B (2x2) = C (2x2):
    Let A = [[a, b], [c, d]]
    Let B = [[e, f], [g, h]]
    
    M1 = (a + d)(e + h)
    M2 = (c + d)e
    M3 = a(f - h)
    M4 = d(g - e)
    M5 = (a + b)h
    M6 = (c - a)(e + f)
    M7 = (b - d)(g + h)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Let's map these to the decomposition U, V, W.
    
    For each block b (0..4):
    B block indices:
    j=0, l=2*b -> e
    j=0, l=2*b+1 -> f
    j=1, l=2*b -> g
    j=1, l=2*b+1 -> h
    
    A indices:
    i=0, j=0 -> a
    i=0, j=1 -> b
    i=1, j=0 -> c
    i=1, j=1 -> d
    
    U indices (n*m=4):
    a: 0*2+0 = 0
    b: 0*2+1 = 1
    c: 1*2+0 = 2
    d: 1*2+1 = 3
    
    V indices (m*p=20):
    e: 0*10 + 2*b = 2*b
    f: 0*10 + 2*b+1 = 2*b+1
    g: 1*10 + 2*b = 10+2*b
    h: 1*10 + 2*b+1 = 10+2*b+1
    
    W indices (p*n=20):
    C11 (i=0, l=2*b): W[2*b*2 + 0] = W[4*b]
    C12 (i=0, l=2*b+1): W[(2*b+1)*2 + 0] = W[4*b+2]
    C21 (i=1, l=2*b): W[2*b*2 + 1] = W[4*b+1]
    C22 (i=1, l=2*b+1): W[(2*b+1)*2 + 1] = W[4*b+3]
    
    Let's check the seed code's W assignments against this.
    Seed P1 (M1):
    W[4*b] += 1 (C11)
    W[4*b+3] += 1 (C22)
    Formula: C11 = M1 + ... ; C22 = M1 - ...
    Matches.
    
    Seed P2 (M2):
    W[4*b+1] += 1 (C21)
    W[4*b+3] += -1 (C22)
    Formula: C21 = M2 + ... ; C22 = ... - M2 ...
    Matches.
    
    Seed P3 (M3):
    W[4*b+2] += 1 (C12)
    W[4*b+3] += 1 (C22)
    Formula: C12 = ... + M3 ; C22 = ... + M3 ...
    Matches.
    
    Seed P4 (M4):
    W[4*b] += 1 (C11)
    W[4*b+1] += 1 (C21)
    Formula: C11 = ... + M4 ... ; C21 = ... + M4
    Matches.
    
    Seed P5 (M5):
    W[4*b] += -1 (C11)
    W[4*b+2] += 1 (C12)
    Formula: C11 = ... - M5 ... ; C12 = ... + M5
    Matches.
    
    Seed P6 (M6):
    W[4*b+3] += 1 (C22)
    Formula: C22 = ... + M6
    Matches.
    
    Seed P7 (M7):
    W[4*b] += 1 (C11)
    Formula: C11 = ... + M7
    Matches.
    
    The seed code is correct. I will reproduce this logic cleanly.
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5
    rank = 7 * num_blocks # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        base = 7 * b
        
        # M1 = (a + d)(e + h)
        col = base
        U[0][col] = 1  # a
        U[3][col] = 1  # d
        V[2*b][col] = 1    # e
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = 1    # C11
        W[4*b+3][col] = 1  # C22
        
        # M2 = (c + d)e
        col = base + 1
        U[2][col] = 1  # c
        U[3][col] = 1  # d
        V[2*b][col] = 1    # e
        W[4*b+1][col] = 1  # C21
        W[4*b+3][col] = -1 # C22
        
        # M3 = a(f - h)
        col = base + 2
        U[0][col] = 1  # a
        V[2*b+1][col] = 1    # f
        V[10 + 2*b + 1][col] = -1 # h
        W[4*b+2][col] = 1   # C12
        W[4*b+3][col] = 1   # C22
        
        # M4 = d(g - e)
        col = base + 3
        U[3][col] = 1  # d
        V[10 + 2*b][col] = 1   # g
        V[2*b][col] = -1       # e
        W[4*b][col] = 1        # C11
        W[4*b+1][col] = 1      # C21
        
        # M5 = (a + b)h
        col = base + 4
        U[0][col] = 1  # a
        U[1][col] = 1  # b
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = -1         # C11
        W[4*b+2][col] = 1        # C12
        
        # M6 = (c - a)(e + f)
        col = base + 5
        U[2][col] = 1   # c
        U[0][col] = -1  # a
        V[2*b][col] = 1   # e
        V[2*b+1][col] = 1 # f
        W[4*b+3][col] = 1 # C22
        
        # M7 = (b - d)(g + h)
        col = base + 6
        U[1][col] = 1   # b
        U[3][col] = -1  # d
        V[10 + 2*b][col] = 1   # g
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = 1        # C11

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
