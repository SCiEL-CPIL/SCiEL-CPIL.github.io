DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    rank = 44
    
    U = [[0.0] * rank for _ in range(n * m)]  # 9 x 44
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 44
    W = [[0.0] * rank for _ in range(p * n)]  # 18 x 44
    
    # A indices: i*m + j = i*3 + j
    # B indices: j*p + k = j*6 + k
    # W indices: k*n + i = k*3 + i
    
    def a_idx(i, j):
        return i * m + j
    
    def b_idx(j, k):
        return j * p + k
    
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    # Block 1: B columns 0,1,2 -> C columns 0,1,2
    # Strassen terms
    # S1 = (A11+A22)*(B11+B22)
    U[a_idx(0,0)][col] += 1; U[a_idx(1,1)][col] += 1
    V[b_idx(0,0)][col] += 1; V[b_idx(1,1)][col] += 1
    W[w_idx(0,0)][col] += 1  # C11 += S1
    W[w_idx(0,1)][col] -= 1  # C22 += S1
    col += 1
    
    # S2 = (A21+A22)*B11
    U[a_idx(1,0)][col] += 1; U[a_idx(1,1)][col] += 1
    V[b_idx(0,0)][col] += 1
    W[w_idx(1,1)][col] += 1  # C21 += S2
    W[w_idx(1,2)][col] -= 1  # C22 -= S2
    col += 1
    
    # S3 = A11*(B12-B22)
    U[a_idx(0,0)][col] += 1
    V[b_idx(1,1)][col] += 1; V[b_idx(2,2)][col] -= 1
    W[w_idx(0,2)][col] += 1  # C12 += S3
    W[w_idx(1,3)][col] += 1  # C22 += S3
    col += 1
    
    # S4 = A22*(B21-B11)
    U[a_idx(1,1)][col] += 1
    V[b_idx(1,1)][col] -= 1; V[b_idx(2,0)][col] += 1
    W[w_idx(1,0)][col] += 1  # C21 += S4
    W[w_idx(1,2)][col] += 1  # C22 += S4
    col += 1
    
    # S5 = (A11+A12)*B22
    U[a_idx(0,0)][col] += 1; U[a_idx(0,1)][col] += 1
    V[b_idx(2,2)][col] += 1
    W[w_idx(0,1)][col] -= 1  # C11 -= S5
    W[w_idx(0,2)][col] += 1  # C12 += S5
    col += 1
    
    # S6 = (A21-A11)*(B11+B12)
    U[a_idx(1,0)][col] += 1; U[a_idx(0,0)][col] -= 1
    V[b_idx(0,0)][col] += 1; V[b_idx(1,1)][col] += 1
    W[w_idx(1,3)][col] += 1  # C22 += S6
    col += 1
    
    # S7 = (A12-A22)*(B21+B22)
    U[a_idx(0,1)][col] += 1; U[a_idx(1,1)][col] -= 1
    V[b_idx(2,0)][col] += 1; V[b_idx(2,1)][col] += 1
    W[w_idx(0,0)][col] += 1  # C11 += S7
    W[w_idx(0,3)][col] += 1  # C22... wait
    col += 1
    
    # Hmm, I'm getting confused with the Strassen formulas. Let me redo this more carefully.
    
    # Actually, let me restart with a cleaner implementation.
    
    return {"U": U, "V": W}  # Placeholder
