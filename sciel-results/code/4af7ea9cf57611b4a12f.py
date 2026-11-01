"""Strassen-based decomposition for 2x2 * 2x14 matrix multiplication."""
import time
import random

DIMENSIONS = [2, 2, 14]

def _strassen_rank7():
    """
    Return the 7 terms of Strassen's algorithm for 2x2 matrix multiplication.
    Each term is (u, v, w) where:
      u[i,j] : coefficient for A[i,j] (2x2)
      v[j',k] : coefficient for B[j',k] (2x2)
      w[i',k'] : coefficient for C[i',k'] (2x2)
    
    Strassen's 7 products for A*B:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Indexing: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1}
    """
    terms = []
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # u: A00 + A11, v: B00+B11, w: C00 + C11
    u = [[1, 0], [1, 0]]  # A00 + A11
    v = [[1, 1], [1, 1]]  # B00+B11 for row 0, B00+B11 for row 1... 
    # Wait, B is 2x2. B[j,k]. 
    # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])? No.
    # Strassen: M1 = (A11+A22)(B11+B22) using 1-indexed.
    # In 0-indexed: M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # But this is only for the block structure. Let me be more careful.
    
    # For 2x2 matrices:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C = [[M1+M4-M5+M7, M3+M5],
    #      [M2+M4,         M1-M2+M3+M6]]
    
    # Let me define each term properly.
    # u vector for A: u[i,j] such that sum_j u[i,j]*A[i,j] gives the A-part of the product
    # Actually u is a 2x2 matrix where u[i,j] is the coefficient.
    # The product term is sum_{i,j} u[i,j] * A[i,j] * (sum_{k,l} v[k,l] * B[k,l]) ... no.
    
    # Rank-1 term: sum_{i,j} u_r[i,j] * A[i,j] * sum_{k} ... 
    # Actually for matrix multiplication tensor:
    # T[(i,j),(k,l),(m,n)] = delta_{j,k} * delta_{l,n} * delta_{i,m}
    # A rank-1 decomposition: T = sum_r u_r[i,j] * v_r[k,l] * w_r[m,n]
    
    # For M1 = (a+d)(e+h):
    # A-part: a + d = A[0,0] + A[1,1] => u1[0,0]=1, u1[1,1]=1, others 0
    # B-part: e + h = B[0,0] + B[1,1] => v1[0,0]=1, v1[1,1]=1, others 0
    # C-part: M1 contributes +1 to C[0,0] and +1 to C[1,1]
    # w1[0,0]=1, w1[1,1]=1, others 0
    
    terms.append({
        'u': [[1, 0], [0, 1]],
        'v': [[1, 0], [0, 1]],
        'w': [[1, 0], [0, 1]]
    })
    
    # M2 = (c+d)e
    # A-part: c + d = A[1,0] + A[1,1] => u2[1,0]=1, u2[1,1]=1
    # B-part: e = B[0,0] => v2[0,0]=1
    # C-part: M2 contributes +1 to C[1,0] and -1 to C[1,1]
    terms.append({
        'u': [[0, 0], [1, 1]],
        'v': [[1, 0], [0, 0]],
        'w': [[0, 0], [1, -1]]
    })
    
    # M3 = a(g-h)
    # A-part: a = A[0,0] => u3[0,0]=1
    # B-part: g - h = B[1,0] - B[1,1] => v3[1,0]=1, v3[1,1]=-1
    # C-part: M3 contributes +1 to C[0,1] and +1 to C[1,1]
    terms.append({
        'u': [[1, 0], [0, 0]],
        'v': [[0, 0], [1, -1]],
        'w': [[0, 1], [0, 1]]
    })
    
    # M4 = d(f-e)
    # A-part: d = A[1,1] => u4[1,1]=1
    # B-part: f - e = B[0,1] - B[0,0] => v4[0,1]=1, v4[0,0]=-1
    # C-part: M4 contributes +1 to C[0,0] and +1 to C[1,0]
    terms.append({
        'u': [[0, 0], [0, 1]],
        'v': [[-1, 1], [0, 0]],
        'w': [[1, 0], [1, 0]]
    })
    
    # M5 = (a+b)h
    # A-part: a + b = A[0,0] + A[0,1] => u5[0,0]=1, u5[0,1]=1
    # B-part: h = B[1,1] => v5[1,1]=1
    # C-part: M5 contributes -1 to C[0,0] and +1 to C[0,1]
    terms.append({
        'u': [[1, 1], [0, 0]],
        'v': [[0, 0], [0, 1]],
        'w': [[-1, 1], [0, 0]]
    })
    
    # M6 = (c-a)(e+f)
    # A-part: c - a = A[1,0] - A[0,0] => u6[1,0]=1, u6[0,0]=-1
    # B-part: e + f = B[0,0] + B[0,1] => v6[0,0]=1, v6[0,1]=1
    # C-part: M6 contributes +1 to C[1,1]
    terms.append({
        'u': [[-1, 0], [1, 0]],
        'v': [[1, 1], [0, 0]],
        'w': [[0, 0], [0, 1]]
    })
    
    # M7 = (b-d)(g+h)
    # A-part: b - d = A[0,1] - A[1,1] => u7[0,1]=1, u7[1,1]=-1
    # B-part: g + h = B[1,0] + B[1,1] => v7[1,0]=1, v7[1,1]=1
    # C-part: M7 contributes +1 to C[0,0]
    terms.append({
        'u': [[0, 1], [0, -1]],
        'v': [[0, 0], [1, 1]],
        'w': [[1, 0], [0, 0]]
    })
    
    return terms

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # We decompose the 2x2 * 2x14 multiplication into 7 blocks of 2x2 * 2x2
    # Each 2x2 * 2x2 block uses Strassen's 7-term algorithm
    # Total rank = 7 blocks * 7 terms = 49
    
    strassen_terms = _strassen_rank7()
    
    num_blocks = p // 2  # 14 / 2 = 7
    
    rank = num_blocks * 7
    
    # U: shape (n*m, rank) = (4, 49)
    # V: shape (m*p, rank) = (28, 49)
    # W: shape (p*n, rank) = (28, 49)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(num_blocks):
        for t in range(7):
            col = block * 7 + t
            term = strassen_terms[t]
            u2 = term['u']  # 2x2
            v2 = term['v']  # 2x2
            w2 = term['w']  # 2x2
            
            # Fill U: A is 2x2, indexed as A[i,j] -> U[i*m+j, col] = u2[i][j]
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = u2[i][j]
            
            # Fill V: B is 2x14. Block `block` covers columns [2*block, 2*block+1]
            # B[j, 2*block + k] -> V[j*p + (2*block + k), col] = v2[j][k]
            for j in range(m):
                for k in range(2):
                    V[j * p + 2 * block + k][col] = v2[j][k]
            
            # Fill W: Output C is 2x14. C[i, 2*block + k] -> W[(2*block + k)*n + i, col]
            # W index: k_out * n + i, where k_out is the output column
            for i in range(n):
                for k in range(2):
                    W[(2 * block + k) * n + i][col] = w2[i][k]
    
    return {"U": U, "V": V, "W": W}
