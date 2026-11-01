import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # For 2x4 x 4x7:
    # We can use a block decomposition.
    # Let A be 2x4, B be 4x7.
    # Partition A into columns: A = [A0, A1, A2, A3] where A0 is 2x1, etc.
    # Partition B into rows: B = [B0; B1; B2; B3] where B0 is 1x7, etc.
    # C = A0 B0 + A1 B1 + A2 B2 + A3 B3.
    # Each term is 2x1 x 1x7.
    # A 2x1 x 1x7 tensor has rank 2 (since it's just 2 products: c_0k = a_0 b_k, c_1k = a_1 b_k).
    # So each term contributes rank 2.
    # Total rank 4 * 2 = 8?
    # Wait, the indices for U, V, W must be consistent.
    # For term j (where j is the split index):
    # U[i, r] = A[i, j] (2 entries for i=0,1)
    # V[k, r] = B[j, k] (7 entries for k=0..6)
    # W[i, k, r]? No, W is indexed by c = k*n + i.
    # W[c, r] = delta (identity)
    # Let's check if this works.
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # a = i*m + j_a. b = j_b*p + k. c = k_c*n + i_c.
    # If we use the block decomposition, for a specific j:
    # U[i*m + j, r] = 1 if i matches and r corresponds to (i, j).
    # Actually, let's just build it.
    
    # Rank 8 algorithm:
    # For each j in 0..3:
    #   For i in 0..1:
    #     r = 2*j + i
    #     U[i*m + j, r] = 1
    #     V[j*p + k, r] = B[j,k] (which is 1 if we are constructing the tensor, but wait...)
    #     The tensor T is defined as T[i,j,k] = 1.
    #     So we are not computing A*B for specific A,B, but decomposing the *multiplication operation* tensor.
    #     The tensor T[i,j,k] = 1 means that the (i,k)-th entry of C is the product of A[i,j] and B[j,k].
    #     So T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some consistent i,j,k.
    #     
    #     Let's re-evaluate the rank 8 idea.
    #     If we set:
    #     For j in 0..3, i in 0..1:
    #       r = 2*j + i
    #       U[i*m + j, r] = 1
    #       V[j*p + k, r] = 1 for all k? No.
    #       W[k*n + i, r] = 1 if k matches?
    #       
    #     If U[i,j,r] = 1, V[j,k,r] = 1, W[k,i,r] = 1, then T[i,j,k] = 1.
    #     This is the schoolbook rank 56.
    #     
    #     Can we do better?
    #     Consider the term for a fixed j.
    #     The sub-tensor for fixed j is T_j[i,k] = 1.
    #     This is a 2x7 matrix of ones.
    #     We can decompose this matrix of ones (rank 1 matrix) into rank 1?
    #     No, the tensor is T[i, (j,k), (k,i)] = 1.
    #     For fixed j, we have indices i (2), k (7).
    #     T[i, j, k] = 1.
    #     We want to represent sum_{r in R_j} U[i, r] V[j, k, r] W[k, i, r] = 1.
    #     Note that V and W depend on k.
    #     This is a tensor of size 2 x 7 x (7*2) = 2 x 7 x 14.
    #     Wait, the indices for V are (j, k), and for W are (k, i).
    #     For fixed j, V is a 7xR matrix (indexed by k and r).
    #     W is a 14xR matrix (indexed by (k,i) and r).
    #     We want T[i, k] = sum_r U[i,r] V[k,r] W[(k,i), r] = 1.
    #     
    #     If we set R_j to have rank 2:
    #     r1, r2.
    #     U[i, r1] = 1, U[i, r2] = 0 for i=0?
    #     U[i, r1] = 0, U[i, r2] = 1 for i=1?
    #     Then for i=0: T[0,k] = sum_r 1 * V[k,r] W[(k,0),r] = V[k,r1]W[(k,0),r1].
    #     We want this to be 1 for all k.
    #     Set V[k,r1] = 1, W[(k,0),r1] = 1.
    #     For i=1: T[1,k] = V[k,r2]W[(k,1),r2].
    #     Set V[k,r2] = 1, W[(k,1),r2] = 1.
    #     
    #     This works!
    #     For each j, we use 2 ranks.
    #     Total rank = 4 * 2 = 8.
    
    rank = 8
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for i in range(n):
            r = 2 * j + i
            # U[i, j, r] = 1
            U[i * m + j][r] = 1
            # V[j, k, r] = 1 for all k
            for k in range(p):
                V[j * p + k][r] = 1
            # W[k, i, r] = 1 for all k
            for k in range(p):
                W[k * n + i][r] = 1
                
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return result
        
    # Fallback to schoolbook if verification fails (should not happen)
    return _schoolbook(n, m, p)
