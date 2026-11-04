def verify(U, V, W, n, m, p):
    rank = len(U[0])
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    for i in range(len(U)):
        if len(U[i]) != rank: return False
    for i in range(len(V)):
        if len(V[i]) != rank: return False
    for i in range(len(W)):
        if len(W[i]) != rank: return False
        
    # Check tensor decomposition
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is O(n*m*p*rank). For 3,4,8, rank 88, it's 3*4*8*88 = 8448 ops. Fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                # The tensor entry should be 1 if this is the schoolbook term, but for hybrid it's different.
                # Actually, the tensor T is fixed: T[i,j,k] = 1 for all i,j,k? No.
                # The matrix multiplication tensor has T[i,j,k] = 1 if the index matches the product.
                # Wait, the tensor T is defined by T[i, j, k] = 1 for all i, j, k? No.
                # The tensor for matrix multiplication A (n x m) * B (m x p) = C (n x p) is:
                # C[i, k] = sum_j A[i, j] B[j, k].
                # The tensor T has indices (i, j, k) and T[i, j, k] = 1.
                # So we must have sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1 for all i, j, k.
                if s != 1:
                    return False
    return True
