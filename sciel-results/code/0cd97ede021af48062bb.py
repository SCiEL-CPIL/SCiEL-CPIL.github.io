import numpy as np
import time
import itertools

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)
    
    # Block-diagonal decomposition with rank 6
    # For each row i of A (i=0,1), decompose the 1x3 by 3x9 multiplication with rank 3
    rank = 6
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    for i in range(n):
        for j in range(m):
            for r in range(m):
                # U[i*m+j, i*m+r] = delta(j, r)
                if j == r:
                    U[i*m+j, i*m+r] = 1.0
                # V[j*p+k, i*m+r] = delta(j, r) for all k
                if j == r:
                    for k in range(p):
                        V[j*p+k, i*m+r] = 1.0
                # W[k*n+i, i*m+r] = 1 for all k
                for k in range(p):
                    W[k*n+i, i*m+r] = 1.0
    
    # Verify the solution
    if verify_solution(U, V, W, T):
        U_list = to_list(U)
        V_list = to_list(V)
        W_list = to_list(W)
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook if verification fails
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    U_list = to_list(U_sb)
    V_list = to_list(V_sb)
    W_list = to_list(W_sb)
    
    return {"U": U_list, "V": V_list, "W": W_list}
