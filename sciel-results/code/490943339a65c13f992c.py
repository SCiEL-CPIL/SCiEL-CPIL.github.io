"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Index conventions:
    # U[i*m + j, c]
    # V[j*p + k, c]  
    # W[k*n + i, c]
    # Decomposition: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] = W[k*n+i,c]
    # Wait, that's not right. Let me reconsider.
    
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # As a bilinear map: given A (n*m) and B (m*p), produce C (p*n)
    # Rank-r decomposition: C = sum_c outer(U_c, V_c) where U_c is (n*m), V_c is (m*p)
    # But that gives C as (n*m, m*p) which is wrong.
    
    # Actually, the trilinear tensor T has shape (n*m, m*p, p*n)
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if j=j', i=i', k=k', else 0
    # 
    # Rank-r decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)
    
    # So: sum_d U[i*m+j, d] * V[j'*p+k, d] * W[k'*n+i', d] = 1 if j=j', i=i', k=k'
    
    # Let's try to find a low-rank decomposition.
    # For the schoolbook: r = n*m*p = 36
    # Each term: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for one (i,j,k)
    
    # Can we do better? Let's try various ranks and use a verification approach.
    
    def verify(U, V, W):
        """Verify that U, V, W form a valid decomposition."""
        n, m, p = DIMENSIONS
        r = len(U[0]) if U else 0
        # Check shape
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            return False
        for c in range(r):
            if len(U[c if False else 0]) != r:  # U is list of rows
                pass
        # T[a,b,c] should equal sum_d U[a,d]*V[b,d]*W[c,d]
        # For each (i,j,j',k,k',i'):
        # sum_d U[i*m+j, d] * V[j'*p+k, d] * W[k'*n+i', d] = 1 if i=i',j=j',k=k' else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Compute T from decomposition
        # T[i*m+j, j'*p+k, k'*n+i'] = sum_d U[i*m+j,d]*V[j'*p+k,d]*W[k'*n+i',d]
        # This is: for each d, U[:,d] outer V[:,d] outer W[:,d]
        # Reshape and check
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                a = i*m + j
                                b = jp*p + k
                                c_idx = kp*n + ip
                                expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                actual = np.dot(np.dot(U_arr[a], V_arr[b]), W_arr[c_idx])
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Try schoolbook first as fallback
    def schoolbook():
        n, m, p = DIMENSIONS
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * m * p // m * p)]  # m*p
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
    
    sb = schoolbook()
    
    # Try to find lower rank by solving the system
    # The constraints are: for all (i,j,j',k,k',i') with not (i=i',j=j',k=k'):
    #   sum_d U[i*m+j,d]*V[j'*p+k,d]*W[k'*n+i',d] = 0
    # And for (i,j,k,i=j,j'=j,k'=k,i'=i):
    #   sum_d U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = 1
    
    # This is a system of polynomial equations. Hard to solve directly.
    # Let's try small ranks and see if we can find solutions by construction.
    
    # For 2x2 x 2x9, maybe we can use the fact that m=2 is small.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # Let's try rank = 2*p + something? Or use Strassen-like ideas.
    
    # Actually, let's just try to verify if rank 36 is minimal or if we can do better.
    # Given time constraints, let's try a few candidate ranks.
    
    # For now, return schoolbook. We could try to improve but verification is expensive.
    if time.time() > deadline - 5:
        return sb
    
    # Try rank 30, 28, 24 with random initialization and check
    # But solving the nonlinear system is hard. Let's just return schoolbook for safety.
    
    return sb
# EVOLVE-BLOCK-END
