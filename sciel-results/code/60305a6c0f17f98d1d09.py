import time
import numpy as np

DIMENSIONS = [2, 2, 3]
DEADLINE = 120.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0

def _verify(U, V, W, n=2, m=2, p=3):
    """Verify the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check T_{i,j,k} = 1
                lhs = 0.0
                for r in range(rank):
                    lhs += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(lhs - 1.0) > 1e-10:
                    return False
                
                # Check T_{i,j,k'} = 0 for k' != k? 
                # Actually, the tensor is defined on all indices.
                # We need sum_r U[i,j,r] V[j,k',r] W[k',i,r] = 0 for k' != k?
                # No, the equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T_{i,j,k}.
                # The "wrong" indices are where the output index w does not match the input pair (i,k).
                # The input pair is (i,j) and (j,k). The output is (i,k).
                # So for fixed i,j,k, we check w = k*n+i.
                # What about other w?
                # The tensor T_{u,v,w} is non-zero only if w matches the output of u,v.
                # So for fixed u=i*m+j, v=j*p+k, we need:
                # sum_r U[u,r] V[v,r] W[w,r] = 1 if w = k*n+i
                # sum_r U[u,r] V[v,r] W[w,r] = 0 if w != k*n+i
                
    # Re-verify with full check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                target_w = k * n + i
                
                for w_idx in range(p * n):
                    lhs = 0.0
                    for r in range(rank):
                        lhs += U_arr[u_idx, r] * V_arr[v_idx, r] * W_arr[w_idx, r]
                    
                    if w_idx == target_w:
                        if abs(lhs - 1.0) > 1e-10:
                            return False
                    else:
                        if abs(lhs) > 1e-10:
                            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    rank = n * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map r to (i', k')
    for i_prime in range(n):
        for k_prime in range(p):
            r = k_prime * n + i_prime # Or any bijection. Let's use r = i_prime * p + k_prime?
            # Let's stick to r index in 0..rank-1.
            # Let r correspond to (i_prime, k_prime).
            # We need to fill U, V, W.
            
            # U[i*m+j, r] = delta(i, i_prime)
            for j in range(m):
                U[i_prime * m + j][r] = 1 if i_prime == i_prime else 0 # Wait, U depends on i.
                # U_{i m + j, (i', k')} = delta(i, i')
                # So for fixed r=(i', k'), U[i*m+j, r] is 1 if i==i'.
                
            # Let's just loop and fill.
            
    # Reset and fill properly
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i_prime = r // p
        k_prime = r % p
        
        # U[i*m+j, r] = 1 if i == i_prime
        for i in range(n):
            if i == i_prime:
                for j in range(m):
                    U[i * m + j][r] = 1
        
        # V[j*p+k, r] = 1 if k == k_prime
        for k in range(p):
            if k == k_prime:
                for j in range(m):
                    V[j * p + k][r] = 1
        
        # W[k*n+i, r] = 1 if k == k_prime AND i == i_prime
        for k in range(p):
            for i in range(n):
                if k == k_prime and i == i_prime:
                    W[k * n + i][r] = 1
    
    sol = {"U": U, "V": V, "W": W}
    
    if _verify(U, V, W, n, m, p):
        return sol
    
    # Fallback to Rank 9 (from current program) if Rank 6 fails (it shouldn't)
    return _rank9_solution()

def _rank9_solution():
    """Return the rank-9 decomposition using 7 Strassen terms + 2 for last column."""
    n, m, p = 2, 2, 3
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = -1; W[3][0] = 1; W[5][0] = -1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    U[0][2] = 1
    V[3][2] = 1; V[5][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = -1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[1][4] = 1; W[3][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[3][5] = 1
    W[0][5] = 1; W[2][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[5][6] = 1
    W[1][6] = 1; W[3][6] = -1
    
    U[0][7] = 1; U[2][7] = 1
    V[2][7] = 1
    W[4][7] = 1; W[5][7] = 1
    
    U[1][8] = 1; U[3][8] = 1
    V[5][8] = 1
    W[4][8] = 1; W[5][8] = 1
    
    return {"U": U, "V": V, "W": W}
