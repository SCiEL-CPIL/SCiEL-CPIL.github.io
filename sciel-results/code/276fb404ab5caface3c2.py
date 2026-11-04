import time
import numpy as np

DIMENSIONS = [2, 3, 3]
DEADLINE = 150.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    dim_u = n * m  # 6
    dim_v = m * p  # 9
    dim_w = p * n  # 6

    # Proposed rank-9 decomposition
    rank = 9
    
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    # Map (j,k) to column index r
    # r = j * p + k
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U entries: i=0 and i=1 for this j
            # U index: i*m + j
            U[0 * m + j][r] = 1
            U[1 * m + j][r] = 1
            
            # V entry: j, k
            # V index: j*p + k
            V[j * p + k][r] = 1
            
            # W entries: k, i=0 and i=1
            # W index: k*n + i
            W[k * n + 0][r] = 1
            W[k * n + 1][r] = 1

    # Rigorous verification
    def verify_full(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check all possible combinations of indices
        # T_{(i,j), (j',k), (k',i')} should be 1 if j==j', k==k', i==i' ?
        # No, T_{(i,j), (j,k), (k,i)} = 1.
        # So for any i, j, k, the sum should be 1.
        # For any i, j, j', k, i' where j != j', the sum should be 0.
        
        # 1. Check diagonal (j==j', k==k', i==i')
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_np[i*m+j, :] * V_np[j*p+k, :] * W_np[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        
        # 2. Check off-diagonal (j != j' or k != k' or i != i')
        # The tensor entry T_{(i,j), (j',k), (k',i')} is 0 unless j==j', k==k', i==i'.
        # Actually, the tensor is T_{(i,j), (j,k), (k,i)}.
        # So T_{(i,j), (j',k'), (k'',i'')} is 1 only if j==j', k==k', i==i'?
        # No, the indices are coupled.
        # The support of the tensor is {( (i,j), (j,k), (k,i) ) | i in 0..n-1, j in 0..m-1, k in 0..p-1 }.
        # So for any (i,j), (j',k'), (k'',i''), if (j, k, i) != (j', k'', i') AND (j,k,i) != (j',k'',i')?
        # No, the entry is 1 if the triple matches the multiplication pattern.
        # The multiplication pattern is defined by the shared indices.
        # So T_{(i,j), (j',k), (k',i')} = 1 if j==j' and k==k' and i==i'.
        # Otherwise 0.
        
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                # If (j,k,i) != (j_prime, k_prime, i_prime), sum must be 0
                                if not (j == j_prime and k == k_prime and i == i_prime):
                                    val = np.sum(U_np[i*m+j, :] * V_np[j_prime*p+k, :] * W_np[k_prime*n+i_prime, :])
                                    if not np.isclose(val, 0.0, atol=1e-9):
                                        return False
        return True

    if verify_full(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to standard 18 if verification fails
    rank_std = n * m * p
    U_std = [[0] * rank_std for _ in range(dim_u)]
    V_std = [[0] * rank_std for _ in range(dim_v)]
    W_std = [[0] * rank_std for _ in range(dim_w)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_std[i * m + j][col] = 1
                V_std[j * p + k][col] = 1
                W_std[k * n + i][col] = 1
                col += 1
                
    return {"U": U_std, "V": V_std, "W": W_std}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
