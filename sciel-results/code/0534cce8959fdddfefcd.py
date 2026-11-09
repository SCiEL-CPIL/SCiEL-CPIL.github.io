import numpy as np
import random
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin from 120s limit

    # Schoolbook decomposition
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Check if schoolbook is valid (it is)
    # Try to find a lower rank decomposition
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try ranks from 20 to 59
    for r in range(20, rank_schoolbook):
        if time.time() > deadline:
            break
        found = False
        for _ in range(100):  # 100 random attempts per rank
            if time.time() > deadline:
                break
            U = np.random.randint(-1, 2, (n * m, r))
            V = np.random.randint(-1, 2, (m * p, r))
            # Solve for W
            # The condition is: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta[i,k]
            # This is a linear system for W
            # Let X = U @ V.T (shape n*m x m*p? No)
            # U is n*m x r. V is m*p x r.
            # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta[i,k]
            # Let A = U @ V.T (shape n*m x m*p)
            # A[i*m+j, j*p+k] = sum_r U[i*m+j, r] V[j*p+k, r]
            # Then sum_k A[i*m+j, j*p+k] W[k*n+i, r]? No, W is p*n x r.
            # The condition is sum_r (sum_j U[i,j,r] A_{ij}) (sum_k V[j,k,r] B_{jk})? No.
            # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta[i,k]
            # This is not a simple linear system for W because W depends on r.
            # Actually, for each r, we have a term.
            # Let's use a different approach: random W as well.
            W = np.random.randint(-1, 2, (p * n, r))
            
            # Check validity
            valid = True
            for i in range(n):
                for k in range(p):
                    val = 0
                    for j in range(m):
                        for r_idx in range(r):
                            val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if val != (1 if i == k else 0):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        if found:
            break

    # Convert to lists of integers
    U_list = [[int(x) for x in row] for row in best_U]
    V_list = [[int(x) for x in row] for row in best_V]
    W_list = [[int(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
