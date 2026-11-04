import numpy as np
import time
import itertools

def construct():
    n, m, p = 3, 4, 8
    
    # The schoolbook rank is n*m*p = 96
    # We'll try to find a lower rank using tensor rank search
    # For M_{3,4,8}, we'll attempt a limited search for rank < 96
    
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    # Start with schoolbook
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U[I, col] = 1
                V[J, col] = 1
                W[K, col] = 1
                col += 1
    
    # Try to find a lower rank using random initialization + optimization
    # We'll use a simple approach: try various ranks below 96 and see if we can find
    # a valid decomposition
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # For small dimensions, we can try a more systematic approach
    # The tensor M_{n,m,p} has a known minimum rank. For M_{3,4,8}, let's try
    # ranks from 96 down to see if we can find a valid decomposition
    
    # Actually, for matrix multiplication, the minimum rank is at least max(n*m, m*p, p*n)
    # and at most n*m*p. For M_{3,4,8}: max(12, 32, 24) = 32, so rank >= 32
    
    # Let's try a random search for lower ranks
    # We'll use a simple iterative approach
    
    for trial_rank in range(95, 90, -1):  # Try ranks 95 down to 91
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):  # 50 random attempts per rank
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U_trial = np.random.randn(n * m, trial_rank) * 0.1
            V_trial = np.random.randn(m * p, trial_rank) * 0.1
            W_trial = np.random.randn(p * n, trial_rank) * 0.1
            
            # Check if this decomposition is valid
            # The decomposition is valid if for all i,j,k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # and for all invalid (I,J,K) combinations, the sum is 0
            
            valid = True
            max_err = 0
            
            # Check valid terms
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        I = i * m + j
                        J = j * p + k
                        K = k * n + i
                        val = np.dot(U_trial[I], V_trial[J] * W_trial[K])
                        err = abs(val - 1.0)
                        if err > 1e-6:
                            valid = False
                            break
                        if err > max_err:
                            max_err = err
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                found = True
                best_rank = trial_rank
                best_U, best_V, best_W = U_trial.copy(), V_trial.copy(), W_trial.copy()
                break
        
        if found:
            break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < rank:
        U, V, W = best_U, best_V, best_W
    else:
        U, V, W = np.zeros((n * m, rank)), np.zeros((m * p, rank)), np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    U[I, col] = 1
                    V[J, col] = 1
                    W[K, col] = 1
                    col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

if __name__ == "__main__":
    res = construct()
    n, m, p = 3, 4, 8
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                dot = np.dot(U[I], V[J] * W[K])
                if abs(dot - 1.0) > max_err:
                    max_err = abs(dot - 1.0)
    print(f"Max Error: {max_err}")
