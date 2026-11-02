import time
import numpy as np

def construct():
    n, m, p = 3, 5, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition: rank = n*m*p = 105
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    def verify_exact(U, V, W, n, m, p):
        """
        Exact verification using numpy einsum for speed.
        T[a,b,c] = sum_r U[b,r] * V[c,r] * W[a,r]
        """
        rank_val = U.shape[1]
        T = np.einsum('br,cr,ar->abc', U, V, W)
        
        # Check diagonal positions (should be 1)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = k * n + i
                    b = i * m + j
                    c = j * p + k
                    if abs(T[a, b, c] - 1.0) > 1e-9:
                        return False
        
        # Check all other positions (should be 0)
        for a in range(p * n):
            k_a = a // n
            i_a = a % n
            for b in range(n * m):
                ii_b = b // m
                j_b = b % m
                for c in range(m * p):
                    jj_c = c // p
                    kk_c = c % p
                    if not (i_a == ii_b and j_b == jj_c and k_a == kk_c):
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True
    
    # The schoolbook solution is guaranteed to work
    best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    best_rank = rank
    
    # Try to find a better solution using random search with numpy-accelerated verification
    lower_bound = max(n * m, m * p, p * n)
    
    for trial_rank in range(104, lower_bound, -1):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(trial_rank * 10000 + attempt * 13 + 7)
            U_r = np.random.randint(-2, 3, size=(n * m, trial_rank)).astype(float) * 0.5
            V_r = np.random.randint(-2, 3, size=(m * p, trial_rank)).astype(float) * 0.5
            W_r = np.random.randint(-2, 3, size=(p * n, trial_rank)).astype(float) * 0.5
            
            if verify_exact(U_r, V_r, W_r, n, m, p):
                best_rank = trial_rank
                best_result = {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist()}
                break
    
    return best_result
