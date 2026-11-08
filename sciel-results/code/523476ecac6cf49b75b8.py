import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = 3, 3, 4
    
    def verify(U, V, W):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        r = U.shape[1]
        # S[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
        S = np.einsum('ijt,jkt,kit->ijk', 
                      U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
        target = np.ones((n, m, p))
        diff = S - target
        return np.max(np.abs(diff)) < 1e-9, S

    def schoolbook():
        U = np.zeros((n*m, n*m*p), dtype=int)
        V = np.zeros((m*p, n*m*p), dtype=int)
        W = np.zeros((p*n, n*m*p), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def try_rank_10():
        # Structure:
        # t=0..6: for k=0,1,2 (3x3x3 part)
        # t=7..9: for k=3 (3x3x1 part)
        
        U = np.zeros((n*m, 10), dtype=int)
        V = np.zeros((m*p, 10), dtype=int)
        W = np.zeros((p*n, 10), dtype=int)
        
        # Initialize k=3 part (deterministic)
        for j in range(m):
            t = 7 + j
            # U[i*m+j, t] = 1
            for i in range(n):
                U[i*m+j, t] = 1
            # V[j*p+3, t] = 1
            V[j*p+3, t] = 1
            # W[3*n+i, t] = 1
            for i in range(n):
                W[3*n+i, t] = 1
        
        # Initialize k=0,1,2 part (random)
        for t in range(7):
            # Sparse random initialization
            for _ in range(3):
                r_u = random.randint(0, n*m-1)
                U[r_u, t] = random.choice([-1, 1])
                r_v = random.randint(0, m*p-1)
                # Only use k=0,1,2 for V in this part?
                # V index is j*p+k. We want to influence k=0,1,2.
                # So k should be 0,1,2.
                j = random.randint(0, m-1)
                k = random.randint(0, 2)
                V[j*p+k, t] = random.choice([-1, 1])
                r_w = random.randint(0, p*n-1)
                # W index is k*n+i. We want to influence k=0,1,2.
                k2 = random.randint(0, 2)
                i = random.randint(0, n-1)
                W[k2*n+i, t] = random.choice([-1, 1])
        
        # Local search
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        def compute_err():
            S = np.einsum('ijt,jkt,kit->ijk', 
                          U.reshape(n,m,10), V.reshape(m,p,10), W.reshape(p,n,10))
            return np.sum((S - 1.0)**2)
        
        current_err = compute_err()
        
        for it in range(50000):
            if time.time() > deadline - 5:
                break
            if current_err < 1e-9:
                break
                
            # Pick a random column
            t = random.randint(0, 9)
            # Pick a random matrix and row
            mat = random.randint(0, 2)
            if mat == 0:
                row = random.randint(0, n*m-1)
                old_val = U[row, t]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val: continue
                U[row, t] = new_val
                new_err = compute_err()
                if new_err < current_err:
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    U[row, t] = old_val
            elif mat == 1:
                row = random.randint(0, m*p-1)
                old_val = V[row, t]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val: continue
                V[row, t] = new_val
                new_err = compute_err()
                if new_err < current_err:
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    V[row, t] = old_val
            else:
                row = random.randint(0, p*n-1)
                old_val = W[row, t]
                new_val = random.choice([-2, -1, 0, 1, 2])
                if new_val == old_val: continue
                W[row, t] = new_val
                new_err = compute_err()
                if new_err < current_err:
                    current_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                else:
                    W[row, t] = old_val
        
        valid, _ = verify(best_U, best_V, best_W)
        if valid:
            return best_U, best_V, best_W
        return None

    # Try rank 10
    sol = try_rank_10()
    if sol is not None:
        U, V, W = sol
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
