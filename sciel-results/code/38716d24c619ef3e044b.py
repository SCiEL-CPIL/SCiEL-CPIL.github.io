import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook for M_{n,m,p}
    def schoolbook(n, m, p):
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
    
    def verify(U, V, W, n, m, p):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True
    
    def try_find_m332_gradient(rank_target, max_iter=5000, lr=0.01, seed=42):
        """
        Find rank-rank_target decomposition for M_{3,3,2} using gradient descent
        with periodic integer rounding.
        """
        n2, m2, p2 = 3, 3, 2
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(n2 * m2, rank_target) * 0.1
        V = rng.randn(m2 * p2, rank_target) * 0.1
        W = rng.randn(p2 * n2, rank_target) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Compute loss: sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        # Compute current sum
                        s = 0.0
                        for r in range(rank_target):
                            s += U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r]
                        err = s - 1.0
                        loss += err * err
                        # Gradients
                        for r in range(rank_target):
                            dU[i*m2+j, r] += 2.0 * err * V[j*p2+k, r] * W[k*n2+i, r]
                            dV[j*p2+k, r] += 2.0 * err * U[i*m2+j, r] * W[k*n2+i, r]
                            dW[k*n2+i, r] += 2.0 * err * U[i*m2+j, r] * V[j*p2+k, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodically try to round to integers
            if it % 100 == 0 and loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
        
        # Try rounding the best solution
        U_round = np.round(best_U).astype(int)
        V_round = np.round(best_V).astype(int)
        W_round = np.round(best_W).astype(int)
        
        # Check if rounded solution is exact
        valid = True
        for i in range(n2):
            for j in range(m2):
                for k in range(p2):
                    s = 0
                    for r in range(rank_target):
                        s += U_round[i*m2+j, r] * V_round[j*p2+k, r] * W_round[k*n2+i, r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U_round.tolist(), V_round.tolist(), W_round.tolist()
        
        return None
    
    def try_find_m332_bruteforce(rank_target, max_tries=2000, seed=42):
        """
        Try random integer assignments for M_{3,3,2}.
        """
        n2, m2, p2 = 3, 3, 2
        rng = np.random.RandomState(seed)
        
        for _ in range(max_tries):
            U = rng.randint(-2, 2, size=(n2*m2, rank_target)).tolist()
            V = rng.randint(-2, 2, size=(m2*p2, rank_target)).tolist()
            W = rng.randint(-2, 2, size=(p2*n2, rank_target)).tolist()
            
            # Check
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        s = 0
                        for r in range(rank_target):
                            s += U[i*m2+j][r] * V[j*p2+k][r] * W[k*n2+i][r]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
        
        return None
    
    def combine_blocks(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
        """
        Combine two block decompositions:
        - Block 1: M_{n,m,p1} with rank r1
        - Block 2: M_{n,m,p2} with rank r2
        Result: M_{n,m,p1+p2} with rank r1+r2
        
        U1: (n*m) x r1
        V1: (m*p1) x r1
        W1: (p1*n) x r1
        
        U2: (n*m) x r2
        V2: (m*p2) x r2
        W2: (p2*n) x r2
        
        Output:
        U: (n*m) x (r1+r2)
        V: (m*(p1+p2)) x (r1+r2)
        W: ((p1+p2)*n) x (r1+r2)
        """
        r1 = len(U1[0])
        r2 = len(U2[0])
        rank_total = r1 + r2
        p_total = p1 + p2
        
        U_full = [[0] * rank_total for _ in range(n * m)]
        V_full = [[0] * rank_total for _ in range(m * p_total)]
        W_full = [[0] * rank_total for _ in range(p_total * n)]
        
        # Copy block 1
        for i in range(n * m):
            for r in range(r1):
                U_full[i][r] = U1[i][r]
        
        for j in range(m):
            for k in range(p1):
                for r in range(r1):
                    V_full[j * p_total + k][r] = V1[j * p1 + k][r]
        
        for k in range(p1):
            for i in range(n):
                for r in range(r1):
                    W_full[k * n + i][r] = W1[k * n + i][r]
        
        # Copy block 2 (offset in rank dimension)
        for i in range(n * m):
            for r in range(r2):
                U_full[i][r1 + r] = U2[i][r]
        
        for j in range(m):
            for k in range(p2):
                for r in range(r2):
                    V_full[j * p_total + p1 + k][r1 + r] = V2[j * p2 + k][r]
        
        for k in range(p2):
            for i in range(n):
                for r in range(r2):
                    W_full[(p1 + k) * n + i][r1 + r] = W2[k * n + i][r]
        
        return {"U": U_full, "V": V_full, "W": W_full}
    
    start_time = time.time()
    deadline = 120  # seconds
    
    # Start with schoolbook
    best_result = schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Strategy: M_{3,3,5} = M_{3,3,3} ⊕ M_{3,3,2}
    # Schoolbook M_{3,3,3} has rank 27
    # If we find M_{3,3,2} with rank r2 < 18, total rank 27 + r2 < 45
    
    # First, get schoolbook for M_{3,3,3}
    sb333 = schoolbook(3, 3, 3)
    U333, V333, W333 = sb333["U"], sb333["V"], sb333["W"]
    
    # Try to find low-rank M_{3,3,2}
    found_m332 = False
    for r2 in range(12, 18):  # Try ranks 12 through 17
        if time.time() - start_time > deadline - 30:
            break
        
        # Try gradient descent approach
        res = try_find_m332_gradient(r2, max_iter=2000, lr=0.05, seed=r2)
        if res:
            U2, V2, W2 = res
            # Combine
            result = combine_blocks(U333, V333, W333, U2, V2, W2, 3, 3, 3, 2)
            # Verify
            if verify(result["U"], result["V"], result["W"], 3, 3, 5):
                best_result = result
                best_rank = 27 + r2
                found_m332 = True
                break
            else:
                print(f"Warning: Combined result for r2={r2} failed verification")
        
        # Try brute force if gradient didn't work
        if time.time() - start_time > deadline - 20:
            break
        res = try_find_m332_bruteforce(r2, max_tries=1000, seed=r2 + 100)
        if res:
            U2, V2, W2 = res
            result = combine_blocks(U333, V333, W333, U2, V2, W2, 3, 3, 3, 2)
            if verify(result["U"], result["V"], result["W"], 3, 3, 5):
                best_result = result
                best_rank = 27 + r2
                found_m332 = True
                break
    
    return best_result
# EVOLVE-BLOCK-END
