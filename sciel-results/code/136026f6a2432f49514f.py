import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin before 140s limit

    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    # Build the target tensor for 2x3x6 multiplication
    # T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i

    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.add((a, b, c))

    # Schoolbook solution: rank = n*m*p = 36
    best_rank = n * m * p
    rank = best_rank
    U = np.zeros((num_A, rank))
    V = np.zeros((num_B, rank))
    W = np.zeros((num_C, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Strategy: Try to find a rank < 36 decomposition.
    # Given the difficulty, we limit the search time significantly.
    # If we don't find a better rank quickly, we return the schoolbook solution.
    
    # Heuristic Search for Rank 35, 34, 33...
    # We use a faster, simpler local search if time permits.
    
    def verify_decomposition(U, V, W, rank):
        max_err = 0.0
        # Check valid triples
        for (a, b, c) in valid_triples:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
                if max_err > 1e-6:
                    return False
        # Check invalid triples (sample)
        np.random.seed(42)
        for _ in range(500):
            a = np.random.randint(0, num_A)
            b = np.random.randint(0, num_B)
            c = np.random.randint(0, num_C)
            if (a, b, c) not in valid_triples:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                err = abs(val)
                if err > max_err:
                    max_err = err
                    if max_err > 1e-6:
                        return False
        return max_err < 1e-6

    # Try a few random restarts for lower ranks, but very briefly
    # This is a "shot in the dark" to see if a low rank solution exists and is easy to find.
    # If it's hard, we skip to schoolbook.
    
    for target_rank in [35, 34, 33]:
        if time.time() > deadline - 30:
            break
            
        found = False
        for restart in range(5): # Reduced restarts
            if time.time() > deadline - 25:
                break
                
            np.random.seed(10000 + target_rank * 10 + restart)
            U_curr = np.random.randn(num_A, target_rank) * 0.5
            V_curr = np.random.randn(num_B, target_rank) * 0.5
            W_curr = np.random.randn(num_C, target_rank) * 0.5
            
            # Simple gradient descent with very few steps
            lr = 0.05
            for step in range(100): # Reduced steps
                if time.time() > deadline - 20:
                    break
                    
                grad_U = np.zeros_like(U_curr)
                grad_V = np.zeros_like(V_curr)
                grad_W = np.zeros_like(W_curr)
                
                # Compute gradients only on valid triples for speed
                for (a, b, c) in valid_triples:
                    val = np.sum(U_curr[a, :] * V_curr[b, :] * W_curr[c, :])
                    err = val - 1.0
                    for r in range(target_rank):
                        grad_U[a, r] += err * V_curr[b, r] * W_curr[c, r]
                        grad_V[b, r] += err * U_curr[a, r] * W_curr[c, r]
                        grad_W[c, r] += err * U_curr[a, r] * V_curr[b, r]
                
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                lr *= 0.99
                
                # Check convergence occasionally
                if step % 10 == 0:
                    if verify_decomposition(U_curr, V_curr, W_curr, target_rank):
                        found = True
                        break
            
            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                break
        
        if best_rank == target_rank:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
