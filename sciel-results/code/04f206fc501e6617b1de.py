import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12
    
    # Identify valid triples (i, j, k) -> (a, b, c)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    # Total number of triples in the tensor space
    total_triples = num_A * num_B * num_C
    valid_set = set(valid_triples)
    
    # Precompute indices for all triples to speed up gradient calculation
    # We only need to compute gradients for the error function.
    # E = sum_{all} (X - T)^2
    # grad_U[a,r] = sum_{b,c} (X[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
    
    def compute_gradients(U, V, W, target_rank):
        """
        Computes gradients of E = 0.5 * sum (X - T)^2 with respect to U, V, W.
        """
        # X[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We can compute X for all a,b,c using einsum
        # But T is sparse.
        # Let's compute X for all valid and invalid triples.
        # Actually, we can just compute the full tensor X and subtract T.
        
        # X has shape (num_A, num_B, num_C)
        # Using einsum: 'ar,br,cr->abc'
        X = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Create target tensor T
        # T is 0 everywhere except valid_triples where it is 1.
        # Instead of creating a full (6, 18, 12) array which is small (1296 elements),
        # we can just initialize zeros and set the valid ones.
        T = np.zeros((num_A, num_B, num_C))
        for a, b, c in valid_triples:
            T[a, b, c] = 1.0
            
        diff = X - T
        
        # Gradients
        # dE/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
        # dE/dV[b,r] = sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
        # dE/dW[c,r] = sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
        
        grad_U = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        grad_V = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        grad_W = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        
        return grad_U, grad_V, grad_W

    def error(U, V, W):
        X = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        # Only check valid triples for error? No, must check all.
        # But for speed in the loop, we can just sum squares of diff.
        # T is constant.
        # Let's precompute T once? No, it's small.
        T = np.zeros((num_A, num_B, num_C))
        for a, b, c in valid_triples:
            T[a, b, c] = 1.0
        diff = X - T
        return np.sum(diff * diff)

    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((num_A, rank_schoolbook))
    V_school = np.zeros((num_B, rank_schoolbook))
    W_school = np.zeros((num_C, rank_schoolbook))
    col = 0
    for a, b, c in valid_triples:
        U_school[a, col] = 1.0
        V_school[b, col] = 1.0
        W_school[c, col] = 1.0
        col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    best_err = 0.0

    # Try to find lower rank
    # We will try rank 35 and 34.
    # Use a simple gradient descent with restarts.
    
    for target_rank in [35, 34]:
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try a few random restarts
        for restart in range(10):
            if time.time() > deadline - 5:
                break
                
            np.random.seed(12345 + target_rank * 100 + restart)
            # Initialize with small random values
            U_curr = np.random.randn(num_A, target_rank) * 0.1
            V_curr = np.random.randn(num_B, target_rank) * 0.1
            W_curr = np.random.randn(num_C, target_rank) * 0.1
            
            # Scale initial guess to roughly match magnitude?
            # The target values are 1. If R=35, each term is roughly 1/35?
            # Let's try scaling by 1/sqrt(target_rank)
            scale = 1.0 / np.sqrt(target_rank)
            U_curr *= scale
            V_curr *= scale
            W_curr *= scale
            
            # Gradient Descent
            lr = 0.01
            prev_err = np.inf
            stalled = 0
            
            for step in range(500):
                if time.time() > deadline - 2:
                    break
                    
                grad_U, grad_V, grad_W = compute_gradients(U_curr, V_curr, W_curr, target_rank)
                
                # Normalize gradients to prevent blowup
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm < 1e-8:
                    break
                    
                # Adaptive learning rate?
                # Let's try a simple backtracking line search or just fixed small step.
                # Given the landscape, fixed small step is safer.
                
                U_curr -= lr * grad_U
                V_curr -= lr * grad_V
                W_curr -= lr * grad_W
                
                curr_err = error(U_curr, V_curr, W_curr)
                
                if curr_err < prev_err - 1e-6:
                    prev_err = curr_err
                    stalled = 0
                    # Increase LR slightly if improving fast?
                    # lr *= 1.01
                else:
                    stalled += 1
                    # Decrease LR if stalled
                    lr *= 0.9
                    # Undo step
                    U_curr += lr * grad_U
                    V_curr += lr * grad_V
                    W_curr += lr * grad_W
                    if stalled > 10:
                        break
                
                if curr_err < 1e-10:
                    found = True
                    break
                    
            if found:
                best_rank = target_rank
                best_U, best_V, best_W = U_curr.copy(), V_curr.copy(), W_curr.copy()
                break
                
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
