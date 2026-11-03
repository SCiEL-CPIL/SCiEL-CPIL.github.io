import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Lower bound for rank
    min_rank = max(n * m, m * p, p * n)
    
    # We target the lower bound rank
    R = min_rank
    
    # We will try to find a solution with rank R.
    # Strategy: Randomly initialize U and V, solve for W linearly,
    # then optimize U and V to make W close to half-integers.
    
    np.random.seed(42)
    
    best_score = float('inf')
    best_U, best_V, best_W = None, None, None

    # Number of random restarts
    num_restarts = 50
    # Steps of gradient descent per restart
    num_steps = 100
    learning_rate = 0.01

    for restart in range(num_restarts):
        # Initialize U and V with small integers
        # Using integers helps with the "half-integer" target for W
        U = np.random.randint(-1, 2, size=(n * m, R)).astype(float)
        V = np.random.randint(-1, 2, size=(m * p, R)).astype(float)
        
        # Add some noise to break symmetries
        U += np.random.randn(n * m, R) * 0.1
        V += np.random.randn(m * p, R) * 0.1

        for step in range(num_steps):
            # 1. Compute W by solving linear systems
            W = np.zeros((p * n, R))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Construct A matrix of shape (m, R)
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # b is vector of ones
                    b = np.ones(m)
                    
                    # Solve A * w = b for w (shape R)
                    # Since A is m x R (5 x 25), it's underdetermined.
                    # We want the minimum norm solution.
                    try:
                        # w = pinv(A) @ b
                        w = np.linalg.lstsq(A, b, rcond=None)[0]
                        W[k * n + i, :] = w
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue

            # 2. Check if W is close to half-integers
            W_rounded = np.round(W * 2) / 2
            diff = W - W_rounded
            integrality_loss = np.sum(diff * diff)
            
            # 3. Check if the rounded W satisfies the equations exactly
            # Equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W_rounded[k*n+i, r] == 1
            # We can compute the max error
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W_rounded[k * n + i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-9:
                # Found an exact solution
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # 4. Gradient Descent step on U and V
            # We want to minimize integrality_loss + alpha * max_err (or sum of squared errors)
            # Let's use sum of squared errors for differentiability
            sum_sq_err = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            
            # This gradient calculation is expensive if done via finite differences.
            # Instead, we can just do a random perturbation if loss increases,
            # or use a simple heuristic.
            # Given the complexity, let's try a simpler "Simulated Annealing" like approach
            # or just check the current state.
            
            # Actually, calculating gradients for this non-linear system is hard.
            # Let's just use the current U, V as the best candidate if integrality_loss is low.
            if integrality_loss < best_score:
                best_score = integrality_loss
                best_U, best_V, best_W = U.copy(), V.copy(), W_rounded.copy()
                
                # If we are very close, try snapping and verifying again
                # (Already done above, but maybe with tighter tolerance?)
                
                # If integrality loss is extremely small, we might have a solution
                # that is numerically unstable.
                
            # Update U and V with small random noise (Random Walk)
            # This is a form of stochastic optimization
            U += np.random.randn(n * m, R) * learning_rate
            V += np.random.randn(m * p, R) * learning_rate

    # If no exact solution found, try to return the best candidate if it's close
    # But the task requires EXACT decomposition.
    # If we didn't find one, we must fallback.
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
