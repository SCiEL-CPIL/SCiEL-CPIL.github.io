import numpy as np
import time
from itertools import product

def construct():
    """
    Attempts to find a rank-49 exact tensor decomposition for 2x5 * 5x5 matrix multiplication
    using a randomized Newton-Raphson solver. Falls back to rank 50 if not found in time.
    """
    n, m, p = 2, 5, 5
    target_rank = 49
    fallback_rank = n * m * p  # 50
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit to be safe within 120s wall clock
    
    def get_fallback():
        rank = fallback_rank
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
        return {'U': U, 'V': V, 'W': W}

    def check_solution(U, V, W):
        """
        Verifies if U, V, W form a valid decomposition.
        Returns True if valid, False otherwise.
        """
        # Dimensions
        nm = n * m
        mp = m * p
        pn = p * n
        r = len(U[0]) if U else 0
        
        # Convert to numpy for fast tensor contraction
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # The tensor T[i, j, k] should equal sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W index is k*n + i.
        
        # We can reshape W to (p, n, r)
        W_reshaped = W_np.reshape(p, n, r)
        
        # Construct the tensor from factors
        # T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        
        # Efficient way:
        # For each r, we have outer products? No, it's a sum of products.
        # Let's just loop over r and add contributions, or use einsum.
        # T_calc = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_reshaped)
        
        try:
            T_calc = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_reshaped)
        except Exception:
            return False
            
        # Target tensor T_target[i, j, k] is 1 if this corresponds to a valid multiplication term?
        # Matrix multiplication C = A B where A is 2x5, B is 5x5.
        # C[i, k] = sum_j A[i, j] B[j, k]
        # The tensor T is defined such that T[i, j, k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
        # In standard matrix multiplication, every pair (i, j, k) is a valid term.
        # So T_target[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        T_target = np.ones((n, m, p), dtype=np.float64)
        
        # Check exact match (with small tolerance for float errors, but we want exact integers/half-integers)
        # Since the inputs are integers/half-integers, the result should be exactly 1.
        # However, Newton's method might leave tiny residuals.
        # We will check if they are close to integers, and if the integer values match 1.
        
        # First, check if values are close to integers
        if not np.allclose(T_calc, np.round(T_calc), atol=1e-6):
            return False
            
        T_int = np.round(T_calc).astype(int)
        
        if not np.array_equal(T_int, T_target.astype(int)):
            return False
            
        return True

    def newton_step(U, V, W, rank):
        """
        Performs one Newton-Raphson step to minimize the error.
        Error E_r = sum_{i,j,k} (T_target[i,j,k] - T_calc[i,j,k]) * delta_{r} ?
        Actually, we want to solve F(U,V,W) = 0 where F is the vector of residuals.
        Residual for each (i,j,k) is R_{ijk} = 1 - sum_r U_{ijr} V_{jkr} W_{kir}.
        We have n*m*p = 50 residuals and 3 * 25 * rank variables.
        For rank=49, variables = 3 * 25 * 49 = 3675. Residuals = 50.
        This is a highly underdetermined system.
        
        Standard Newton's method on the residuals requires inverting a Jacobian of size 50x3675?
        No, we can use the Gauss-Newton method or simply project the gradient.
        
        Let's use a simpler approach: 
        The equation is linear in each factor if the other two are fixed.
        We can use coordinate descent or block coordinate updates.
        However, to get exact integers, we need a specific structure.
        
        Alternative Strategy: 
        Since we suspect rank 49 is possible, we can look for a "sparse" solution.
        Most known low-rank decompositions use 0s and 1s (or +/-1).
        
        Let's try a randomized search over a restricted set of values {-1, 0, 1} for the factors,
        but that space is huge.
        
        Better: Use the structure of the problem.
        The tensor is the "matrix multiplication tensor".
        
        Let's try a different optimization: 
        Minimize ||T_calc - T_target||^2.
        Gradient descent with respect to U, V, W.
        grad_U[i,j,r] = - sum_{k} R_{ijk} V_{jkr} W_{kir}
        grad_V[j,k,r] = - sum_{i} R_{ijk} U_{ijr} W_{kir}
        grad_W[k,i,r] = - sum_{j} R_{ijk} U_{ijr} V_{jkr}
        
        We can do a few steps of gradient descent to get close to a solution,
        then snap to nearest integers and verify.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # Learning rate
        lr = 0.1
        
        # Compute T_calc
        T_calc = np.einsum('ijr,jkr,kir->ijk', U_np, V_np, W_reshaped)
        T_target = np.ones((n, m, p), dtype=np.float64)
        Residual = T_target - T_calc
        
        # Gradients
        # d/dU[i,j,r] sum_k Residual[i,j,k] V[j,k,r] W[k,i,r]
        # Note: Residual is (n, m, p). V is (m, p, rank). W is (p, n, rank).
        
        grad_U = np.einsum('ijk,jkr,kir->ijr', Residual, V_np, W_reshaped)
        grad_V = np.einsum('ijk,ijr,kir->jkr', Residual, U_np, W_reshaped)
        grad_W = np.einsum('ijk,ijr,jkr->kir', Residual, U_np, V_np)
        
        # Update
        U_np -= lr * grad_U
        V_np -= lr * grad_V
        W_np -= lr * grad_W
        
        return U_np.tolist(), V_np.tolist(), W_np.tolist()

    # Attempt search for rank 49
    best_result = None
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Initialize with random values
        # We can try to initialize close to a known decomposition or random.
        # Random initialization in [-1, 1]
        U = [[np.random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[np.random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[np.random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Run gradient descent for a few steps
        for step in range(100):
            if time.time() > deadline:
                break
                
            U, V, W = newton_step(U, V, W, target_rank)
            
            # Periodically check if we are close to an integer solution
            if step % 10 == 0:
                # Snap to nearest integer
                U_int = [[int(round(x)) for x in row] for row in U]
                V_int = [[int(round(x)) for x in row] for row in V]
                W_int = [[int(round(x)) for x in row] for row in W]
                
                # Check if this integer solution is valid
                if check_solution(U_int, V_int, W_int):
                    return {'U': U_int, 'V': V_int, 'W': W_int}
        
        # After GD, check the final state snapped to integers
        U_int = [[int(round(x)) for x in row] for row in U]
        V_int = [[int(round(x)) for x in row] for row in V]
        W_int = [[int(round(x)) for x in row] for row in W]
        
        if check_solution(U_int, V_int, W_int):
            return {'U': U_int, 'V': V_int, 'W': W_int}

    # If rank 49 not found, try a structured search for rank 49 based on Strassen-like blocks
    # The inspiration mentioned Strassen block decomposition reduced to 49.
    # Let's explicitly construct the rank 49 solution if we can find it.
    # One known rank 49 decomposition for 2x5x5:
    # Split 2x5 into two 2x2 blocks and one 2x1? No, 5 is not divisible by 2.
    # 5 = 2 + 2 + 1.
    # A = [A1 A2 A3] where A1, A2 are 2x2, A3 is 2x1.
    # B = [B1 B2 B3] where B1, B2 are 5x2, B3 is 5x1.
    # This mixing is complex.
    
    # Given the time constraint and the difficulty of finding rank 49 from scratch,
    # and the fact that the "Strassen block" inspiration scored 1.02 (implying rank 49 was found),
    # I will attempt to reconstruct a specific rank 49 decomposition if possible,
    # or fall back to 50.
    
    # Since I cannot reliably generate the specific rank 49 structure in code without it being hardcoded,
    # and the gradient descent might not converge to the exact integer lattice point,
    # I will return the fallback rank 50 to ensure a valid score.
    # The search above is a "shot in the dark" that might hit rank 49.
    
    return get_fallback()
