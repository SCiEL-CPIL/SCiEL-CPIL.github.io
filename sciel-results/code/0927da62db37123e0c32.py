"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def make_schoolbook():
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

    best_result = make_schoolbook()
    best_rank = n * m * p

    def tensor_dot(U, V, W, r):
        """Compute the tensor contraction efficiently using einsum."""
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return T

    def check_decomposition(U, V, W, r, tol=1e-9):
        """Check if the decomposition is exact."""
        T = tensor_dot(U, V, W, r)
        return np.allclose(T, 1.0, atol=tol, rtol=0)

    def try_polish(U, V, W, r, steps=200, lr=0.01):
        """High-precision polish pass to refine near-solutions."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            T = tensor_dot(U, V, W, r)
            residual = T - 1.0
            max_res = np.max(np.abs(residual))
            
            if max_res < 1e-12:
                break
            
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            grad_U = np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
            grad_V = np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
            grad_W = np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)
            
            # Adaptive learning rate
            eff_lr = lr * max(0.1, 1e-6 / (max_res + 1e-15))
            
            U -= eff_lr * grad_U.reshape(n*m, r)
            V -= eff_lr * grad_V.reshape(m*p, r)
            W -= eff_lr * grad_W.reshape(p*n, r)
        
        return U, V, W

    def structured_search(r, seed=42):
        """Try structured initializations that exploit matrix multiplication structure."""
        np.random.seed(seed)
        
        # Try various structured patterns
        patterns = []
        
        # Pattern 1: Block-diagonal structure
        # Try to group terms by row/column blocks
        
        # Pattern 2: Random but with structure
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with some structure
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Assign each rank component to a subset of (i,j,k) indices
            # This creates a structured initialization
            for c in range(r):
                # Randomly choose which (i,j,k) this term covers
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
            
            # Polish
            U, V, W = try_polish(U, V, W, r, steps=100, lr=0.05)
            
            if check_decomposition(U, V, W, r, tol=1e-9):
                return U, V, W
        
        return None

    # Search for lower rank decompositions
    # Start from just below schoolbook and work down
    min_rank_to_try = max(20, best_rank - 50)
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # Try structured search
        result = structured_search(r, seed=42 + r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # Continue searching for even lower rank
            continue
        
        # If time allows, also try pure random with gradient descent
        if time.time() < deadline - 30:
            for trial in range(3):
                if time.time() > deadline:
                    break
                
                np.random.seed(1000 + r + trial)
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
                
                # Run gradient descent
                for step in range(200):
                    if time.time() > deadline:
                        break
                    
                    T = tensor_dot(U, V, W, r)
                    residual = T - 1.0
                    max_res = np.max(np.abs(residual))
                    
                    if max_res < 1e-10:
                        break
                    
                    U3 = U.reshape(n, m, r)
                    V3 = V.reshape(m, p, r)
                    W3 = W.reshape(p, n, r)
                    
                    grad_U = np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
                    grad_V = np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
                    grad_W = np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)
                    
                    U -= 0.05 * grad_U.reshape(n*m, r)
                    V -= 0.05 * grad_V.reshape(m*p, r)
                    W -= 0.05 * grad_W.reshape(p*n, r)
                
                if check_decomposition(U, V, W, r, tol=1e-9):
                    best_rank = r
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
    
    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
