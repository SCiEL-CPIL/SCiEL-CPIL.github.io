import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Define the tensor support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                support.append((a, b, c))
    
    # All possible (a,b,c) triples
    all_triples = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
    support_set = set(support)
    
    # Try to find lower rank decompositions
    # Strategy: Use the fact that we can fix W to simplify the problem
    # For each candidate rank r, try to find a solution
    
    def verify(U, V, W, r):
        """Verify that U, V, W give a valid rank-r decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check all support entries
        for (a, b, c) in support:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > 1e-6:
                return False
        
        # Check all non-support entries
        for (a, b, c) in all_triples:
            if (a, b, c) not in support_set:
                val = np.sum(U[a] * V[b] * W[c])
                if abs(val) > 1e-6:
                    return False
        
        return True
    
    # Try ranks from 25 to 29
    # For each rank, try multiple random initializations
    
    for r in range(25, best_rank):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + r)
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Use gradient descent to minimize the residual
            # Residual = sum over all (a,b,c) of (computed - target)^2
            
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Compute residual and gradient
                # For each (a,b,c): residual = sum_r U[a,r]*V[b,r]*W[c,r] - target
                # d(residual)/d(U[a',r']) = V[b,r'] * W[c,r'] if a'=a else 0
                # etc.
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Compute for support entries (target=1)
                for (a, b, c) in support:
                    for r_idx in range(r):
                        u, v, w = U[a, r_idx], V[b, r_idx], W[c, r_idx]
                        # Contribution to residual
                        res = u * v * w
                        # Gradient
                        grad_U[a, r_idx] += 2 * (res - 1.0) * v * w
                        grad_V[b, r_idx] += 2 * (res - 1.0) * u * w
                        grad_W[c, r_idx] += 2 * (res - 1.0) * u * v
                
                # Compute for non-support entries (target=0)
                # This is expensive, so sample
                if iteration % 5 == 0:  # Every 5th iteration, check more non-support
                    sample_size = min(50, len(all_triples) - len(support))
                    non_support = [t for t in all_triples if t not in support_set]
                    np.random.shuffle(non_support)
                    for (a, b, c) in non_support[:sample_size]:
                        for r_idx in range(r):
                            u, v, w = U[a, r_idx], V[b, r_idx], W[c, r_idx]
                            res = u * v * w
                            grad_U[a, r_idx] += 2 * res * v * w
                            grad_V[b, r_idx] += 2 * res * u * w
                            grad_W[c, r_idx] += 2 * res * u * v
                
                # Update with learning rate
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a valid solution
            if verify(U.tolist(), V.tolist(), W.tolist(), r):
                found = True
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found:
            break
    
    return best_result
