"""Program for matrix multiplication tensor decomposition 2x5 by 5x6."""
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Build the full multiplication tensor of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Try to find a low-rank decomposition using iterative CP decomposition
    # We'll use a simple power-iteration style approach to minimize rank
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Start with a random initialization and try to reduce rank
    # We'll try various rank values and use a greedy approach
    
    for target_rank in range(1, n * m * p + 1):
        # Try multiple random restarts
        found = False
        for attempt in range(20):
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Iterative refinement: project T onto the current subspace
            # and update each factor
            for iteration in range(100):
                # Update U: for each column, project T onto V and W
                new_U = np.zeros_like(U)
                for r in range(target_rank):
                    # T contracted with V[:, r] and W[:, r]
                    # T is (n*m, m*p, p*n), V is (m*p, rank), W is (p*n, rank)
                    # For each r, we want U[:, r] such that
                    # sum_r U[:, r] * V[:, r] * W[:, r] approximates T
                    # This is a 3-way tensor, so we do:
                    # new_U[:, r] = T * V[:, r] * W[:, r] (tensor contraction)
                    # T * V[:, r] gives (n*m, p*n), then * W[:, r] gives (n*m,)
                    TV = np.einsum('ijk,j->ik', T, V[:, r])
                    new_U[:, r] = np.einsum('ik,k->i', TV, W[:, r])
                
                # Update V
                new_V = np.zeros_like(V)
                for r in range(target_rank):
                    TW = np.einsum('ijk,k->ij', T, W[:, r])
                    new_V[:, r] = np.einsum('ij,i->j', TW, U[:, r])
                
                # Update W
                new_W = np.zeros_like(W)
                for r in range(target_rank):
                    TU = np.einsum('ijk,i->jk', T, U[:, r])
                    new_W[:, r] = np.einsum('jk,k->j', TU, V[:, r])
                
                # Normalize to prevent divergence
                for r in range(target_rank):
                    norm_u = np.linalg.norm(U[:, r])
                    norm_v = np.linalg.norm(V[:, r])
                    norm_w = np.linalg.norm(W[:, r])
                    if norm_u > 0 and norm_v > 0 and norm_w > 0:
                        scale = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                        U[:, r] /= norm_u * scale
                        V[:, r] /= norm_v * scale
                        W[:, r] /= norm_w * scale
                    elif norm_u > 0:
                        U[:, r] /= norm_u
                    elif norm_v > 0:
                        V[:, r] /= norm_v
                    elif norm_w > 0:
                        W[:, r] /= norm_w
                
                U, V, W = new_U, new_V, new_W
                
                # Check convergence
                if iteration > 5:
                    prev_U, prev_V, prev_W = U.copy(), V.copy(), W.copy()
                    diff = np.linalg.norm(U - prev_U) + np.linalg.norm(V - prev_V) + np.linalg.norm(W - prev_W)
                    if diff < 1e-8:
                        break
            
            # Check if the decomposition is exact
            # Reconstruct T from U, V, W
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T - T_recon)
            if error < 1e-6:
                found = True
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if found:
            break
    
    # If no low-rank decomposition found, use the trivial rank-60 decomposition
    if best_U is None:
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
    else:
        # Convert numpy arrays to lists of floats
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
