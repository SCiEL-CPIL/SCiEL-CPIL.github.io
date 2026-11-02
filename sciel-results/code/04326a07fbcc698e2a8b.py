"""Rewritten program using a vectorized construction approach for the schoolbook decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Create the factor matrices using vectorized operations
    # U[i*m + j, c] = 1 when c corresponds to (i,j,k)
    # V[j*p + k, c] = 1 when c corresponds to (i,j,k)  
    # W[k*n + i, c] = 1 when c corresponds to (i,j,k)
    
    # Generate all (i,j,k) triples
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    i_flat = i_vals.flatten()
    j_flat = j_vals.flatten()
    k_flat = k_vals.flatten()
    
    # Create U, V, W as sparse-like dense matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for c in range(rank):
        i, j, k = i_flat[c], j_flat[c], k_flat[c]
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
