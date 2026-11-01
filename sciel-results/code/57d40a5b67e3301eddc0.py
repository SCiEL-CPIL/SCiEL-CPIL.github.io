"""
Refined program for the suite task.
Uses Strassen's algorithm for rank-7 decomposition of 2x2 matrix multiplication.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal (rank 7) tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm.
    
    Returns:
        dict with keys 'U', 'V', 'W' containing the factor matrices.
        U: (n*m, rank) = (4, 7)
        V: (m*p, rank) = (4, 7)
        W: (p*n, rank) = (4, 7)
        
    Indexing convention:
    U[i*m + j, r]
    V[j*p + k, r]
    W[k*n + i, r]
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin below 120s limit
    
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize matrices with zeros using NumPy
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to get flat indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen's 7 scalar products:
    # Let A = [[a00, a01], [a10, a11]]
    # Let B = [[b00, b01], [b10, b11]]
    # C = A * B = [[c00, c01], [c10, c11]]
    
    # m0 = (a00 + a11) * (b00 + b11)
    U[u_idx(0,0), 0] = 1
    U[u_idx(1,1), 0] = 1
    V[v_idx(0,0), 0] = 1
    V[v_idx(1,1), 0] = 1
    W[w_idx(0,0), 0] = 1  # contributes to c00
    W[w_idx(1,1), 0] = 1  # contributes to c11
    
    # m1 = (a10 + a11) * b00
    U[u_idx(1,0), 1] = 1
    U[u_idx(1,1), 1] = 1
    V[v_idx(0,0), 1] = 1
    W[w_idx(0,1), 1] = 1  # contributes to c10
    W[w_idx(1,1), 1] = -1 # contributes to c11
    
    # m2 = a00 * (b01 - b11)
    U[u_idx(0,0), 2] = 1
    V[v_idx(0,1), 2] = 1
    V[v_idx(1,1), 2] = -1
    W[w_idx(1,0), 2] = 1  # contributes to c01
    W[w_idx(1,1), 2] = 1  # contributes to c11
    
    # m3 = a11 * (b10 - b00)
    U[u_idx(1,1), 3] = 1
    V[v_idx(1,0), 3] = 1
    V[v_idx(0,0), 3] = -1
    W[w_idx(0,0), 3] = 1  # contributes to c00
    W[w_idx(0,1), 3] = 1  # contributes to c10
    
    # m4 = (a00 + a01) * b11
    U[u_idx(0,0), 4] = 1
    U[u_idx(0,1), 4] = 1
    V[v_idx(1,1), 4] = 1
    W[w_idx(0,0), 4] = -1 # contributes to c00
    W[w_idx(1,0), 4] = 1  # contributes to c01
    
    # m5 = (a10 - a00) * (b00 + b01)
    U[u_idx(1,0), 5] = 1
    U[u_idx(0,0), 5] = -1
    V[v_idx(0,0), 5] = 1
    V[v_idx(0,1), 5] = 1
    W[w_idx(1,1), 5] = 1  # contributes to c11
    
    # m6 = (a01 - a11) * (b10 + b11)
    U[u_idx(0,1), 6] = 1
    U[u_idx(1,1), 6] = -1
    V[v_idx(1,0), 6] = 1
    V[v_idx(1,1), 6] = 1
    W[w_idx(0,0), 6] = 1  # contributes to c00

    # Check deadline (though computation is instantaneous)
    if time.time() - start_time > deadline:
        raise TimeoutError("Deadline exceeded")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
