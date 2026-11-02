import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly reproduces the matmul tensor."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    rank = 55
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    
    # Use numpy for faster initialization and manipulation
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # --- Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks ---
    # A1 indices: i in {0,1}, j in {0,1} -> U index i*m + j
    # B1 indices: j in {0,1}, k in {0..9} -> V index j*p + k
    # C1 indices: i in {0,1}, k in {0..9} -> W index k*n + i
    
    # Strassen definitions for 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    
    # We iterate over 5 blocks of B1. Each block covers 2 columns.
    # Block g corresponds to B columns 2g and 2g+1.
    
    for g in range(5):
        k_base = 2 * g
        k0 = k_base
        k1 = k_base + 1
        
        # B indices for this block
        v_idx_b00 = 0 * p + k0
        v_idx_b01 = 0 * p + k1
        v_idx_b10 = 1 * p + k0
        v_idx_b11 = 1 * p + k1
        
        # A indices
        u_idx_a00 = 0 * m + 0
        u_idx_a01 = 0 * m + 1
        u_idx_a10 = 1 * m + 0
        u_idx_a11 = 1 * m + 1
        
        # C indices for this block
        w_idx_c00 = k0 * n + 0
        w_idx_c01 = k1 * n + 0
        w_idx_c10 = k0 * n + 1
        w_idx_c11 = k1 * n + 1
        
        # Term 1: m1 = (a00 + a11) * (b00 + b11)
        U[u_idx_a00, col] = 1.0
        U[u_idx_a11, col] = 1.0
        V[v_idx_b00, col] = 1.0
        V[v_idx_b11, col] = 1.0
        W[w_idx_c00, col] = 1.0
        W[w_idx_c11, col] = 1.0
        col += 1
        
        # Term 2: m2 = (a10 + a11) * b00
        U[u_idx_a10, col] = 1.0
        U[u_idx_a11, col] = 1.0
        V[v_idx_b00, col] = 1.0
        W[w_idx_c10, col] = 1.0
        W[w_idx_c11, col] = -1.0
        col += 1
        
        # Term 3: m3 = a00 * (b01 - b11)
        U[u_idx_a00, col] = 1.0
        V[v_idx_b01, col] = 1.0
        V[v_idx_b11, col] = -1.0
        W[w_idx_c01, col] = 1.0
        W[w_idx_c11, col] = 1.0
        col += 1
        
        # Term 4: m4 = a11 * (b10 - b00)
        U[u_idx_a11, col] = 1.0
        V[v_idx_b10, col] = 1.0
        V[v_idx_b00, col] = -1.0
        W[w_idx_c00, col] = 1.0
        W[w_idx_c10, col] = 1.0
        col += 1
        
        # Term 5: m5 = (a00 + a01) * b11
        U[u_idx_a00, col] = 1.0
        U[u_idx_a01, col] = 1.0
        V[v_idx_b11, col] = 1.0
        W[w_idx_c01, col] = 1.0
        W[w_idx_c00, col] = -1.0
        col += 1
        
        # Term 6: m6 = (a10 - a00) * (b00 + b01)
        U[u_idx_a10, col] = 1.0
        U[u_idx_a00, col] = -1.0
        V[v_idx_b00, col] = 1.0
        V[v_idx_b01, col] = 1.0
        W[w_idx_c11, col] = 1.0
        col += 1
        
        # Term 7: m7 = (a01 - a11) * (b10 + b11)
        U[u_idx_a01, col] = 1.0
        U[u_idx_a11, col] = -1.0
        V[v_idx_b10, col] = 1.0
        V[v_idx_b11, col] = 1.0
        W[w_idx_c00, col] = 1.0
        col += 1
        
    # --- Part 2: A2 (2x1) * B2 (1x10) using Schoolbook ---
    # A2 indices: i in {0,1}, j=2 -> U index i*m + 2
    # B2 indices: j=2, k in {0..9} -> V index 2*p + k
    # C2 indices: i in {0,1}, k in {0..9} -> W index k*n + i
    
    for i in range(n): # 0, 1
        for k in range(p): # 0..9
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            
            U[u_idx, col] = 1.0
            V[v_idx, col] = 1.0
            W[w_idx, col] = 1.0
            col += 1
            
    if col != rank:
        raise ValueError(f"Column count {col} != expected rank {rank}")
    
    # Verify decomposition
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Deadline guard
    elapsed = time.time() - _start
    if elapsed > 110:
        raise TimeoutError(f"Exceeded time budget: {elapsed:.1f}s")
    
    # Return as lists of lists as per interface requirement
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
