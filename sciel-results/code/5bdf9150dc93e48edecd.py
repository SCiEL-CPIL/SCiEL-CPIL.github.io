import time
import sys
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen 2x2 * 2x2 decomposition (rank 7)
    U_local = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ], dtype=float)
    
    V_local = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ], dtype=float)
    
    W_local = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ], dtype=float)
    
    # Global mapping for A1 (cols 0,1 of A): local (i,j) -> global i*4+j
    a1_map = [0, 1, 4, 5]
    # Global mapping for A2 (cols 2,3 of A): local (i,j) -> global i*4+2+j
    a2_map = [2, 3, 6, 7]
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(actual - 1.0) > 1e-10:
                        return False
        return True
    
    def try_rank(rank):
        """Try to find a valid decomposition with given rank using optimization."""
        if time.time() - start_time > 135:
            return None
        
        # Start with Strassen-based initialization if rank >= 56
        if rank >= 56:
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Build the 56-rank Strassen solution
            for c_pair in range(4):
                base_r = c_pair * 7
                b1_map = [0*8 + 2*c_pair, 0*8 + 2*c_pair + 1,
                          1*8 + 2*c_pair, 1*8 + 2*c_pair + 1]
                c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                         (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        if U_local[la, l] != 0:
                            U[a1_map[la], r] += U_local[la, l]
                    for lb in range(4):
                        if V_local[lb, l] != 0:
                            V[b1_map[lb], r] += V_local[lb, l]
                    for lc in range(4):
                        if W_local[lc, l] != 0:
                            W[c_map[lc], r] += W_local[lc, l]
            
            for c_pair in range(4):
                base_r = 28 + c_pair * 7
                b2_map = [2*8 + 2*c_pair, 2*8 + 2*c_pair + 1,
                          3*8 + 2*c_pair, 3*8 + 2*c_pair + 1]
                c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                         (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
                
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        if U_local[la, l] != 0:
                            U[a2_map[la], r] += U_local[la, l]
                    for lb in range(4):
                        if V_local[lb, l] != 0:
                            V[b2_map[lb], r] += V_local[lb, l]
                    for lc in range(4):
                        if W_local[lc, l] != 0:
                            W[c_map[lc], r] += W_local[lc, l]
            
            if verify(U, V, W, rank):
                return (U, V, W)
        
        # For rank < 56 or if Strassen didn't work, try optimization
        # This is a non-convex problem, so we use multiple random restarts
        best_result = None
        
        for attempt in range(5):
            if time.time() - start_time > 135:
                break
            
            # Random initialization
            np.random.seed(42 + attempt)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to avoid overflow/underflow
            U = U / np.sqrt(np.sum(U**2))
            V = V / np.sqrt(np.sum(V**2))
            W = W / np.sqrt(np.sum(W**2))
            
            # Simple gradient-free local search
            for iteration in range(100):
                if time.time() - start_time > 135:
                    break
                
                # Check if valid
                if verify(U, V, W, rank):
                    return (U, V, W)
                
                # Perturb one matrix
                matrix = np.random.choice(['U', 'V', 'W'])
                if matrix == 'U':
                    row = np.random.randint(n * m)
                    col = np.random.randint(rank)
                    U[row, col] += 0.1 * np.random.randn()
                elif matrix == 'V':
                    row = np.random.randint(m * p)
                    col = np.random.randint(rank)
                    V[row, col] += 0.1 * np.random.randn()
                else:
                    row = np.random.randint(p * n)
                    col = np.random.randint(rank)
                    W[row, col] += 0.1 * np.random.randn()
        
        return None
    
    # Try ranks from 48 to 63
    for rank in range(48, 64):
        result = try_rank(rank)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the 56-rank Strassen solution
    U = np.zeros((n * m, 56), dtype=float)
    V = np.zeros((m * p, 56), dtype=float)
    W = np.zeros((p * n, 56), dtype=float)
    
    for c_pair in range(4):
        base_r = c_pair * 7
        b1_map = [0*8 + 2*c_pair, 0*8 + 2*c_pair + 1,
                  1*8 + 2*c_pair, 1*8 + 2*c_pair + 1]
        c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                 (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a1_map[la], r] += U_local[la, l]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[b1_map[lb], r] += V_local[lb, l]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[c_map[lc], r] += W_local[lc, l]
    
    for c_pair in range(4):
        base_r = 28 + c_pair * 7
        b2_map = [2*8 + 2*c_pair, 2*8 + 2*c_pair + 1,
                  3*8 + 2*c_pair, 3*8 + 2*c_pair + 1]
        c_map = [(2*c_pair)*2 + 0, (2*c_pair+1)*2 + 0,
                 (2*c_pair)*2 + 1, (2*c_pair+1)*2 + 1]
        
        for l in range(7):
            r = base_r + l
            for la in range(4):
                if U_local[la, l] != 0:
                    U[a2_map[la], r] += U_local[la, l]
            for lb in range(4):
                if V_local[lb, l] != 0:
                    V[b2_map[lb], r] += V_local[lb, l]
            for lc in range(4):
                if W_local[lc, l] != 0:
                    W[c_map[lc], r] += W_local[lc, l]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
