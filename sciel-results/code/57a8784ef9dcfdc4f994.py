"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not U[0]:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_structured_search(n, m, p, deadline):
    """Try structured decompositions based on known techniques."""
    best = None
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Strategy: Try to find a decomposition with rank < n*m*p
    # Use a greedy approach: start with schoolbook, try to merge terms
    
    # First, let's try a structured approach based on the algebraic structure
    # For matrix multiplication, we can use the fact that:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    
    # Let's try to find a decomposition by solving a system of equations
    # for small ranks
    
    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 15, 15) = 15
    # But realistically, we need more
    
    for target_rank in range(15, 45):
        if time.time() > deadline:
            break
        
        # Try to find a rank-target_rank decomposition
        # We'll use a random initialization and then try to refine
        # This is a heuristic approach
        
        # Initialize with random sparse vectors
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small integer values
            U_init = np.zeros((n*m, target_rank), dtype=np.float64)
            V_init = np.zeros((m*p, target_rank), dtype=np.float64)
            W_init = np.zeros((p*n, target_rank), dtype=np.float64)
            
            # Fill with random values from {-1, 0, 1}
            for c in range(target_rank):
                # Randomly choose which entries are non-zero
                u_mask = np.random.random(n*m) < 0.3
                v_mask = np.random.random(m*p) < 0.3
                w_mask = np.random.random(p*n) < 0.3
                
                U_init[:, c] = np.where(u_mask, np.random.choice([-1, 1], n*m), 0)
                V_init[:, c] = np.where(v_mask, np.random.choice([-1, 1], m*p), 0)
                W_init[:, c] = np.where(w_mask, np.random.choice([-1, 1], p*n), 0)
            
            # Check if this random decomposition works (unlikely, but let's try)
            T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for c in range(target_rank):
                T_decomp += np.einsum('a,b,c->abc', U_init[:,c], V_init[:,c], W_init[:,c])
            
            if np.allclose(T_decomp, T_target, atol=1e-8):
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_init]
                V_list = [[float(x) for x in row] for row in V_init]
                W_list = [[float(x) for x in row] for row in W_init]
                result = {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
                if best is None or target_rank < best["rank"]:
                    best = result
                found = True
                break
        
        if found:
            break
    
    if best is None:
        return _schoolbook(n, m, p)
    return best

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    best_result = _schoolbook(n, m, p)
    
    # Try structured approaches
    structured = _try_structured_search(n, m, p, deadline)
    if structured and structured["rank"] < best_result["rank"]:
        best_result = structured
    
    return best_result
# EVOLVE-BLOCK-END
