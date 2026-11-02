"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p, rank):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    # Build the multiplication tensor T[i,j,k] = 1 (all ones for this bilinear map)
    # Actually, matrix mult: (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k
    # The decomposition requires: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    # Wait, the W index is [k*n+i, r] per the trace convention.
    # Let me re-read: W[k*n+i, r]
    # U has shape (n*m, rank), indexed by (i*m+j)
    # V has shape (m*p, rank), indexed by (j*p+k)
    # W has shape (p*n, rank), indexed by (k*n+i)
    
    # The tensor equation: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_bruteforce(n, m, p, rank, timeout=120):
    """
    Try to find a rank decomposition with small entries.
    Uses randomized search with local refinement.
    """
    deadline = time.time() + timeout
    
    # We need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Strategy: try random rank-r matrices over {0, 0.5, 1, 1.5, 2} and check.
    
    # First, let's try a structured approach based on known lower bounds.
    # For 2x3x8: schoolbook is 48. Known results suggest rank around 40-45 might work.
    # Let's try a more systematic search.
    
    # Approach: Use the fact that for small matrices, we can set up the system
    # and try to find sparse solutions.
    
    # Number of equations: n*m*p = 48
    # Number of unknowns per column: n*m + m*p + p*n = 6 + 24 + 16 = 46
    # So each column has 46 unknowns but contributes to 48 equations.
    # We need rank columns to collectively satisfy 48 equations.
    
    # Let's try rank = 40 and see if we can find a solution.
    # Use a greedy/iterative approach.
    
    best_result = None
    
    # Try various rank values from high to low
    for try_rank in range(min(rank, 48), 35, -1):
        if time.time() > deadline - 5:
            break
            
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Initialize with random sparse matrices
            scale = random.choice([1, 0.5, 1.0, 1.5, 2.0])
            
            U = np.zeros((n*m, try_rank), dtype=float)
            V = np.zeros((m*p, try_rank), dtype=float)
            W = np.zeros((p*n, try_rank), dtype=float)
            
            # Randomly fill with small probability of non-zero
            for r in range(try_rank):
                # Each column gets some random entries
                for idx in range(n*m):
                    if random.random() < 0.3:
                        U[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2, -0.5, -1.5])
                for idx in range(m*p):
                    if random.random() < 0.2:
                        V[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2, -0.5, -1.5])
                for idx in range(p*n):
                    if random.random() < 0.3:
                        W[idx, r] = random.choice([-1, 0.5, 1, 1.5, 2, -0.5, -1.5])
            
            # Check the decomposition
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0.0
                        for c in range(try_rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-8:
                            valid = False
                            break
            if valid:
                # Convert to list format
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 150
    
    # First, try to find a better decomposition than schoolbook
    # Schoolbook rank is n*m*p = 48
    
    # Known: For 2x3x8, there are results suggesting rank can be less than 48.
    # Let's try a structured approach first.
    
    # Try a specific pattern based on Strassen-like ideas adapted to this shape.
    # Actually, let's just do a smart search.
    
    # First fallback: schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    
    # Try to find better decomposition
    result = _try_rank_bruteforce(n, m, p, n*m*p, timeout=140)
    
    if result is not None:
        # Verify it
        rank = len(result["U"][0])
        if _verify(result["U"], result["V"], result["W"], n, m, p, rank):
            return result
    
    # Fallback to schoolbook
    return schoolbook
# EVOLVE-BLOCK-END
