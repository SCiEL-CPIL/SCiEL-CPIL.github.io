"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            # W index: W[k*n+i, c] corresponds to C[i,k]
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _is_nice_half_integer(x, tol=1e-9):
    """Check if x is close to an integer or half-integer."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    return abs(doubled - rounded) < tol

def _try_structured_search(n, m, p, start_time, deadline):
    """
    Search for low-rank decompositions using the structure:
    For fixed U, V columns, W is determined by:
    W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
    
    We search over structured U, V patterns with small integer entries.
    """
    np.random.seed(42)
    best_rank = n * m * p
    best_result = _build_schoolbook(n, m, p)
    
    # Precompute index mappings
    # U index: i*m + j for i in range(n), j in range(m)
    # V index: j*p + k for j in range(m), k in range(p)
    # W index: k*n + i for k in range(p), i in range(n)
    
    # Strategy: try ranks from low to high, with structured patterns
    # For each rank, try multiple random structured patterns
    
    # Generate a pool of basis patterns for U and V
    # Each pattern is a vector with entries in {-1, 0, 1}
    
    # Use a greedy approach: build up the decomposition term by term
    # Start with the full tensor and try to subtract rank-1 components
    
    # Alternative: direct search with structured random initialization
    
    # Let's try a systematic approach:
    # 1. For each target rank, generate random U, V with entries in {-1, 0, 1}
    # 2. Compute implied W
    # 3. Check if W is nice (integer or half-integer)
    # 4. If yes, verify the full decomposition
    
    # To increase success rate, use multiple random seeds and patterns
    
    # First, try very structured patterns based on known decompositions
    
    # For 4x4x4, Strassen uses rank 49. For 4x4x7, we can extend this.
    # Let's try ranks from 50 to 112
    
    # Generate diverse random patterns
    pattern_pool_size = 500
    U_patterns = []
    V_patterns = []
    
    # Create diverse patterns
    for _ in range(pattern_pool_size):
        u_pat = np.random.choice([-1, 0, 1], size=n*m)
        v_pat = np.random.choice([-1, 0, 1], size=m*p)
        U_patterns.append(u_pat)
        V_patterns.append(v_pat)
    
    # Also add some structured patterns
    # Block diagonal patterns
    for i in range(n):
        for j in range(m):
            u_pat = np.zeros(n*m)
            u_pat[i*m+j] = 1
            U_patterns.append(u_pat)
    for j in range(m):
        for k in range(p):
            v_pat = np.zeros(m*p)
            v_pat[j*p+k] = 1
            V_patterns.append(v_pat)
    
    # Try different target ranks
    # Start from lower ranks and work up
    for target_rank in range(50, 113):
        if time.time() > deadline:
            break
        
        # For this target rank, try to find a valid decomposition
        # Use random combinations of patterns
        
        max_attempts = 100 if target_rank < 80 else 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Randomly select target_rank patterns
            indices = np.random.choice(len(U_patterns), size=target_rank, replace=False)
            U_mat = np.array([U_patterns[idx] for idx in indices], dtype=float).T  # (n*m, target_rank)
            V_mat = np.array([V_patterns[idx] for idx in indices], dtype=float).T  # (m*p, target_rank)
            
            # Compute implied W
            W_mat = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                uc = U_mat[:, c]  # (n*m,)
                vc = V_mat[:, c]  # (m*p,)
                
                for k in range(p):
                    for i in range(n):
                        # sum_j U[i*m+j, c] * V[j*p+k, c]
                        denom = 0.0
                        for j in range(m):
                            denom += uc[i*m+j] * vc[j*p+k]
                        if abs(denom) < 1e-12:
                            valid = False
                            break
                        W_mat[k*n+i, c] = 1.0 / denom
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W is nice (integer or half-integer)
            if not all(_is_nice_half_integer(W_mat[idx, c]) for idx in range(p*n) for c in range(target_rank)):
                continue
            
            # Convert to exact half-integers
            U_list = [[int(round(U_mat[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_mat[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[float(int(round(W_mat[i, c] * 2))) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            # Verify
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we found a lower rank, update best
        # (We return immediately on first valid find for this rank)
    
    return best_result

def _try_greedy_subtraction(n, m, p, start_time, deadline):
    """
    Greedy approach: start with the full multiplication tensor and
    iteratively subtract rank-1 components.
    """
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want to represent T as sum of rank-1 tensors
    
    # Represent T as a 3D array
    T = np.ones((n, m, p))
    
    # Try to greedily find rank-1 approximations
    # A rank-1 tensor is u[i,j] * v[j,k] * w[k,i]
    # For the multiplication tensor, each "term" A[i,j]*B[j,k] contributes to C[i,k]
    
    # This approach is complex; skip for now and rely on the structured search
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    
    # Try to find a better decomposition
    try:
        result = _try_structured_search(n, m, p, start_time, deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
