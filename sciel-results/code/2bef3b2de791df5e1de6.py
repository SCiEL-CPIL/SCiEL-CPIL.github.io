"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = delta(i,j,k) for the standard multiplication
    # Actually, the multiplication tensor is: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] should be 1 if this term appears in the sum, else 0
    # In our indexing: T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] contributes to C[i,k]
    
    # Let's verify by computing the actual multiplication for random matrices
    for _ in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Reconstruct A and B factors
            A_c = np.zeros((n, m))
            for idx in range(n * m):
                i, j = idx // m, idx % m
                A_c[i, j] = uc[idx]
            B_c = np.zeros((m, p))
            for idx in range(m * p):
                j, k = idx // p, idx % p
                B_c[j, k] = vc[idx]
            # The contribution to C from this rank-1 term
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

def _try_strassen_style(n, m, p):
    """
    Try to find a lower-rank decomposition using structured patterns.
    We use a basis of rank-1 tensors with entries in {-1, 0, 1} and solve for coefficients.
    """
    start_time = time.time()
    best_result = _build_schoolbook(n, m, p)
    
    # For small matrices, we can try to find decompositions with rank < n*m*p
    # by using a set of structured rank-1 tensors and solving a linear system
    
    # The multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We want: T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    
    # Strategy: generate candidate rank-1 tensors with small integer entries
    # and solve for coefficients using least squares
    
    # For 4x4x7, the schoolbook rank is 112. Known bounds suggest we can do better.
    # Let's try ranks from 112 down to see what's feasible.
    
    # Generate a pool of structured rank-1 tensors
    # Each tensor is defined by (u, v, w) where:
    #   u: vector of length n*m with entries in {-1, 0, 1}
    #   v: vector of length m*p with entries in {-1, 0, 1}  
    #   w: vector of length p*n with entries in {-1, 0, 1}
    
    # This is a huge search space, so we'll use a greedy approach:
    # 1. Start with the full tensor
    # 2. Iteratively find the best rank-1 approximation and subtract
    
    # Actually, let's use a different approach: use known results and try to verify
    # For 4x4x4, the tensor rank is known to be 49 (Strassen-like)
    # For 4x4x7, we can pad or use a direct approach
    
    # Let's try a direct numerical approach: 
    # Set up the system and use iterative refinement
    
    # The constraint is: for all (i,j,k):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of n*m*p equations in rank*3 unknowns (but bilinear/cubic)
    
    # Alternative: fix U and V, solve for W (linear in W)
    # For fixed U, V: for each (k,i):
    #   W[k*n+i, c] appears in equations for all j:
    #   sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    #   => W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c] = 1
    #   => W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
    
    # So if we fix U and V, W is determined (where the denominator is nonzero)
    
    # Let's try: parameterize U and V with random small entries, compute W, check if it works
    # and if W has nice values (integers or half-integers)
    
    np.random.seed(42)
    
    # Try various ranks
    for target_rank in range(112, 50, -1):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        for attempt in range(20):
            if time.time() - start_time > DEADLINE_SECONDS - 5:
                break
            # Generate random U and V with small integer entries
            U_rand = np.random.randint(-2, 3, size=(n*m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m*p, target_rank))
            
            # Compute W
            W_rand = np.zeros((p*n, target_rank))
            valid = True
            for c in range(target_rank):
                for k in range(p):
                    for i in range(n):
                        denom = 0.0
                        for j in range(m):
                            denom += U_rand[i*m+j, c] * V_rand[j*p+k, c]
                        if abs(denom) < 1e-12:
                            valid = False
                            break
                        W_rand[k*n+i, c] = 1.0 / denom
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if W has nice values (close to integers or half-integers)
            W_check = W_rand * 2
            if not np.allclose(W_check, np.round(W_check), atol=1e-6):
                continue
            
            # Verify the decomposition
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
            W_list = [[int(round(W_rand[i, c] * 2)) / 2.0 for c in range(target_rank)] for i in range(p*n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    
    # Try to find a better decomposition
    try:
        result = _try_strassen_style(n, m, p)
        if result is not None:
            # Verify it's valid
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE_SECONDS:
        pass  # Just return what we have
    
    return best
# EVOLVE-BLOCK-END
