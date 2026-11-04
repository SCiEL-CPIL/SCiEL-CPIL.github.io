"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120
    t0 = time.time()
    deadline = t0 + 140.0  # leave 10s safety margin before 150s kill
    
    # Best solution so far (schoolbook)
    best_rank = schoolbook_rank
    best_U = np.zeros((n * m, schoolbook_rank), dtype=float)
    best_V = np.zeros((m * p, schoolbook_rank), dtype=float)
    best_W = np.zeros((p * n, schoolbook_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j, col] = 1.0
                best_V[j * p + k, col] = 1.0
                best_W[k * n + i, col] = 1.0
                col += 1
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        for i in range(n):
            for k in range(p):
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # In tensor form: T[i,k,j] = delta[i,i]*delta[k,k]*delta[j,j]... 
                # Actually the tensor M[i,k,j] = 1 if we consider C[i,k] = sum_j A[i,j]B[j,k]
                # The decomposition says: M[i,k,j] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                # Wait, let me re-read the convention.
                # U shape (n*m, rank): U[i*m+j, r] corresponds to A[i,j]
                # V shape (m*p, rank): V[j*p+k, r] corresponds to B[j,k]  
                # W shape (p*n, rank): W[k*n+i, r] corresponds to C[i,k]
                # The tensor equation is: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r] = W[k*n+i,r]... no
                # Actually: the matrix multiplication tensor T has T[i,j,k] = 1 for the (i,j) entry of A times (j,k) entry of B giving (i,k) entry of C.
                # The rank-r decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                # With the flattening: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                # So we need: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                # Wait, that's not right either. Let me think again.
                # Standard tensor decomposition of matrix multiplication:
                # C = A * B means C[i,k] = sum_j A[i,j] * B[j,k]
                # The multiplication tensor M has M[i,k,j] = 1 (it's a 3-tensor where M[i,k,j] gives the coefficient)
                # Actually M is defined such that <M, A⊗B⊗C^T> = 0 for all valid A,B,C
                # More directly: the tensor T[i,j,k] = δ_{i,i}δ_{j,j}δ_{k,k}... no.
                # Let me use the standard formulation:
                # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
                # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor in the right indices)
                # No wait. T[i,j,k] represents: A[i,j] * B[j,k] contributes to C[i,k]
                # So the tensor is T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
                # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
                # With U indexed as U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
                
                expected = 1.0
                actual = 0.0
                for r in range(r):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
        return True
    
    # Try to find lower rank decompositions using random search
    # Strategy: try various ranks from a reasonable lower bound up to schoolbook
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(20, 30, 24) = 30
    # Also Strassen-like bounds might give better estimates
    
    # Let's try a structured approach: use block decomposition ideas
    # For 4x5x6, we can try to exploit the rectangular structure
    
    # First, let's try some structured low-rank decompositions
    # Idea: partition the matrices and use block multiplication
    
    # Try rank values from a lower estimate upward
    # A reasonable starting point: try ranks that are divisors or near divisors
    candidate_ranks = sorted(set([
        30, 32, 36, 40, 42, 45, 48, 50, 54, 56, 60, 64, 66, 70, 72, 75, 80, 84, 90, 96, 100, 105, 108, 112, 114, 116, 118, 120
    ]))
    
    random.seed(42)
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            break
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50 if rank < 60 else 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Try a few gradient-free optimization steps
            # Use coordinate descent: fix two, optimize the third
            for step in range(100):
                if time.time() > deadline:
                    break
                
                # Optimize W given U, V
                for idx in range(p * n):
                    # W[idx, r] should satisfy: for all i,j such that k*n+i == idx
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    # Let k = idx // n, i = idx % n
                    k = idx // n
                    i = idx % n
                    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # This is a system: for each j, sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
                    # Let A_j[r] = U[i*m+j, r] * V[j*p+k, r]
                    # Then sum_r A_j[r] * W[k*n+i, r] = 1 for all j
                    # This is an underdetermined system (m equations, rank unknowns)
                    # We can solve it in least squares sense
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    # Solve A * W_col = b in least squares
                    try:
                        W_col, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[idx, :] = W_col
                    except:
                        pass
                
                if time.time() > deadline:
                    break
                
                # Optimize U given V, W
                for idx in range(n * m):
                    i = idx // m
                    j = idx % m
                    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                    # Let B_k[r] = V[j*p+k, r] * W[k*n+i, r]
                    # Then sum_r U[i*m+j, r] * B_k[r] = 1 for all k
                    B = np.zeros((p, rank))
                    for k in range(p):
                        B[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_col, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                        U[idx, :] = U_col
                    except:
                        pass
                
                if time.time() > deadline:
                    break
                
                # Optimize V given U, W
                for idx in range(m * p):
                    j = idx // p
                    k = idx % p
                    # Need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # Let C_i[r] = U[i*m+j, r] * W[k*n+i, r]
                    # Then sum_r V[j*p+k, r] * C_i[r] = 1 for all i
                    C = np.zeros((n, rank))
                    for i in range(n):
                        C[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_col, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                        V[idx, :] = V_col
                    except:
                        pass
            
            # Check if we found a valid decomposition
            if verify(U, V, W):
                best_rank = rank
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                found = True
                break
        
        if found:
            # Found a better rank, continue searching for even lower
            continue
    
    # Return the best solution found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
