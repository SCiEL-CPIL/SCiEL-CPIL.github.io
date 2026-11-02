"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Returns {'U', 'V', 'W'} where:
      U has shape (n*m, rank)
      V has shape (m*p, rank)
      W has shape (p*n, rank)
    The decomposition satisfies: for all i,j,k, sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c]
    Actually, the correct condition is: the matrix multiplication tensor T[i,j,k] = delta[i,j,k] (in the appropriate sense)
    is decomposed as T = sum_c U_c \otimes V_c \otimes W_c.
    
    Specifically, for matrix multiplication C = A @ B where A is 3x3 and B is 3x7:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    In the standard decomposition, we need:
    sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] ... no that's not right.
    
    Let me re-read the task. The output factor follows the official trace convention W[k*n+i,r].
    
    The matrix multiplication tensor for (n,m,p) dimensions has entries T[i,j,k] which is 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k].
    
    A rank-r decomposition gives matrices U (n*m x r), V (m*p x r), W (p*n x r) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Wait, that's not quite right either. Let me think again.
    
    The standard tensor decomposition for matrix multiplication:
    T[i,j,k] = sum_c U_{i,j,c} * V_{j,k,c} * W_{k,i,c}
    
    Where U has indices (i,j,c), V has indices (j,k,c), W has indices (k,i,c).
    
    So U is shape (n*m, r) with U[i*m+j, c], V is shape (m*p, r) with V[j*p+k, c], W is shape (p*n, r) with W[k*n+i, c].
    
    And the condition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (since each term A[i,j]*B[j,k] contributes to C[i,k]).
    
    Wait no. T[i,j,k] is 1 if the product A[i,j]*B[j,k] appears in the computation of C[i,k]. Since C[i,k] = sum_j' A[i,j']*B[j',k], the term A[i,j]*B[j,k] appears when j'=j. So T[i,j,k] = 1 for all i,j,k.
    
    So we need: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for all i in [0,n), j in [0,m), k in [0,p).
    
    The schoolbook solution has rank n*m*p with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 where c = i*m*p + j*p + k.
    
    We want to find a lower rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Start with schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]
    
    # Try to find lower rank decompositions using random search
    # Strategy: try ranks from a lower bound up to the schoolbook rank
    # The theoretical lower bound for 3x3x3 is 19 (Strassen), for 3x3x7 it's at least 19
    
    # Let's try random initialization and then use local search to refine
    # We'll use a simple approach: randomly initialize U, V, W with small rank,
    # then check if the decomposition is valid, and if not, try to fix it.
    
    # Actually, let's use a more systematic approach.
    # For small ranks, we can try to solve the system of equations.
    
    # The condition is: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # This is a system of nonlinear equations in the entries of U, V, W.
    
    # Let's try a greedy approach: build up the decomposition one rank at a time.
    
    def try_rank(r):
        """Try to find a rank-r decomposition. Return (U, V, W) or None."""
        if r >= best_rank:
            return None
        
        # Random restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with sparsity
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            
            # Initialize with random non-zero entries
            for c in range(r):
                # Choose random positions for non-zeros
                u_idx = random.randint(0, nm - 1)
                v_idx = random.randint(0, mp - 1)
                w_idx = random.randint(0, pn - 1)
                
                U[u_idx, c] = random.choice([1, -1])
                V[v_idx, c] = random.choice([1, -1])
                W[w_idx, c] = random.choice([1, -1])
            
            # Check if this gives a valid decomposition
            # For each (i,j,k), compute sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # and see if it equals 1
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U, V, W
        
        return None
    
    # Try ranks from 19 up to best_rank - 1
    for r in range(19, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W = U, V, W
            best_rank = r
            break  # Found the minimum rank we can find
    
    # Convert to lists for the return format
    U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_U]
    V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_V]
    W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else (2*x if abs(2*x - round(2*x)) < 1e-9 else x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
