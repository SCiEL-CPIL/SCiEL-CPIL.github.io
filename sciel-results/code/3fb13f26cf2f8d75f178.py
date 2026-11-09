import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin within 120s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # Actually, the tensor is defined such that:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # In tensor form: T is a 3-tensor where T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # for some i,j,k, else 0
    
    # We want: T = sum_{r=1}^{rank} u_r ⊗ v_r ⊗ w_r
    # where u_r is vector of length n*m, v_r of length m*p, w_r of length p*n
    
    # For exact decomposition with integer/half-integer entries,
    # let's try to find a decomposition with rank < schoolbook_rank
    
    # Strategy: Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    
    # For this specific small case, let's try a structured approach:
    # The matrix multiplication tensor M_{2,5,5} might have a known low-rank decomposition
    
    # Let's first verify the schoolbook works, then try to improve
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try to find a decomposition with rank r
    # We'll use a simple approach: try random matrices and check
    
    # For exact arithmetic, use Fraction
    # But that's slow. Let's try with rational numbers represented as fractions
    
    def verify_decomposition(U, V, W, T_target, tol=1e-10):
        """Verify that sum_r U[:,r] V[:,r] W[:,r] == T_target"""
        n1, r = U.shape
        n2, _ = V.shape
        n3, _ = W.shape
        
        # Compute the reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check exact match (with tolerance for floating point)
        if np.allclose(T_recon, T_target, atol=tol, rtol=0):
            return True
        return False
    
    # Try to find a lower rank decomposition
    # For rank r, we need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that the tensor equation holds
    
    # This is a system of (n*m)*(m*p)*(p*n) equations with 3*r*(n*m + m*p + p*n) unknowns
    # For n=2, m=5, p=5: 10*25*10 = 2500 equations
    # Unknowns: 3*r*(10+25+10) = 135*r
    # For r=25: 3375 unknowns > 2500 equations (overdetermined but might have solution)
    
    # Let's try a randomized approach with exact verification
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # First, store schoolbook as fallback
    U_school = np.zeros((n*m, schoolbook_rank), dtype=int)
    V_school = np.zeros((m*p, schoolbook_rank), dtype=int)
    W_school = np.zeros((p*n, schoolbook_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank decompositions
    # We'll use a simple iterative approach
    
    # For each candidate rank, try random initializations
    for target_rank in range(25, schoolbook_rank):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with small integers
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize U, V, W with random values from {-1, 0, 1}
            U = np.random.randint(-1, 2, (n*m, target_rank))
            V = np.random.randint(-1, 2, (m*p, target_rank))
            W = np.random.randint(-1, 2, (p*n, target_rank))
            
            # Check if this gives the correct tensor
            if verify_decomposition(U, V, W, T):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Ensure entries are integers or half-integers
    # If any entry is not a multiple of 0.5, we have a problem
    # But since we used integer initialization, this should be fine
    
    return {"U": U_list, "V": V_list, "W": W_list}
