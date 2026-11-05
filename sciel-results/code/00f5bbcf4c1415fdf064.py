"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    """
    Find a low-rank exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    
    # Build target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute: for each output index c, the set of (a,b) pairs that contribute
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Strategy: Use structured random initialization and solve linear systems
    # For a given rank R, fix U and V, then solve for W
    # The constraint is: for all (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    def verify_decomposition(U, V, W, T, tol=1e-10):
        """Verify that U, V, W exactly decompose T."""
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(T_recon, T, atol=tol, rtol=tol)
    
    def try_rank_structured(R, max_attempts=50):
        """Try to find a rank-R decomposition using structured random search."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return False
            
            # Generate random U, V with entries in {-1, 0, 1}
            np.random.seed(42 + attempt * 1000 + R)
            U = np.random.choice([-1, 0, 1], size=(n * m, R))
            V = np.random.choice([-1, 0, 1], size=(m * p, R))
            
            # Check if U and V have sufficient rank
            if np.linalg.matrix_rank(U) < min(n*m, R):
                continue
            if np.linalg.matrix_rank(V) < min(m*p, R):
                continue
            
            # Build the linear system for W
            # For each c, we need: sum_r W[c,r] * (U[:,r] * V[:,r])[a,b] = T[a,b,c]
            # Let K_r = kron(U[:,r], V[:,r]) which is a vector of size (n*m)*(m*p)
            # Then for each c: K @ W[c,:] = T[:,:,c].flatten()
            # where K is a matrix of shape ((n*m)*(m*p), R) with columns K_r
            
            # Build K matrix
            K = np.zeros((n * m * m * p, R))
            for r in range(R):
                K[:, r] = np.kron(U[:, r], V[:, r])
            
            # Solve for W
            W = np.zeros((p * n, R))
            valid = True
            for c in range(p * n):
                b_vec = T[:, :, c].flatten()
                # Solve K @ W[c,:] = b_vec
                try:
                    W_c, residuals, rank_K, sv = np.linalg.lstsq(K, b_vec, rcond=None)
                    W[c, :] = W_c
                    if len(residuals) > 0 and residuals[0] > 1e-8:
                        valid = False
                        break
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the decomposition
            if verify_decomposition(U, V, W, T, tol=1e-9):
                # Try to snap to rational values (integers or halves)
                # Check if entries are close to multiples of 0.5
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                if verify_decomposition(U_snapped, V_snapped, W_snapped, T, tol=1e-9):
                    best_U = U_snapped
                    best_V = V_snapped
                    best_W = W_snapped
                    best_rank = R
                    return True
                else:
                    # Try snapping to integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if verify_decomposition(U_int, V_int, W_int, T, tol=1e-9):
                        best_U = U_int
                        best_V = V_int
                        best_W = W_int
                        best_rank = R
                        return True
                    else:
                        # Accept the floating point solution if it's exact
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = R
                        return True
        
        return False
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = 24
    # Upper bound: schoolbook = 72
    
    # Try ranks from 24 to 71
    found_improvement = False
    for R in range(24, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        # More attempts for lower ranks
        attempts = 100 if R < 40 else 30
        
        if try_rank_structured(R, attempts):
            found_improvement = True
            break
    
    # If no improvement found, try a different strategy: 
    # Use the fact that 3x3x8 can be decomposed as blocks
    # Split the 8 columns into two 4-column blocks
    # Each 3x3x4 block can potentially be decomposed with rank ~18-20
    
    if not found_improvement and time.time() < deadline - 20:
        # Try block decomposition approach
        # Split p=8 into p1=4, p2=4
        p1, p2 = 4, 4
        
        # For each block, try to find a decomposition
        # Block 1: 3x3 * 3x4, output indices k*n+i for k=0..3
        # Block 2: 3x3 * 3x4, output indices k*n+i for k=4..7
        
        # Actually, let's try a different approach: use known results
        # The tensor rank of 3x3x4 matrix multiplication is known to be at most 18
        # (from work on small matrix multiplication)
        
        # For now, let's try a more systematic search with better initialization
        # Use structured random matrices with some symmetry
        
        for R in range(24, 40):
            if time.time() > deadline - 10:
                break
            
            # Try structured initialization: use random but with some structure
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                np.random.seed(12345 + attempt * 7 + R * 13)
                
                # Initialize U with some structure: group by i
                U = np.zeros((n * m, R))
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        U[idx, :] = np.random.choice([-1, 0, 1], size=R)
                
                # Initialize V with some structure: group by j
                V = np.zeros((m * p, R))
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        V[idx, :] = np.random.choice([-1, 0, 1], size=R)
                
                # Build K and solve for W
                K = np.zeros((n * m * m * p, R))
                for r in range(R):
                    K[:, r] = np.kron(U[:, r], V[:, r])
                
                W = np.zeros((p * n, R))
                valid = True
                for c in range(p * n):
                    b_vec = T[:, :, c].flatten()
                    try:
                        W_c, residuals, rank_K, sv = np.linalg.lstsq(K, b_vec, rcond=None)
                        W[c, :] = W_c
                        if len(residuals) > 0 and residuals[0] > 1e-8:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if verify_decomposition(U, V, W, T, tol=1e-9):
                    U_snapped = np.round(U * 2) / 2
                    V_snapped = np.round(V * 2) / 2
                    W_snapped = np.round(W * 2) / 2
                    
                    if verify_decomposition(U_snapped, V_snapped, W_snapped, T, tol=1e-9):
                        best_U = U_snapped
                        best_V = V_snapped
                        best_W = W_snapped
                        best_rank = R
                        found_improvement = True
                        break
                    else:
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = R
                        found_improvement = True
                        break
            
            if found_improvement:
                break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
