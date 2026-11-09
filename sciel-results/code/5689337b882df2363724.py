import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = 150.0 - 15.0  # 15-second safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T as a 3D array
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid rank-r decomposition."""
        r = U.shape[1]
        T_rec = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        return np.allclose(T_rec, T, atol=tol)
    
    def try_systematic_search(r, max_attempts=5000):
        """
        Try to find a rank-r decomposition using systematic search over 
        structured integer/rational factor matrices.
        
        Key insight: For matrix multiplication tensors, optimal decompositions
        often use entries from {0, 1, -1} or {0, ±1, ±1/2}.
        """
        if time.time() > deadline:
            return None
        
        # The tensor has 63 non-zero entries. Each rank-1 term can cover
        # multiple entries if the factors align properly.
        
        # Strategy: Use a structured approach where we try to find
        # U, V, W with sparse patterns that collectively cover all tensor entries.
        
        # For each candidate rank r, try different initialization strategies
        
        best_error = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + attempt * 7)
            
            # Initialize with structured random patterns
            # Use a mixture of dense and sparse patterns
            
            if attempt % 3 == 0:
                # Dense initialization
                U = np.random.randint(0, 3, (nm, r)) - 1  # {-1, 0, 1}
                V = np.random.randint(0, 3, (mp, r)) - 1
                W = np.random.randint(0, 3, (pn, r)) - 1
            elif attempt % 3 == 1:
                # Sparse initialization - each column has few non-zeros
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for r_idx in range(r):
                    # Choose 1-3 positions for U
                    u_pos = np.random.choice(nm, size=np.random.randint(1, 4), replace=False)
                    for pos in u_pos:
                        U[pos, r_idx] = np.random.choice([-1, 1])
                    
                    # Choose 1-3 positions for V
                    v_pos = np.random.choice(mp, size=np.random.randint(1, 4), replace=False)
                    for pos in v_pos:
                        V[pos, r_idx] = np.random.choice([-1, 1])
                    
                    # Choose 1-3 positions for W
                    w_pos = np.random.choice(pn, size=np.random.randint(1, 4), replace=False)
                    for pos in w_pos:
                        W[pos, r_idx] = np.random.choice([-1, 1])
            else:
                # Block-structured initialization
                # Group indices into blocks and assign patterns
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                
                # Create block structure based on the tensor's natural grouping
                # i in 0..2, j in 0..2, k in 0..6
                
                for r_idx in range(r):
                    # Assign block patterns
                    i_block = r_idx % n
                    j_block = (r_idx // n) % m
                    k_blocks = range((r_idx // (n*m)) % p, p, max(1, p // (r // (n*m))))
                    
                    U[i_block * m + j_block, r_idx] = 1
                    
                    for k in k_blocks:
                        if k < p:
                            V[j_block * p + k, r_idx] = 1
                            W[k * n + i_block, r_idx] = 1
            
            # Compute reconstruction error
            T_rec = np.zeros((nm, mp, pn))
            for r_idx in range(r):
                T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            error = np.max(np.abs(T_rec - T))
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # If we found an exact decomposition, return immediately
            if error < 1e-8:
                return (U, V, W)
            
            # If error is small, try local refinement
            if error < 0.1 and attempt % 10 == 0:
                # Try simple local search: flip some entries
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random matrix, row, column to modify
                    mat_choice = np.random.randint(0, 3)
                    if mat_choice == 0:
                        row, col = np.random.randint(0, nm), np.random.randint(0, r)
                        old_val = U[row, col]
                        U[row, col] = (old_val + 1) % 3 - 1 if old_val in [-1, 0, 1] else 1
                    elif mat_choice == 1:
                        row, col = np.random.randint(0, mp), np.random.randint(0, r)
                        old_val = V[row, col]
                        V[row, col] = (old_val + 1) % 3 - 1 if old_val in [-1, 0, 1] else 1
                    else:
                        row, col = np.random.randint(0, pn), np.random.randint(0, r)
                        old_val = W[row, col]
                        W[row, col] = (old_val + 1) % 3 - 1 if old_val in [-1, 0, 1] else 1
                    
                    # Recompute error
                    T_rec = np.zeros((nm, mp, pn))
                    for r_idx in range(r):
                        T_rec += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
                    new_error = np.max(np.abs(T_rec - T))
                    
                    if new_error < error:
                        error = new_error
                        if error < 1e-8:
                            return (U, V, W)
                    else:
                        # Revert
                        if mat_choice == 0:
                            U[row, col] = old_val
                        elif mat_choice == 1:
                            V[row, col] = old_val
                        else:
                            W[row, col] = old_val
        
        # Check if best solution is valid
        if best_U is not None and check_decomposition(best_U, best_V, best_W):
            return (best_U, best_V, best_W)
        
        return None
    
    def try_hierarchical_decomposition(r):
        """
        Try a hierarchical approach: first find a good decomposition for a
        sub-problem, then extend it.
        
        Key insight: 3x3 x 3x7 can be viewed as three 3x3 x 3x1 problems
        concatenated, or we can use the structure more cleverly.
        """
        if time.time() > deadline:
            return None
        
        # Strategy: Use the fact that the k-dimension has 7 elements.
        # We can try to find a decomposition that exploits the structure.
        
        # For each i (output row), we need to compute:
        # C[i, k] = sum_j A[i, j] * B[j, k]
        
        # The tensor is: T[i*m+j, j*p+k, k*n+i] = 1
        
        # Let's try to find a decomposition where:
        # U has structure: U[i*m+j, r] depends on (i, j, r)
        # V has structure: V[j*p+k, r] depends on (j, k, r)  
        # W has structure: W[k*n+i, r] depends on (k, i, r)
        
        # Try using a "separable" approach where we first handle the
        # i-j structure, then the j-k structure
        
        # For now, fall back to systematic search
        return try_systematic_search(r, max_attempts=3000)
    
    # Try to find the minimum rank
    best_result = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    # Theoretical lower bounds:
    # - max(nm, mp, pn) = max(9, 21, 21) = 21
    # - For matrix multiplication, rank >= max(n*p, m*p, n*m) in some cases
    # - Known results suggest rank around 40-50 for this size
    
    # Try ranks from a reasonable upper bound down to lower bound
    # Start with a few key ranks and expand if needed
    
    ranks_to_try = [50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try systematic search
        result = try_systematic_search(r, max_attempts=2000)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_result = (U, V, W)
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
