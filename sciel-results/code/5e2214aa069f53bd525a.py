"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Rank-23 decomposition for 3x3 matrix multiplication
    # Convention: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (tensor factor)
    # U has shape (n*m, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> j*p+k
    # W has shape (p*n, rank), indexed by (k,i) -> k*n+i (trace convention)
    #
    # The tensor identity is:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k

    def verify(U, V, W, rank):
        """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if val != 1:
                        return False
        return True

    def make_rank27():
        """Schoolbook rank-27 decomposition (always valid)."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Strategy: Find a rank-23 decomposition by identifying 4 linear dependencies
    # among the 27 schoolbook terms and merging them.
    #
    # The tensor identity is: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    #
    # We'll use a local search approach:
    # 1. Start with a structured initial guess based on Strassen-like patterns
    # 2. Use gradient-free optimization (random perturbations + verification)
    # 3. Fall back to rank-27 if rank-23 is not found in time
    
    import random
    random.seed(12345)
    
    search_deadline = time.time() + 115.0  # safety margin
    
    # First, try known structured constructions for rank-23
    # Based on the fact that 3x3 = 2x2 + 1x1 extension
    
    def try_structured_rank23():
        """Try structured constructions for rank-23."""
        # Construction: Use Strassen's 7-term formula for the 2x2 sub-block
        # and add 16 more terms for the remaining parts, then find 10 more merges.
        # Actually, let's use a more direct approach.
        
        # The 27 schoolbook terms are indexed by (i,j,k) for i,j,k in {0,1,2}
        # We need to find 4 linear relations to reduce to 23 terms.
        
        # A known approach: consider the tensor as a 3x3x3 array of 1s.
        # The rank is the minimum number of rank-1 tensors that sum to it.
        
        # Let's try a specific construction based on the literature.
        # One valid rank-23 decomposition:
        
        # Define 23 terms with specific support patterns
        # This is based on the work of Bini et al. and subsequent refinements.
        
        # I'll encode a candidate decomposition and verify it.
        
        # Term definitions: each term is a triple (support_U, support_V, support_W)
        # where support_X is a dict mapping index -> coefficient
        
        # Let me use a systematic encoding approach.
        # The 23 terms will be constructed to satisfy the tensor identity.
        
        # For now, return None to indicate this needs more work.
        return None
    
    # Try the structured approach first
    result = try_structured_rank23()
    if result is not None:
        U, V, W = result
        if verify(U, V, W, 23):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Use a local search starting from near-sparse random matrices
    # with a bias toward finding solutions
    
    best_U, best_V, best_W = None, None, None
    
    # Use a simulated annealing-like approach
    # Objective: minimize the number of violated tensor equations
    
    def count_violations(U, V, W, rank):
        """Count how many (i,j,k) tuples violate the identity."""
        violations = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if val != 1:
                        violations += 1
        return violations
    
    def local_search():
        """Local search for rank-23 decomposition."""
        rank = 23
        best_violations = 27
        best_state = None
        
        # Initialize with sparse random matrices
        U = np.zeros((9, rank), dtype=int)
        V = np.zeros((9, rank), dtype=int)
        W = np.zeros((9, rank), dtype=int)
        
        for r in range(rank):
            for idx in range(9):
                if random.random() < 0.3:
                    U[idx, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    V[idx, r] = random.choice([-1, 1])
                if random.random() < 0.3:
                    W[idx, r] = random.choice([-1, 1])
        
        current_violations = count_violations(U, V, W, rank)
        
        if current_violations == 0:
            return U.copy(), V.copy(), W.copy()
        
        best_state = (U.copy(), V.copy(), W.copy(), current_violations)
        
        # Local search iterations
        max_iters = 100000
        for it in range(max_iters):
            if time.time() > search_deadline:
                break
            
            # Random perturbation: flip a random entry in one matrix
            mat = random.choice([0, 1, 2])  # 0=U, 1=V, 2=W
            row = random.randint(0, 8)
            col = random.randint(0, rank - 1)
            
            if mat == 0:
                old_val = U[row, col]
                U[row, col] = random.choice([-1, 0, 1])
            elif mat == 1:
                old_val = V[row, col]
                V[row, col] = random.choice([-1, 0, 1])
            else:
                old_val = W[row, col]
                W[row, col] = random.choice([-1, 0, 1])
            
            new_violations = count_violations(U, V, W, rank)
            
            if new_violations < current_violations:
                current_violations = new_violations
                best_state = (U.copy(), V.copy(), W.copy(), new_violations)
                if new_violations == 0:
                    return U.copy(), V.copy(), W.copy()
            else:
                # Revert
                if mat == 0:
                    U[row, col] = old_val
                elif mat == 1:
                    V[row, col] = old_val
                else:
                    W[row, col] = old_val
        
        if best_state[3] == 0:
            return best_state[0], best_state[1], best_state[2]
        return None
    
    # Try local search
    result = local_search()
    if result is not None:
        U, V, W = result
        if verify(U, V, W, 23):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank-27
    U, V, W = make_rank27()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
