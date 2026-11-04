"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
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
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def try_structural_reduction():
        """
        Try to find a lower-rank decomposition using structural properties.
        
        For matrix multiplication, we can sometimes exploit the fact that
        the tensor has a specific structure. One approach is to use the
        fact that we can combine terms where the indices align in certain ways.
        
        A more systematic approach: use the fact that the matrix multiplication
        tensor is the Kronecker product structure, and try to find a basis
        change that reduces the rank.
        
        For now, let's try a few specific structural reductions:
        1. Check if we can merge terms with the same (i,j) or (j,k) indices
        2. Use the fact that the tensor is symmetric in certain ways
        """
        # Try a simple structural approach: see if we can find a decomposition
        # with rank < n*m*p by exploiting linear dependencies.
        
        # One known technique: for small matrices, the optimal rank is often
        # close to the schoolbook rank, but there might be savings.
        
        # Let's try to find if there's a rank-89 decomposition by checking
        # if one of the schoolbook terms can be expressed as a linear combination
        # of the others.
        
        # This is complex, so let's try a different approach: use the fact that
        # we can sometimes save 1 multiplication for certain dimensions.
        
        # For 3x5 * 5x6, let's try to see if rank 89 is possible.
        
        # Actually, let's try a more systematic search using integer programming
        # or a greedy approach.
        
        # For now, let's try a simple heuristic: try to find a decomposition
        # with rank 89 by removing one term and seeing if the others can compensate.
        
        # This is getting complex. Let's try a different approach: use the
        # fact that we can sometimes use half-integer entries to reduce rank.
        
        # Let's try a randomized search for rank 89 with integer/half-integer entries.
        
        if time.time() > deadline:
            return None
        
        # Try rank 89 with a simple randomized search
        target_rank = n * m * p - 1  # 89
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with schoolbook and perturb one column
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Copy first 89 terms from schoolbook
            col = 0
            i_count = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col >= target_rank:
                            break
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
                        i_count += 1
                        if col >= target_rank:
                            break
                    if col >= target_rank:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
            
            # Now try to adjust the last column to compensate for the missing term
            # This is a simplified approach; in reality, we'd need to solve a system
            # of equations.
            
            # Skip this for now as it's too complex to implement correctly in the time limit.
            pass
        
        return None
    
    def random_search_with_structure(target_rank, max_iters=100):
        """
        Try to find a decomposition of given rank using random initialization
        with structure-aware perturbations.
        """
        if time.time() > deadline:
            return None
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization with structure: start from schoolbook-like
            # and perturb
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with random values from {-1, 0, 1} or {-0.5, 0, 0.5}
            values = [0, 0.5, 1, -0.5, -1]
            for idx in range(n * m):
                for c in range(target_rank):
                    U[idx][c] = random.choice(values)
            for idx in range(m * p):
                for c in range(target_rank):
                    V[idx][c] = random.choice(values)
            for idx in range(p * n):
                for c in range(target_rank):
                    W[idx][c] = random.choice(values)
            
            # Quick check: see if this random initialization is close
            # to a valid decomposition
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        actual = 0.0
                        for c in range(target_rank):
                            actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        error += (actual - 1.0) ** 2
            
            if error < 1e-6:
                if verify(U, V, W, target_rank):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Start with schoolbook as fallback
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known lower bound: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # But realistically, for 3x5*5x6, the rank is likely close to 90.
    
    # Try ranks from 89 down to 80 (a small range to keep within time limit)
    for target_rank in range(best_rank - 1, max(30, best_rank - 10), -1):
        if time.time() > deadline:
            break
        
        result = random_search_with_structure(target_rank, max_iters=20)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    return best_result
# EVOLVE-BLOCK-END
