"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x6 matrix.
    Starts with the schoolbook decomposition (rank 90) and attempts to reduce rank via
    structured random search with exact verification.
    """
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Deadline guard: leave 5s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Build the multiplication tensor T[i, j, k] = 1 for all i,j,k
    # The tensor is T such that (A * B)[i, k] = sum_j A[i, j] * B[j, k]
    # Tensor form: T_{i,j,k} where output index is (i,k) with convention W[k*n+i]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition requires: for all i, j, k:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Schoolbook solution
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try to find lower-rank decompositions using structured random search
    # Strategy: try ranks from schoolbook_rank-1 down, with limited attempts per rank
    # Use integer/half-integer entries only
    
    def try_random_decomposition(rank, max_iters=5000):
        """Try to find a valid decomposition of given rank using random search."""
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Initialize with small random half-integers
            U = [[random.choice([-0.5, 0, 0.5, 1, -1]) for _ in range(rank)] for _ in range(nm)]
            V = [[random.choice([-0.5, 0, 0.5, 1, -1]) for _ in range(rank)] for _ in range(mp)]
            W = [[random.choice([-0.5, 0, 0.5, 1, -1]) for _ in range(rank)] for _ in range(pn)]
            
            if verify(U, V, W, rank):
                return {"U": U, "V": V, "W": W}
        return None
    
    # Try progressively lower ranks
    # For 3x5x6, known lower bounds suggest rank might be reducible below 90
    # but exact minimal rank is hard to find. Let's try a few lower ranks.
    
    # First, try a structured approach: exploit symmetry and known factorizations
    # For small matrices, try rank = n*m + m*p + p*n - something (Strassen-like)
    
    # Try a few candidate ranks
    candidate_ranks = []
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 50), -1):
        candidate_ranks.append(r)
    
    # Also try some specific ranks that might work
    for r in [89, 88, 87, 86, 85, 84, 83, 82, 81, 80]:
        if r not in candidate_ranks and r < schoolbook_rank:
            candidate_ranks.append(r)
    
    # Limit total search time
    search_start = time.time()
    max_search_time = 130.0
    
    for rank in sorted(candidate_ranks, reverse=True):
        if time.time() > deadline or (time.time() - search_start) > max_search_time:
            break
        
        # Quick check: can we find a valid decomposition?
        result = try_random_decomposition(rank, max_iters=2000)
        if result is not None:
            best = result
            # Found a lower rank, try to go even lower
            # But be conservative: only try a few more ranks
            if time.time() > deadline - 10:
                break
    
    # Final verification
    if not verify(best["U"], best["V"], best["W"], len(best["U"][0])):
        # Fallback to schoolbook
        best = schoolbook()
    
    return best
# EVOLVE-BLOCK-END
