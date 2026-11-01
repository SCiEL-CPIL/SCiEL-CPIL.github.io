"""Matrix multiplication 2x3x10 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with schoolbook rank 60
    rank_schoolbook = n * m * p
    
    # Try to find lower rank via structured search
    # For 2x3x10, try ranks from 59 down
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        rank = len(U[0])
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    # Get schoolbook as fallback
    U_fb, V_fb, W_fb, rank_fb = schoolbook_decomposition()
    
    # Try to find rank 59 by merging pairs
    # The idea: if two rank-1 terms can be replaced by one, we reduce rank by 1
    # For terms (i,j,k) and (i',j',k'), they can be merged if the resulting
    # product still gives 1 for all (i,j,k)
    
    # Try all pairs of schoolbook terms to see if we can merge them
    # Term (i,j,k) contributes to U[i*m+j], V[j*p+k], W[k*n+i]
    
    # A merge is possible if we can find coefficients such that:
    # For the merged term, the product U_new * V_new * W_new 
    # equals the sum of the two original products
    
    # This is complex. Let's try a simpler approach:
    # For 2x3x10, known results suggest rank might be 59 or lower
    
    # Try randomized local search: start from schoolbook, try to eliminate one term
    # by adjusting others
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_fb
    
    # Try to find rank 59 decomposition
    # Strategy: for each pair of terms, check if we can eliminate one
    # by modifying the other terms slightly
    
    # Actually, let's try a direct approach for small rank reductions
    # For rank r = 59, we need to find U,V,W with 59 columns
    
    # Use the fact that for 2x3x10, we can potentially save 1 rank
    # by exploiting the 2-row structure
    
    # Try: can we express one term as a combination of others?
    # The tensor is 2x3x10 with all entries 1
    # We need sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
    
    # For i=0 and i=1, we have separate sums
    # sum_r U[0,j,r] V[j,k,r] W[k,0,r] = 1
    # sum_r U[1,j,r] V[j,k,r] W[k,1,r] = 1
    
    # If we can make W[k,0,r] and W[k,1,r] related, we might save rank
    
    # Let's try: set W[k,1,r] = W[k,0,r] for all r, then we need
    # sum_r U[0,j,r] V[j,k,r] W[k,0,r] = 1
    # sum_r U[1,j,r] V[j,k,r] W[k,0,r] = 1
    # This means U[0,j,r] = U[1,j,r] for all j,r, so we can merge i=0 and i=1
    # But then we only need 3*10=30 terms, which is too optimistic
    
    # The issue is that U[0,j,r] and U[1,j,r] need to be different in general
    # to satisfy both equations
    
    # Let's try a more systematic approach with random restarts
    random.seed(42)
    
    # Try rank 59 with random initialization and local search
    target_rank = 59
    
    def try_random_decomposition(rank, max_iters=1000):
        """Try to find a rank-r decomposition via random search."""
        U = np.random.uniform(-2, 2, (nm, rank))
        V = np.random.uniform(-2, 2, (mp, rank))
        W = np.random.uniform(-2, 2, (pn, rank))
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute error
            err = 0
            worst = (0, 0, 0)
            worst_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for r in range(rank):
                            total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        e = abs(total - 1)
                        if e > err:
                            err = e
                            worst = (i, j, k)
                        if e > worst_err:
                            worst_err = e
            
            if err < 1e-8:
                # Convert to list format
                U_list = [[float(U[i, r]) for r in range(rank)] for i in range(nm)]
                V_list = [[float(V[i, r]) for r in range(rank)] for i in range(mp)]
                W_list = [[float(W[i, r]) for r in range(rank)] for i in range(pn)]
                return U_list, V_list, W_list
            
            # Gradient-free update: perturb the worst entry
            i, j, k = worst
            # Perturb U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
            for r in range(rank):
                U[i*m+j, r] += random.gauss(0, 0.1)
                V[j*p+k, r] += random.gauss(0, 0.1)
                W[k*n+i, r] += random.gauss(0, 0.1)
        
        return None
    
    # Try a few random restarts for rank 59
    for attempt in range(5):
        if time.time() > deadline:
            break
        result = try_random_decomposition(target_rank, max_iters=200)
        if result is not None:
            U_list, V_list, W_list = result
            # Verify
            if verify_decomposition(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    if time.time() > deadline:
        return best_result
    
    # Try rank 58
    target_rank = 58
    for attempt in range(3):
        if time.time() > deadline:
            break
        result = try_random_decomposition(target_rank, max_iters=100)
        if result is not None:
            U_list, V_list, W_list = result
            if verify_decomposition(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    if time.time() > deadline:
        return best_result
    
    # Try rank 57
    target_rank = 57
    for attempt in range(2):
        if time.time() > deadline:
            break
        result = try_random_decomposition(target_rank, max_iters=50)
        if result is not None:
            U_list, V_list, W_list = result
            if verify_decomposition(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
    
    return best_result
# EVOLVE-BLOCK-END
