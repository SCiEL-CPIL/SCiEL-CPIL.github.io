"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Symmetrize for cost computation efficiency (cost is invariant to symmetrization)
    # Actually, we should use the given matrices as-is for correctness
    # But note: sum_{i,j} a[i,j]*b[p[i],p[j]] - if a and b are not symmetric, we use them directly
    
    def total_cost(p):
        # p is array of length n, p[i] = location of facility i
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # Use matrix multiplication for speed
        # b_perm = b[np.ix_(p, p)]
        # cost = np.sum(a * b_perm)
        # More efficient: use einsum or matmul
        # Actually: sum_{i,j} a[i,j]*b[p[i],p[j]] = trace(a * b_perm.T) if we think carefully
        # b[p[i], p[j]] for all i,j is b[np.ix_(p,p)]
        # sum a[i,j]*b[p[i],p[j]] = np.sum(a * b[np.ix_(p,p)])
        # This is O(n^2) which is fine for n=100
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))
    
    # Initial solution: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    best_p = p.copy()
    best_cost = total_cost(p)
    
    # Vectorized 2-opt / swap local search
    def local_search(p, deadline):
        """Perform swap-based local search with vectorized cost delta computation."""
        current_cost = total_cost(p)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # For each pair (i,j), compute delta if we swap p[i] and p[j]
            # This is O(n^2) per pass, but we can vectorize the delta computation
            
            # Cost = sum_{k,l} a[k,l]*b[p[k],p[l]]
            # If we swap positions of facilities i and j (i.e., swap p[i] and p[j]):
            # Only terms involving i or j change
            # Let's denote pi = p[i], pj = p[j]
            # Before: terms with i or j:
            #   sum_l a[i,l]*b[pi,p[l]] + sum_l a[l,i]*b[p[l],pi] + 
            #   sum_l a[j,l]*b[pj,p[l]] + sum_l a[l,j]*b[p[l],pj] - a[i,i]*b[pi,pi] - a[j,j]*b[pj,pj]
            #   (the -a[i,i]*b[pi,pi] and -a[j,j]*b[pj,pj] because i=j term was counted twice in the sums)
            # Actually, let's be more careful. The full cost is:
            # C = sum_{k,l} a[k,l]*b[p[k],p[l]]
            # Terms involving i: sum_l a[i,l]*b[pi,p[l]] + sum_{l != i} a[l,i]*b[p[l],pi]
            # But it's easier to think: C = sum_{k != i, l != j} ... + terms with i or j
            # 
            # Let me use a different approach. Define:
            # row_i = a[i, :]  (flow from i to all)
            # col_i = a[:, i]  (flow from all to i)
            # Similarly for j.
            # 
            # When we swap p[i] <-> p[j]:
            # New cost = C - (contribution of i and j with old positions) + (contribution of i and j with new positions)
            # 
            # Contribution of facility i at position pi:
            #   sum_l a[i,l]*b[pi,p[l]] + sum_{l != i} a[l,i]*b[p[l],pi]
            #   = row_i @ b[pi, :] + col_i @ b[:, pi] - a[i,i]*b[pi,pi]
            #   (subtract a[i,i]*b[pi,pi] because it's counted in both terms)
            # 
            # This is getting complex. Let me use a simpler vectorized approach.
            
            # For n=100, we can afford O(n^2) per iteration with some vectorization
            # Let's precompute row and column sums contributions
            
            # Alternative: just do random swaps with acceptance
            # But let's try systematic 2-opt with vectorized delta
            
            # Precompute: for each facility i, the "row contribution" and "col contribution"
            # Actually, let's just iterate through pairs and compute delta efficiently
            
            # For swapping p[i] and p[j] where pi = p[i], pj = p[j]:
            # Delta = -sum_{k != i,j} a[i,k]*b[pi,p[k]] - sum_{k != i,j} a[k,i]*b[p[k],pi]
            #         -sum_{k != i,j} a[j,k]*b[pj,p[k]] - sum_{k != i,j} a[k,j]*b[p[k],pj]
            #         +sum_{k != i,j} a[i,k]*b[pj,p[k]] + sum_{k != i,j} a[k,i]*b[p[k],pj]
            #         +sum_{k != i,j} a[j,k]*b[pi,p[k]] + sum_{k != i,j} a[k,j]*b[p[k],pi]
            #         - a[i,i]*b[pi,pi] - a[j,j]*b[pj,pj] + a[i,i]*b[pj,pj] + a[j,j]*b[pi,pi]
            #         - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi] + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
            
            # This is complex. Let me use a different strategy: 
            # Do random restarts with simple local search
            
            # For now, let's do a simple hill climbing with random swaps
            for _ in range(500):  # limited iterations per pass
                if time.time() >= deadline:
                    return p, current_cost
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                pi, pj = p[i], p[j]
                # Compute new cost by swapping
                p[i], p[j] = pj, pi
                new_cost = total_cost(p)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    p[i], p[j] = pi, pj
                if time.time() >= deadline:
                    return p, current_cost
            # If no improvement in this pass, try a few more random perturbations
            if not improved:
                # Try a few random swaps to escape local optimum
                for _ in range(10):
                    if time.time() >= deadline:
                        return p, current_cost
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i == j:
                        continue
                    p[i], p[j] = p[j], p[i]
                    current_cost = total_cost(p)
        return p, current_cost
    
    # Main loop: multiple restarts
    while time.time() < deadline:
        p, cost = local_search(p, deadline)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # If we've converged, do a random perturbation
        if time.time() >= deadline:
            break
        
        # Random perturbation: swap k random pairs
        p = best_p.copy()
        k = random.randint(3, 10)
        for _ in range(k):
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            p[i], p[j] = p[j], p[i]
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
