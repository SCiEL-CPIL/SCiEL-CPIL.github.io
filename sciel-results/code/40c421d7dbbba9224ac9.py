import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # 5s safety margin
    
    # Initial solution: Rank matching
    # Heavy flows (high row sums in A) should go to central locations (low row sums in B)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort indices
    # We want p[heavy_facility] = central_location
    # heavy = argsort(-a_sums) -> indices of facilities sorted by descending sum
    # central = argsort(b_sums) -> indices of locations sorted by ascending sum
    heavy = np.argsort(-a_sums, kind='stable')
    central = np.argsort(b_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Calculate initial cost
    # cost = sum(a[i,j] * b[p[i], p[j]])
    # Using matrix multiplication for speed: cost = np.sum(a * b[np.ix_(p, p)])
    # But b[np.ix_(p,p)] is O(n^2) memory. For n=150 it's fine.
    cost = np.sum(a * b[np.ix_(p, p)])
    best_p = p.copy()
    best_cost = cost
    
    def compute_delta(i, j, p, a, b):
        """
        Compute cost difference if we swap p[i] and p[j].
        Returns delta such that new_cost = old_cost + delta.
        """
        pi = p[i]
        pj = p[j]
        
        # Terms involving i and j interacting with others k != i, j
        # delta = sum_{k != i, j} [ a[i,k] * (b[pj, pk] - b[pi, pk]) + a[j,k] * (b[pi, pk] - b[pj, pk]) ]
        #       + a[i,j] * (b[pj, pi] - b[pi, pj]) + a[j,i] * (b[pi, pj] - b[pj, pi])
        
        # Let's vectorize for a fixed i, finding best j.
        # This function is for a single pair, but we will vectorize the search.
        pass

    def local_search(p, cost, a, b, n, time_limit, start_time):
        """
        Perform best-improvement 2-opt local search.
        Returns updated p, cost, and whether an improvement was made.
        """
        improved = True
        while improved:
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            
            # Iterate over i
            for i in range(n):
                pi = p[i]
                # Vectorized calculation of delta for all j > i (to avoid duplicates)
                # Actually, let's just compute for all j != i and take min, then apply.
                # To save time, we can compute for j > i only, but symmetry of a and b?
                # a and b are not necessarily symmetric. So we must check all pairs or handle carefully.
                # Standard 2-opt checks all unordered pairs {i,j}.
                
                # Let's compute delta for all j != i
                # delta_j = sum_{k != i, j} a[i,k](b[pj,pk] - b[pi,pk]) + a[j,k](b[pi,pk] - b[pj,pk]) 
                #          + a[i,j](b[pj,pi] - b[pi,pj]) + a[j,i](b[pi,pj] - b[pj,pi])
                
                # This is hard to fully vectorize for all i,j at once without O(n^3) memory.
                # Looping over i and vectorizing over j is O(n^2) vector ops of size n.
                
                js = np.arange(n)
                js = js[js != i]
                
                pjs = p[js]
                pis = np.full(len(js), pi, dtype=np.int64)
                
                # b[pj, pk] vs b[pi, pk]
                # pk is p[js] for the first term? No, pk is p[k] where k != i, j.
                # This dependency on k (which is j in the sum) makes it tricky.
                
                # Let's stick to the standard formula:
                # delta(i,j) = 
                #   sum_{k != i, j} a[i,k] * b[pj, pk] - sum_{k != i, j} a[i,k] * b[pi, pk]
                # + sum_{k != i, j} a[j,k] * b[pi, pk] - sum_{k != i, j} a[j,k] * b[pj, pk]
                # + a[i,j] * b[pj, pi] + a[j,i] * b[pi, pj]
                # - a[i,j] * b[pi, pj] - a[j,i] * b[pj, pi]
                
                # Let S1 = sum_{k != i, j} a[i,k] * b[pj, pk]
                # Let S2 = sum_{k != i, j} a[i,k] * b[pi, pk]
                # Let S3 = sum_{k != i, j} a[j,k] * b[pi, pk]
                # Let S4 = sum_{k != i, j} a[j,k] * b[pj, pk]
                
                # Note that pk depends on k.
                # For a fixed i and candidate j, we can compute S1..S4 by summing over k.
                # This is still O(n) per pair (i,j).
                
                # Let's optimize the inner loop.
                # For a fixed i, we want to find best j.
                # We can precompute some terms?
                
                # Alternative: Use the "exchange" formula from literature.
                # delta = (a[i,:]-a[j,:]) @ (b[pi,:]-b[pj,:]) ... no.
                
                # Let's just loop j inside i, but use NumPy for the sum over k.
                # n=150, so 150*150 = 22500 pairs.
                # Each pair requires O(n) sum. 22500 * 150 = 3.3M operations.
                # This is very fast in Python if the inner loop is vectorized.
                
                # We will loop i, then loop j (or vectorize j).
                # Let's vectorize j for a fixed i.
                
                # For fixed i:
                # pi = p[i]
                # We want delta_j for all j != i.
                
                # Term 1: sum_{k != i, j} a[i,k] * b[pj, pk]
                # This depends on pj and pk.
                
                # This is getting complex to vectorize fully for all j at once.
                # Let's just do a simple loop over j, but use NumPy for the sum over k.
                # 22500 iterations, each doing a few NumPy vector ops of size 150.
                # Should take < 1 second per full pass.
                
                for j in range(i + 1, n):
                    if time.time() - start_time > time_limit:
                        return p, cost, False
                    
                    pj = p[j]
                    
                    # Exclude i and j from k
                    mask = np.ones(n, dtype=bool)
                    mask[i] = False
                    mask[j] = False
                    ks = np.where(mask)[0]
                    pks = p[ks]
                    
                    # S1 = sum a[i,ks] * b[pj, pks]
                    # S2 = sum a[i,ks] * b[pi, pks]
                    # S3 = sum a[j,ks] * b[pi, pks]
                    # S4 = sum a[j,ks] * b[pj, pks]
                    
                    # Vectorized sums
                    s1 = np.dot(a[i, ks], b[pj, pks])
                    s2 = np.dot(a[i, ks], b[pi, pks])
                    s3 = np.dot(a[j, ks], b[pi, pks])
                    s4 = np.dot(a[j, ks], b[pj, pks])
                    
                    # Direct terms
                    d1 = a[i, j] * b[pj, pi]
                    d2 = a[j, i] * b[pi, pj]
                    d3 = a[i, j] * b[pi, pj]
                    d4 = a[j, i] * b[pj, pi]
                    
                    delta = (s1 - s2 + s3 - s4) + (d1 + d2 - d3 - d4)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_delta < 0:
                # Apply swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += best_delta
                improved = True
                # Update best if better than global best
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                # No improvement in this pass
                break
        
        return p, cost, improved

    # Run local search from initial solution
    p, cost, _ = local_search(p, cost, a, b, n, time_limit, start_time)
    
    # If time remains, try a few random restarts/perturbations
    # We will do up to 5 restarts if time allows
    for restart in range(5):
        if time.time() - start_time > time_limit - 10: # Keep 10s buffer
            break
        
        # Perturb best solution
        p_curr = best_p.copy()
        # Swap 2-5 random pairs
        num_swaps = np.random.randint(2, 6)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, 2, replace=False)
            p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
        
        # Calculate cost for perturbed solution
        # cost_curr = np.sum(a * b[np.ix_(p_curr, p_curr)])
        # This is O(n^2). Fast enough.
        cost_curr = np.sum(a * b[np.ix_(p_curr, p_curr)])
        
        # Local search from perturbed solution
        p_curr, cost_curr, _ = local_search(p_curr, cost_curr, a, b, n, time_limit, start_time)
        
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr.copy()
            
    return best_p.tolist()
