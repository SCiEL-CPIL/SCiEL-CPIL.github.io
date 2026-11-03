"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    # Precompute for fast cost evaluation
    # cost(p) = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can write cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] @ b)[:, p[i]] ... but let's think in terms of incremental updates.
    
    # For a 2-opt swap of p[i] and p[j], the change in cost can be computed efficiently.
    # Let's precompute: for each facility i, and each location l, the "row contribution"
    # Actually, let's define:
    #   C[i, l] = sum_j a[i, j] * b[l, p[j]]  ... but this depends on p.
    # Better: use the standard QAP 2-opt delta formula.
    
    # cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let's define for current permutation p:
    #   For swapping p[i] <-> p[j]:
    #   Only terms involving i or j change.
    #   Old contribution of i,j pair and their interactions:
    #   Let x = p[i], y = p[j]
    #   Terms that change:
    #     a[i,j] b[x,y] + a[j,i] b[y,x]  (the i-j and j-i terms)
    #     sum_{k != i,j} a[i,k] b[x, p[k]] + a[k,i] b[p[k], x]
    #     sum_{k != i,j} a[j,k] b[y, p[k]] + a[k,j] b[p[k], y]
    #   New:
    #     a[i,j] b[y,x] + a[j,i] b[x,y]
    #     sum_{k != i,j} a[i,k] b[y, p[k]] + a[k,i] b[p[k], y]
    #     sum_{k != i,j} a[j,k] b[x, p[k]] + a[k,j] b[p[k], x]
    #   
    # This is O(n) per swap evaluation. With n=150, that's 150 ops per swap.
    # Number of swaps: n*(n-1)/2 = ~11175. Total per iteration: ~1.7M ops.
    # In 120s, we can do maybe 1000-5000 iterations depending on overhead.
    
    # Let's use a smarter incremental approach.
    # Precompute:
    #   row_sum_a[i] = sum_j a[i,j]  (not directly useful)
    #   Instead, maintain:
    #   For each location l, and each facility i:
    #     S[i, l] = sum_{k: p[k] != l} a[i,k] * b[l, p[k]] + sum_{k: p[k] != l} a[k,i] * b[p[k], l]
    #   This is the "external" contribution to facility i if it were at location l, 
    #   excluding the interaction with the facility currently at l.
    
    # Actually, let's just do the O(n) delta computation per candidate swap.
    # For n=150, this should be fast enough in NumPy.
    
    # Vectorized Simulated Annealing
    
    # Precompute matrices as float64 for speed
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    
    # Initial solution: Rank matching (Greedy heuristic)
    # Sort facilities by total flow (heaviest first)
    # Sort locations by total distance (most central first)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(p):
        return float(np.sum(af * bf[np.ix_(p, p)]))
        
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    time_limit = 115  # Leave margin for overhead
    
    # Simulated Annealing parameters
    T = 100000.0  # Initial temperature (tuned based on typical cost magnitudes)
    T_min = 1.0
    cooling_rate = 0.995
    
    # Precompute indices for vectorization
    # We want to evaluate all pairs (i, j) with i < j simultaneously
    # This is a bit tricky with standard broadcasting for n=150, 
    # but we can do it in chunks or use a specific structure.
    # Actually, for n=150, n^2 = 22500. 
    # We can create a matrix of deltas for all i, j.
    
    # To compute delta for all swaps (i, j) efficiently:
    # delta(i, j) = 
    #   (af[i,j]*bf[p[j],p[i]] + af[j,i]*bf[p[i],p[j]] - af[i,j]*bf[p[i],p[j]] - af[j,i]*bf[p[j],p[i]])
    #   + sum_{k != i, j} [
    #       af[i,k]*(bf[p[j],p[k]] - bf[p[i],p[k]]) 
    #       + af[k,i]*(bf[p[k],p[j]] - bf[p[k],p[i]])
    #       + af[j,k]*(bf[p[i],p[k]] - bf[p[j],p[k]])
    #       + af[k,j]*(bf[p[k],p[i]] - bf[p[k],p[j]])
    #     ]
    
    # Let's define:
    # R[i, l] = sum_{k != i} af[i,k] * bf[l, p[k]] + sum_{k != i} af[k,i] * bf[p[k], l]
    # This R[i, l] represents the cost contribution of facility i at location l, 
    # excluding the interaction with itself (which is 0 in QAP usually, or handled separately)
    # and excluding the interaction with the facility currently at l? 
    # No, R[i, l] should be the cost of placing i at l, assuming all other facilities are fixed.
    # Cost(i at l) = sum_{k != i} af[i,k] bf[l, p[k]] + sum_{k != i} af[k,i] bf[p[k], l]
    
    # If we swap i and j (x=p[i], y=p[j]):
    # New Cost(i at y) = sum_{k != i, j} af[i,k] bf[y, p[k]] + af[i,j] bf[y, x] + sum_{k != i, j} af[k,i] bf[p[k], y] + af[j,i] bf[x, y]
    # Old Cost(i at x) = sum_{k != i, j} af[i,k] bf[x, p[k]] + af[i,j] bf[x, y] + sum_{k != i, j} af[k,i] bf[p[k], x] + af[j,i] bf[y, x]
    
    # Let S[i, l] = sum_{k != i} af[i,k] bf[l, p[k]] + sum_{k != i} af[k,i] bf[p[k], l]
    # Then Old Cost(i at x) = S[i, x] - (af[i,j] bf[x, y] + af[j,i] bf[y, x])  (subtract j's contribution which is included in S[i,x] if k=j)
    # Wait, S[i, x] includes k=j. 
    # S[i, x] = sum_{k != i} ... 
    # Term for k=j in S[i, x] is af[i,j] bf[x, p[j]] + af[j,i] bf[p[j], x] = af[i,j] bf[x, y] + af[j,i] bf[y, x].
    # So, Cost(i at x, others fixed) = S[i, x] - (af[i,j] bf[x, y] + af[j,i] bf[y, x]) + (af[i,j] bf[x, y] + af[j,i] bf[y, x]) ? No.
    # The total cost is sum_i S[i, p[i]] / 2 ? No, S[i, p[i]] counts each pair twice?
    # Let's check:
    # Sum_i S[i, p[i]] = Sum_i Sum_{k != i} (af[i,k] bf[p[i], p[k]] + af[k,i] bf[p[k], p[i]])
    # = Sum_{i != k} af[i,k] bf[p[i], p[k]] + Sum_{i != k} af[k,i] bf[p[k], p[i]]
    # Let i' = k, k' = i in second sum: Sum_{k' != i'} af[i',k'] bf[p[i'], p[k']]
    # So Sum_i S[i, p[i]] = 2 * Total Cost.
    # So Total Cost = 0.5 * Sum_i S[i, p[i]].
    
    # If we swap i and j:
    # Only S[i, p[i]] and S[j, p[j]] change? No, S[k, p[k]] for k != i, j also changes?
    # S[k, p[k]] depends on p. If p[i] and p[j] swap, then for any k, the term involving p[i] or p[j] changes.
    # So all S[k, p[k]] change. This approach is O(n^2) to update S.
    
    # Alternative: Direct Delta Calculation Vectorized.
    # For a specific swap (i, j):
    # Delta = NewCost - OldCost
    # = (Cost(i at y) + Cost(j at x)) - (Cost(i at x) + Cost(j at y))
    #   + (Change in interactions between i, j and rest)
    # Actually, the "Rest" interactions are accounted for if we define:
    # C[i, l] = sum_{k != i} af[i,k] bf[l, p[k]] + sum_{k != i} af[k,i] bf[p[k], l]
    # This is the "external" cost of placing i at l.
    # If we place i at x and j at y:
    # Total Cost = 0.5 * (C[i, x] + C[j, y] + Sum_{k != i, j} C[k, p[k]]) + af[i,j]bf[x,y] + af[j,i]bf[y,x]
    # Wait, C[i, x] includes interaction with j (since k=j is in sum).
    # C[i, x] = ... + af[i,j]bf[x,y] + af[j,i]bf[y,x] + ...
    # So if we sum C[i, p[i]] for all i, we double count everything.
    # Total Cost = 0.5 * Sum_i C[i, p[i]].
    
    # If we swap i and j:
    # New Total Cost = 0.5 * (C'[i, y] + C'[j, x] + Sum_{k != i, j} C'[k, p[k]])
    # Where C' is calculated with the new permutation p' (where i and j are swapped).
    # For k != i, j:
    # C'[k, p[k]] = sum_{m != k} af[k,m] bf[p[k], p'[m]] + sum_{m != k} af[m,k] bf[p'[m], p[k]]
    # p'[m] is p[m] for m != i, j.
    # p'[i] = y, p'[j] = x.
    # So for k != i, j:
    # C'[k, p[k]] = sum_{m != k, i, j} af[k,m] bf[p[k], p[m]] + af[k,i]bf[p[k],y] + af[k,j]bf[p[k],x]
    #               + sum_{m != k, i, j} af[m,k] bf[p[m], p[k]] + af[i,k]bf[y,p[k]] + af[j,k]bf[x,p[k]]
    # Old C[k, p[k]] = sum_{m != k} af[k,m] bf[p[k], p[m]] + sum_{m != k} af[m,k] bf[p[m], p[k]]
    # = sum_{m != k, i, j} ... + af[k,i]bf[p[k],x] + af[k,j]bf[p[k],y] + af[i,k]bf[x,p[k]] + af[j,k]bf[y,p[k]]
    
    # Difference for k != i, j:
    # Delta_k = C'[k, p[k]] - C[k, p[k]]
    # = af[k,i](bf[p[k],y] - bf[p[k],x]) + af[k,j](bf[p[k],x] - bf[p[k],y])
    #   + af[i,k](bf[y,p[k]] - bf[x,p[k]]) + af[j,k](bf[x,p[k]] - bf[y,p[k]])
    
    # For i:
    # C'[i, y] = sum_{m != i} af[i,m] bf[y, p'[m]] + sum_{m != i} af[m,i] bf[p'[m], y]
    # = af[i,j]bf[y,x] + sum_{m != i, j} af[i,m]bf[y,p[m]] + af[j,i]bf[x,y] + sum_{m != i, j} af[m,i]bf[p[m],y]
    # Old C[i, x] = af[i,j]bf[x,y] + sum_{m != i, j} af[i,m]bf[x,p[m]] + af[j,i]bf[y,x] + sum_{m != i, j} af[m,i]bf[p[m],x]
    
    # Delta_i = C'[i, y] - C[i, x]
    # = af[i,j](bf[y,x] - bf[x,y]) + af[j,i](bf[x,y] - bf[y,x])
    #   + sum_{m != i, j} af[i,m](bf[y,p[m]] - bf[x,p[m]])
    #   + sum_{m != i, j} af[m,i](bf[p[m],y] - bf[p[m],x])
    
    # Similarly for j:
    # Delta_j = C'[j, x] - C[j, y]
    # = af[j,i](bf[x,y] - bf[y,x]) + af[i,j](bf[y,x] - bf[x,y])
    #   + sum_{m != i, j} af[j,m](bf[x,p[m]] - bf[y,p[m]])
    #   + sum_{m != i, j} af[m,j](bf[p[m],x] - bf[p[m],y])
    
    # Total Delta = 0.5 * (Delta_i + Delta_j + Sum_{k != i, j} Delta_k)
    
    # This is still O(n) per swap. To vectorize, we need to compute this for all (i, j) pairs at once.
    # This is complex. 
    # Given n=150, O(n) per swap with NumPy is fast.
    # Let's stick to a simple loop but optimize the inner product.
    
    # However, we can precompute the "potential" for each facility i to move to each location l.
    # Let P[i, l] = sum_{k != i} af[i,k] bf[l, p[k]] + sum_{k != i} af[k,i] bf[p[k], l]
    # This is C[i, l].
    # We can maintain C as a matrix of shape (n, n).
    # When we swap p[i] and p[j] (x and y):
    # 1. Update C for all k.
    #    For k != i, j:
    #    C[k, l] changes only if l is x or y? No, C[k, l] depends on p.
    #    If p changes at i and j, then for any k, C[k, l] changes for all l?
    #    No, C[k, l] = sum_{m != k} af[k,m] bf[l, p[m]] + ...
    #    If m=i, term is af[k,i] bf[l, p[i]]. If p[i] changes from x to y, this term changes for all l.
    #    So C[k, l] changes for all k, l. This is O(n^2) update.
    #    With n=150, n^2 = 22500. This is very fast in NumPy.
    
    # Strategy:
    # 1. Initialize C (n x n matrix) where C[i, l] is the cost of placing i at l, given current p.
    # 2. In each step, pick a random pair (i, j) or iterate through all?
    #    If we iterate through all, we need to update C after each swap?
    #    No, we can evaluate all swaps based on current C.
    #    But if we apply one swap, C becomes invalid.
    #    So we must update C after every swap.
    #    Cost of update: O(n^2).
    #    Cost of evaluation: We can find the best swap in O(n^2) by looking at C?
    #    Actually, if we have C, the delta for swapping i, j is:
    #    Delta = 0.5 * ( (C[i, y] - C[i, x]) + (C[j, x] - C[j, y]) + Sum_{k != i, j} (C'[k, p[k]] - C[k, p[k]]) )
    #    Wait, C[i, y] calculated with current p includes interaction with j at y?
    #    C[i, y] = sum_{m != i} af[i,m] bf[y, p[m]] + ...
    #    If m=j, term is af[i,j] bf[y, y]? No, p[j]=y. So bf[y, y].
    #    This is getting confusing.
    
    # Let's use the explicit Delta formula with NumPy vectorization over k for each (i, j).
    # But doing it in a Python loop for 11k pairs is slow.
    # 
    # Alternative: Randomized Local Search / Simulated Annealing with Random Moves.
    # Pick random i, j. Compute Delta. Accept if Delta < 0 or with prob exp(-Delta/T).
    # This allows O(1) or O(n) per step.
    # With O(n) per step, we can do ~100,000 steps in 115s.
    # This is much better than 1,000 steps of full neighborhood search.
    
    T = 50000.0
    T_min = 1.0
    cooling_rate = 0.999
    
    # Precompute row/col sums of a? No, just use af.
    
    while time.time() - start_time < time_limit:
        # Pick random distinct i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        x = p[i]
        y = p[j]
        
        # Compute Delta
        # Delta = 0.5 * (Delta_i + Delta_j + Sum_{k != i, j} Delta_k)
        
        # Vectorized computation over k
        # We need sums over k != i, j
        
        # Terms for i moving x -> y:
        # af[i, k] * (bf[y, p[k]] - bf[x, p[k]]) + af[k, i] * (bf[p[k], y] - bf[p[k], x])
        # Terms for j moving y -> x:
        # af[j, k] * (bf[x, p[k]] - bf[y, p[k]]) + af[k, j] * (bf[p[k], x] - bf[p[k], y])
        
        # Sum over all k, then subtract k=i and k=j terms?
        # Or just mask.
        
        # Create mask for k != i, j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        pk = p[mask]
        
        # Vectorized sums
        # Note: bf[y, pk] is vector of length n-2
        
        term1 = af[i, mask] * (bf[y, pk] - bf[x, pk])
        term2 = af[mask, i] * (bf[pk, y] - bf[pk, x])
        term3 = af[j, mask] * (bf[x, pk] - bf[y, pk])
        term4 = af[mask, j] * (bf[pk, x] - bf[pk, y])
        
        sum_k = np.sum(term1 + term2 + term3 + term4)
        
        # Diagonal terms (i, j interaction)
        # Old: af[i,j]bf[x,y] + af[j,i]bf[y,x]
        # New: af[i,j]bf[y,x] + af[j,i]bf[x,y]
        diag_old = af[i, j] * bf[x, y] + af[j, i] * bf[y, x]
        diag_new = af[i, j] * bf[y, x] + af[j, i] * bf[x, y]
        
        # Wait, the formula derived earlier:
        # Delta = 0.5 * (Delta_i + Delta_j + Sum_{k != i, j} Delta_k)
        # Delta_i included the diagonal change for i?
        # My previous derivation:
        # Delta_i = C'[i, y] - C[i, x]
        # C[i, x] includes af[i,j]bf[x,y] + af[j,i]bf[y,x].
        # C'[i, y] includes af[i,j]bf[y,x] + af[j,i]bf[x,y].
        # So Delta_i includes the diagonal change.
        # Similarly Delta_j includes the diagonal change (swapped).
        # So Delta_i + Delta_j includes 2 * (diag_new - diag_old).
        # And Sum_{k} Delta_k does not include diagonal.
        # So Total Delta = 0.5 * (2*(diag_new - diag_old) + sum_k)
        # = (diag_new - diag_old) + 0.5 * sum_k
        
        # Let's verify.
        # Total Cost = 0.5 * Sum C[i, p[i]]
        # Delta = 0.5 * (Delta_i + Delta_j + Sum_{k != i, j} Delta_k)
        # Delta_i = (af[i,j]bf[y,x] + af[j,i]bf[x,y] - af[i,j]bf[x,y] - af[j,i]bf[y,x]) + Sum_{k != i, j} (af[i,k](bf[y,p[k]]-bf[x,p[k]]) + af[k,i](bf[p[k],y]-bf[p[k],x]))
        # Let D_diag = af[i,j]bf[y,x] + af[j,i]bf[x,y] - af[i,j]bf[x,y] - af[j,i]bf[y,x]
        # Delta_i = D_diag + Sum_{k != i, j} ...
        # Delta_j = D_diag + Sum_{k != i, j} ... (with j terms)
        # Sum_{k != i, j} Delta_k = Sum_{k != i, j} (af[k,i](bf[p[k],y]-bf[p[k],x]) + af[k,j](bf[p[k],x]-bf[p[k],y]) + af[i,k](bf[y,p[k]]-bf[x,p[k]]) + af[j,k](bf[x,p[k]]-bf[y,p[k]]))
        
        # Note that Sum_{k} in my vectorized code is:
        # sum_k = Sum_{k != i, j} [ af[i,k](bf[y,p[k]]-bf[x,p[k]]) + af[k,i](bf[p[k],y]-bf[p[k],x]) + af[j,k](bf[x,p[k]]-bf[y,p[k]]) + af[k,j](bf[p[k],x]-bf[p[k],y]) ]
        
        # This matches the "Sum_{k != i, j}" parts of Delta_i, Delta_j, and Delta_k combined?
        # Delta_i sum part: af[i,k](...) + af[k,i](...)
        # Delta_j sum part: af[j,k](...) + af[k,j](...)
        # Delta_k sum part: af[k,i](...) + af[k,j](...) + af[i,k](...) + af[j,k](...)
        
        # So Delta_i + Delta_j + Sum Delta_k =
        # 2 * D_diag + 2 * (Sum_{k} [af[i,k](...) + af[k,i](...)] + Sum_{k} [af[j,k](...) + af[k,j](...)] )
        # Wait.
        # Sum_{k} in vectorized code is exactly the sum of the "sum parts" of Delta_i and Delta_j.
        # Let S_i = Sum_{k != i, j} (af[i,k](bf[y,p[k]]-bf[x,p[k]]) + af[k,i](bf[p[k],y]-bf[p[k],x]))
        # Let S_j = Sum_{k != i, j} (af[j,k](bf[x,p[k]]-bf[y,p[k]]) + af[k,j](bf[p[k],x]-bf[p[k],y]))
        # Then sum_k = S_i + S_j.
        
        # Delta_i = D_diag + S_i
        # Delta_j = D_diag + S_j
        # Delta_k (for each k) = (af[k,i](bf[p[k],y]-bf[p[k],x]) + af[i,k](bf[y,p[k]]-bf[x,p[k]]) + af[k,j](bf[p[k],x]-bf[p[k],y]) + af[j,k](bf[x,p[k]]-bf[y,p[k]]))
        # Sum_{k != i, j} Delta_k = Sum_{k} (af[k,i](...) + af[i,k](...)) + Sum_{k} (af[k,j](...) + af[j,k](...))
        # = S_i + S_j
        
        # So Delta = 0.5 * ( (D_diag + S_i) + (D_diag + S_j) + (S_i + S_j) )
        # = 0.5 * ( 2*D_diag + 2*S_i + 2*S_j )
        # = D_diag + S_i + S_j
        # = D_diag + sum_k
        
        delta = (diag_new - diag_old) + sum_k
        
        # Simulated Annealing Acceptance
        if delta < 0:
            accept = True
        else:
            if T > T_min:
                accept = np.random.rand() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling_rate
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
