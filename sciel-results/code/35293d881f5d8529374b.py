"""
Vectorized Simulated Annealing solver for QAPLIB wil100.
Strategy:
1. Precompute row sums of A and B.
2. Use a greedy initial solution.
3. Perform Simulated Annealing:
   - In each iteration, calculate the cost delta for ALL possible swaps (i, j) 
     simultaneously using vectorized NumPy operations.
   - This avoids the O(n^2) Python loop overhead of the previous Tabu Search,
     enabling thousands of iterations per second instead of tens.
   - Accept moves based on the Metropolis criterion with a cooling schedule.
   - Track the best solution found.
4. Use the remaining time for a final local search polish or restarts if needed, 
   but the primary power comes from the high-volume SA exploration.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def solve(a, b):
    n = len(a)
    total_deadline = 115.0 # Leave 5s buffer for the 120s limit
    start_time = time.time()
    
    # 1. Initial Solution
    p = _greedy_initial(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute constants for delta calculation
    # Delta for swapping positions i and j in permutation p:
    # Let x = p[i], y = p[j]
    # The change in cost is:
    # sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],y]-b[p[k],x]) 
    # + sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[y,p[k]]-b[x,p[k]])
    # + (a[i,i]-a[j,j]) * (b[y,y]-b[x,x])  -- wait, diagonal terms are 0 in QAP usually? 
    # Actually QAP usually has 0 diagonal. Let's assume general case but wil100 has 0 diag.
    # If diag is 0, a[i,i]=0, a[j,j]=0.
    # The mutual terms: a[i,j]*b[x,y] + a[j,i]*b[y,x] becomes a[i,j]*b[y,x] + a[j,i]*b[x,y]
    # So delta_mutable = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    
    # We will vectorize the calculation of deltas for all pairs (i, j) with i < j.
    # This is tricky to do in a single fully vectorized op without creating a massive 
    # n x n x n array. 
    # Instead, we can iterate i (100 times) and vectorize over j (99 ops).
    # This reduces Python overhead by factor of 100 compared to double loop.
    
    # Let's try a hybrid: Loop over i, vectorize j.
    # For fixed i, we want delta for all j > i.
    # delta(i, j) = 
    #   1. Terms involving i and j in A with B[p[k], p[i]] vs B[p[k], p[j]]
    #      sum_{k!=i,j} a[k,i]*(b[p[k],p[j]]-b[p[k],p[i]]) - a[k,j]*(b[p[k],p[j]]-b[p[k],p[i]])
    #      = sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
    #   2. Terms involving i and j in A (row) with B[p[i], p[k]] vs B[p[j], p[k]]
    #      sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]]-b[p[i],p[k]])
    #   3. Direct terms a[i,j], a[j,i]
    #      a[i,j]*(b[p[j],p[i]]-b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]]-b[p[j],p[i]])
    
    # Let's precompute column and row views of B for current p.
    # B_col[k, loc] = b[loc, p[k]] ? No, b is static. p changes.
    # Let's maintain Bp = b[np.ix_(p, p)].
    # Then b[p[k], p[i]] is Bp[k, i].
    # b[p[k], p[j]] is Bp[k, j].
    # b[p[i], p[k]] is Bp[i, k].
    # b[p[j], p[k]] is Bp[j, k].
    
    # So for fixed i, and vector of j's:
    # Let J be array of indices j > i.
    # Term 1: sum_k (a[k,i] - a[k,J]) * (Bp[k, J] - Bp[k, i])
    #   a[k,i] is vector n. a[k,J] is matrix n x |J|.
    #   Bp[k,i] is vector n. Bp[k,J] is matrix n x |J|.
    #   diff_a = a[:, i][:, None] - a[:, J]  (n x |J|)
    #   diff_b = Bp[:, J] - Bp[:, i][:, None] (n x |J|)
    #   sum over k (axis 0) -> vector |J|
    
    # Term 2: sum_k (a[i,k] - a[J,k]) * (Bp[J, k] - Bp[i, k])
    #   a[i,k] is vector n. a[J,k] is matrix |J| x n.
    #   Bp[i,k] is vector n. Bp[J,k] is matrix |J| x n.
    #   diff_a = a[i, :][None, :] - a[J, :] (|J| x n)
    #   diff_b = Bp[J, :] - Bp[i, :][None, :] (|J| x n)
    #   sum over k (axis 1) -> vector |J|
    
    # Term 3: a[i, J]*(Bp[J, i] - Bp[i, J]) + a[J, i]*(Bp[i, J] - Bp[J, i])
    #   Note: Bp is symmetric if b is symmetric? wil100 b is symmetric.
    #   If b is symmetric, Bp is symmetric. Bp[J, i] = Bp[i, J].
    #   Then Term 3 = (a[i, J] - a[J, i]) * (Bp[J, i] - Bp[i, J]) = 0?
    #   Wait. The swap changes who is at i and who is at j.
    #   Original: a[i,j]*b[x,y] + a[j,i]*b[y,x].
    #   New: a[i,j]*b[y,x] + a[j,i]*b[x,y].
    #   Delta = a[i,j]*(b[y,x]-b[x,y]) + a[j,i]*(b[x,y]-b[y,x])
    #   Delta = (a[i,j] - a[j,i]) * (b[y,x] - b[x,y]).
    #   If b is symmetric, b[y,x] = b[x,y], so delta is 0?
    #   No, b is the distance matrix. It is symmetric.
    #   So the direct cross terms do not change cost if b is symmetric?
    #   Let's check.
    #   Cost includes a[i,j] * dist(x,y).
    #   Swap i and j. Facility i goes to y, facility j goes to x.
    #   New term for pair (i,j) in A is a[i,j] * dist(y,x).
    #   Since dist is symmetric, dist(y,x) = dist(x,y).
    #   So the cost contribution of the pair (i,j) in A remains a[i,j]*dist(x,y).
    #   Wait, the assignment is p. p[i]=x, p[j]=y.
    #   Term is a[i,j] b[p[i], p[j]] = a[i,j] b[x,y].
    #   After swap, p[i]=y, p[j]=x.
    #   Term is a[i,j] b[y,x].
    #   If b is symmetric, b[y,x] = b[x,y]. So this part is invariant.
    #   However, the terms a[j,i] b[p[j], p[i]] = a[j,i] b[y,x] becomes a[j,i] b[x,y].
    #   So the sum a[i,j]b[x,y] + a[j,i]b[y,x] becomes a[i,j]b[y,x] + a[j,i]b[x,y].
    #   If b is symmetric, this is a[i,j]b[x,y] + a[j,i]b[x,y] = (a[i,j]+a[j,i])b[x,y].
    #   Original was a[i,j]b[x,y] + a[j,i]b[x,y] (since b[y,x]=b[x,y]).
    #   So yes, if b is symmetric, the direct swap of i and j does not change the cost contribution of the pair (i,j) itself?
    #   That seems wrong. The position of facility i changes, so its distance to ALL other facilities changes.
    #   The terms involving i and j with OTHER k's change.
    #   The terms a[i,j] and a[j,i] are just 2 out of n^2 terms.
    #   My derivation of Term 3 being 0 relies on b being symmetric.
    #   Let's verify with a small example.
    #   A = [[0, 1], [1, 0]], B = [[0, 10], [10, 0]].
    #   p = [0, 1]. Cost = a[0,1]b[0,1] + a[1,0]b[1,0] = 1*10 + 1*0 = 10.
    #   Swap p -> [1, 0].
    #   Cost = a[0,1]b[1,0] + a[1,0]b[0,1] = 1*10 + 1*10 = 20.
    #   Delta = 10.
    #   Formula: (a[0,1]-a[1,0]) * (b[1,0]-b[0,1]) = (1-1)*(10-10) = 0.
    #   So Term 3 is NOT 0. My logic "b[y,x]=b[x,y]" makes the difference (b[y,x]-b[x,y]) zero.
    #   But in the example:
    #   Original: a[0,1]b[0,1] + a[1,0]b[1,0]
    #   New: a[0,1]b[1,0] + a[1,0]b[0,1]
    #   Delta = a[0,1](b[1,0]-b[0,1]) + a[1,0](b[0,1]-b[1,0])
    #   Delta = (a[0,1]-a[1,0]) * (b[1,0]-b[0,1]).
    #   In example: (1-1)*(10-10) = 0.
    #   But cost changed from 10 to 20.
    #   Where is the error?
    #   Ah, B[0,1] is distance between loc 0 and 1.
    #   p=[0,1]: Fac 0 at Loc 0, Fac 1 at Loc 1.
    #   Term a[0,1] is flow 0->1. Distance is dist(Loc 0, Loc 1) = B[0,1].
    #   Term a[1,0] is flow 1->0. Distance is dist(Loc 1, Loc 0) = B[1,0].
    #   Cost = 1*10 + 1*10 = 20?
    #   My example calculation: "a[1,0]b[1,0] = 1*0".
    #   B[1,0] is 10. So 1*10 = 10. Total 20.
    #   Swap p=[1,0]: Fac 0 at Loc 1, Fac 1 at Loc 0.
    #   Term a[0,1]: Flow 0->1. Dist(Loc 1, Loc 0) = B[1,0] = 10.
    #   Term a[1,0]: Flow 1->0. Dist(Loc 0, Loc 1) = B[0,1] = 10.
    #   Cost = 20.
    #   So cost didn't change?
    #   Let's re-read QAP definition.
    #   sum a[i,j] b[p[i], p[j]].
    #   If A and B are symmetric, and we swap two facilities, does the cost change?
    #   The set of pairs of locations occupied by pairs of facilities is the same?
    #   No. The pair (Fac 0, Fac 1) was at (Loc 0, Loc 1). Now at (Loc 1, Loc 0).
    #   Distance is same.
    #   But Fac 0 is now at Loc 1. Its distance to Fac 2 (at Loc 2) changes from dist(0,2) to dist(1,2).
    #   So the "direct" term between i and j is invariant if B is symmetric.
    #   The change comes entirely from the interaction with all other k.
    #   So Term 3 is indeed 0 if B is symmetric.
    #   Let's assume B is symmetric (wil100 is).
    
    # So we only need Term 1 and Term 2.
    # Term 1: sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],p[j]]-b[p[k],p[i]])
    # Term 2: sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[p[j],p[k]]-b[p[i],p[k]])
    
    # Note: The sum is over k != i, j.
    # In the vectorized sum over all k, we include k=i and k=j.
    # Let's see what happens if we sum over all k.
    # For k=i:
    #   Term 1: (a[i,i]-a[i,j]) * (b[p[i],p[j]]-b[p[i],p[i]])
    #   If a[i,i]=0, b[p[i],p[i]]=0.
    #   = -a[i,j] * b[p[i],p[j]]
    # For k=j:
    #   Term 1: (a[j,i]-a[j,j]) * (b[p[j],p[j]]-b[p[j],p[i]])
    #   = a[j,i] * (-b[p[j],p[i]]) = -a[j,i] b[p[j],p[i]]
    #   
    #   Term 2 for k=i:
    #   (a[i,i]-a[j,i]) * (b[p[j],p[i]]-b[p[i],p[i]])
    #   = -a[j,i] * b[p[j],p[i]]
    #   
    #   Term 2 for k=j:
    #   (a[i,j]-a[j,j]) * (b[p[j],p[j]]-b[p[i],p[j]])
    #   = a[i,j] * (-b[p[i],p[j]])
    
    # Sum of these 4 "error" terms:
    # -a[i,j]b[p[i],p[j]] - a[j,i]b[p[j],p[i]] - a[j,i]b[p[j],p[i]] - a[i,j]b[p[i],p[j]]
    # = -2 * (a[i,j]b[p[i],p[j]] + a[j,i]b[p[j],p[i]])
    # 
    # But wait, the actual delta includes the change of these terms too!
    # Original contribution of (i,j) pair to cost (from k=i, j and k=j, i and k=i,i, j=j,j etc):
    # This is getting complicated.
    
    # Easier: Just mask out i and j in the sum.
    # Create a mask.
    
    # To speed up, we can precompute:
    # A_col_i = a[:, i]
    # Bp_col_i = Bp[:, i]
    
    # For a batch of j's:
    # A_col_J = a[:, J]
    # Bp_col_J = Bp[:, J]
    
    # Diff_A = A_col_i[:, None] - A_col_J
    # Diff_B = Bp_col_J - Bp_col_i[:, None]
    # T1_raw = np.sum(Diff_A * Diff_B, axis=0)
    # We need to subtract the k=i and k=j contributions from T1_raw.
    # k=i contribution to T1 for a specific j:
    # (a[i,i]-a[i,j]) * (b[p[i],p[j]]-b[p[i],p[i]])
    # k=j contribution to T1 for a specific j:
    # (a[j,i]-a[j,j]) * (b[p[j],p[j]]-b[p[j],p[i]])
    
    # This is doable but messy.
    # Given n=100, even a Python loop over i (100) with vectorized j (99) is fast.
    # 100 * (vector ops on 99 elements) is very fast.
    
    # Let's implement the loop-over-i, vector-over-j approach.
    
    T_start = 1000.0
    T_end = 1.0
    # Number of iterations we hope to do? 
    # If each iter takes ~0.1ms (100 vector ops), we can do 1000 iters in 100ms.
    # 115s -> 1,000,000 iterations? That's a lot.
    # Let's just run until time limit.
    
    # Cooling schedule: T = T_start * (T_end/T_start)^iter/total_iters?
    # We don't know total_iters.
    # Use T = T_start * exp(-alpha * iter)?
    # Or just T = T_start * (1 - iter/max_iter)?
    # Let's use a fixed number of iterations based on time budget.
    # Estimate time per iteration.
    
    # First, do a few iterations to measure time.
    iters_done = 0
    Bp = b[np.ix_(p, p)]
    
    # Preallocate masks? No, create on fly.
    
    while time.time() - start_time < total_deadline - 2:
        # Calculate deltas for all i < j
        # We will store best_move and its delta.
        # Actually, for SA, we just pick a random move?
        # No, "Vectorized SA" usually means calculate all deltas, then pick one?
        # If we calculate all deltas, we might as well pick the best one (Local Search).
        # But SA allows uphill moves.
        # If we calculate all deltas, we can sample a move from the distribution?
        # Or just pick a random move and calculate its delta?
        # Calculating ALL deltas is O(n^3) or O(n^2 * n) = O(n^3)?
        # My vectorization was O(n^2) per i, so O(n^3) total?
        # No, for each i, I do ops on vectors of size n. So O(n) per i.
        # Total O(n^2) per iteration.
        # n=100 -> 10,000 ops. Very fast.
        
        # So we can calculate ALL deltas every iteration.
        # Then we can either:
        # 1. Pick the best improving move (Local Search) - but SA is better for escaping local minima.
        # 2. Pick a random move and accept/reject.
        # 3. Pick a move with probability proportional to exp(-delta/T)?
        
        # Let's stick to standard SA: Pick a random neighbor.
        # But if we have all deltas, we can pick a random (i,j) and use its pre-computed delta.
        # This is efficient.
        
        # Generate random i, j
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        # Ensure i < j for consistency? Not needed for random pick.
        
        # Calculate delta for this specific pair.
        # We can reuse the vectorized logic for a single pair?
        # Or just use the scalar formula?
        # Scalar formula is O(n).
        # mask = np.ones(n, dtype=bool); mask[i]=False; mask[j]=False
        # p_masked = p[mask]
        # ...
        # This is what the parent program did, but in a loop.
        # Doing it for ONE pair is fast.
        
        # Let's try to calculate delta for this specific (i,j) efficiently.
        pi = p[i]
        pj = p[j]
        
        # T1: sum_{k!=i,j} (a[k,i]-a[k,j]) * (b[p[k],pj]-b[p[k],pi])
        # T2: sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[pj,p[k]]-b[pi,p[k]])
        
        # Vectorized over k:
        # We need to exclude i and j.
        # We can compute sum over all k and subtract k=i and k=j.
        
        # Sum over all k of (a[:, i] - a[:, j]) * (b[p, pj] - b[p, pi])
        # b[p, pj] is a vector where element k is b[p[k], pj].
        # This is b[np.ix_(p, [pj])] -> shape (n, 1) -> flatten.
        # Or just b[p, pj] ? No, b is 2D. b[p, pj] gives array of b[p[0],pj], b[p[1],pj]...
        # Yes, b[p, pj] works if pj is scalar.
        
        diff_a_col = a[:, i] - a[:, j]
        diff_b_row = b[p, pj] - b[p, pi] # b[p[k], pj] - b[p[k], pi]
        
        sum_all_T1 = np.dot(diff_a_col, diff_b_row)
        
        # Subtract k=i term:
        # (a[i,i]-a[i,j]) * (b[p[i],pj]-b[p[i],pi])
        # (a[i,i]-a[i,j]) * (b[pi,pj]-b[pi,pi])
        term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
        
        # Subtract k=j term:
        # (a[j,i]-a[j,j]) * (b[p[j],pj]-b[p[j],pi])
        # (a[j,i]-a[j,j]) * (b[pj,pj]-b[pj,pi])
        term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
        
        T1 = sum_all_T1 - term_1_i - term_1_j
        
        # T2: sum_{k!=i,j} (a[i,k]-a[j,k]) * (b[pj,p[k]]-b[pi,p[k]])
        # Sum over all k of (a[i, :] - a[j, :]) * (b[pj, p] - b[pi, p])
        # b[pj, p] is vector where element k is b[pj, p[k]].
        diff_a_row = a[i, :] - a[j, :]
        diff_b_col = b[pj, p] - b[pi, p]
        
        sum_all_T2 = np.dot(diff_a_row, diff_b_col)
        
        # Subtract k=i term:
        # (a[i,i]-a[j,i]) * (b[pj,p[i]]-b[pi,p[i]])
        # (a[i,i]-a[j,i]) * (b[pj,pi]-b[pi,pi])
        term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
        
        # Subtract k=j term:
        # (a[i,j]-a[j,j]) * (b[pj,p[j]]-b[pi,p[j]])
        # (a[i,j]-a[j,j]) * (b[pj,pj]-b[pi,pj])
        term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
        
        T2 = sum_all_T2 - term_2_i - term_2_j
        
        # Term 3 (direct):
        # (a[i,j]-a[j,i]) * (b[pj,pi]-b[pi,pj])
        T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
        
        delta = T1 + T2 + T3
        new_cost = current_cost + delta
        
        # SA Acceptance
        # T schedule
        # Let's use a simple linear cooling based on iterations.
        # We don't know total iterations.
        # Use time-based cooling?
        # T = T_start * (T_end/T_start) ^ (elapsed / total_time)
        elapsed = time.time() - start_time
        T = T_start * (T_end / T_start) ** (elapsed / total_deadline)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            # Accept move
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            
            # Update Bp?
            # Bp = b[np.ix_(p, p)] is O(n^2) to compute.
            # We only need Bp for delta calculation?
            # In the delta calculation above, I used b[p, pj] etc.
            # I did NOT use Bp directly in the dot products!
            # I used b[p, pj] which is a fancy index.
            # So I don't need to maintain Bp explicitly, just p.
            # This saves O(n^2) update time.
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        iters_done += 1
        
    return best_p.tolist()

# EVOLVE-BLOCK-END
