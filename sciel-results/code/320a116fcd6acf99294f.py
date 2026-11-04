"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Pre-compute for speed
    # We need to be careful with memory and speed for n=150.
    # Cost calculation: sum a[i,j] * b[p[i], p[j]]
    
    def calc_cost(p):
        # Vectorized cost calculation
        # b_perm[i, j] = b[p[i], p[j]]
        # cost = sum a[i, j] * b_perm[i, j]
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))

    def delta_swap(p, i, j):
        """
        Calculate the change in cost if we swap locations p[i] and p[j].
        Let u = p[i], v = p[j].
        New assignment: p'[i]=v, p'[j]=u.
        
        The cost is C = sum_{k,l} a[k,l] b[p[k], p[l]].
        
        The change comes from terms involving i or j.
        Terms unchanged: k,l not in {i,j}.
        Terms changed:
        1. k=i, l=i: a[i,i] * (b[u,u] -> b[v,v])
        2. k=i, l=j: a[i,j] * (b[u,v] -> b[v,u])
        3. k=j, l=i: a[j,i] * (b[v,u] -> b[u,v])
        4. k=j, l=j: a[j,j] * (b[v,v] -> b[u,u])
        5. k=i, l=k (k not i,j): a[i,k] * (b[u, p[k]] -> b[v, p[k]])
        6. k=j, l=k (k not i,j): a[j,k] * (b[v, p[k]] -> b[u, p[k]])
        7. k=k (k not i,j), l=i: a[k,i] * (b[p[k], u] -> b[p[k], v])
        8. k=k (k not i,j), l=j: a[k,j] * (b[p[k], v] -> b[p[k], u])
        
        This is O(n) per swap evaluation if we precompute row/col sums or use matrix ops.
        However, for SA, we need many evaluations.
        
        Efficient Delta Calculation:
        Let B be the matrix b.
        Let P be the permutation.
        Let M = a @ B[P, P] ? No, that's not right.
        
        Let's derive the delta more carefully.
        C = sum_{k,l} a_{kl} b_{p_k p_l}
        
        Delta = C' - C
        
        Let's precompute for a given permutation p:
        R_i = sum_{l != i} a[i, l] * b[p[i], p[l]]  (Row contribution of i to others? No, a[i,l] is flow from i to l)
        Actually, let's look at the structure.
        
        C = sum_{i} sum_{j} a[i,j] b[p[i], p[j]]
        
        If we swap p[i] and p[j] (let u=p[i], v=p[j]):
        The terms involving i or j change.
        
        Let S = sum_{k not in {i,j}} 
        C = (terms with i or j) + S
        
        Terms with i or j:
        T = a[i,i]b[u,u] + a[i,j]b[u,v] + a[j,i]b[v,u] + a[j,j]b[v,v]
           + sum_{k not in {i,j}} ( a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] )
           + sum_{k not in {i,j}} ( a[j,k]b[v,p[k]] + a[k,j]b[p[k],v] )
           
        New T':
        T' = a[i,i]b[v,v] + a[i,j]b[v,u] + a[j,i]b[u,v] + a[j,j]b[u,u]
            + sum_{k not in {i,j}} ( a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] )
            + sum_{k not in {i,j}} ( a[j,k]b[u,p[k]] + a[k,j]b[p[k],u] )
            
        Delta = T' - T
        
        We can precompute for each facility i:
        RowSum_i(v) = sum_{k not in {i,j}} a[i,k] * b[v, p[k]]  (where v is the location assigned to i)
        ColSum_i(u) = sum_{k not in {i,j}} a[k,i] * b[p[k], u]  (where u is the location assigned to i)
        
        But these depend on the current permutation.
        
        Alternative:
        Precompute M = a @ b ? No.
        
        Let's use the matrix form.
        Let X be the assignment matrix (0/1).
        Cost = Tr(A X B X^T) = sum_{i,j,k,l} A_{ik} X_{ij} B_{jl} X_{kl}
        
        For a swap of locations u and v assigned to facilities i and j:
        This is equivalent to swapping rows i and j in X? No, swapping the locations means changing the columns?
        Actually, p[i] is the location of facility i.
        So X[i, p[i]] = 1.
        Swapping p[i] and p[j] means:
        X[i, u] becomes 0, X[i, v] becomes 1.
        X[j, v] becomes 0, X[j, u] becomes 1.
        
        This is a 4-point update in X.
        
        Let's stick to the O(n) delta calculation using precomputed vectors.
        For a fixed permutation p, define:
        f_i(v) = sum_{k} a[i,k] * b[v, p[k]]  (Total flow from i to all k, if i is at v)
        g_i(u) = sum_{k} a[k,i] * b[p[k], u]  (Total flow from all k to i, if i is at u)
        
        Note: f_i(p[i]) includes the term k=i: a[i,i]b[p[i],p[i]].
        
        Current Cost C = sum_i f_i(p[i]) / 2 ? 
        sum_i sum_k a[i,k]b[p[i],p[k]] = sum_i f_i(p[i]).
        Yes.
        
        If we swap p[i]=u and p[j]=v:
        New f_i(v) = sum_k a[i,k] b[v, p'[k]]
        New f_j(u) = sum_k a[j,k] b[u, p'[k]]
        
        The change in f_i:
        f_i(v) - f_i(u)
        = (sum_{k not i,j} a[i,k]b[v,p[k]] + a[i,i]b[v,v] + a[i,j]b[v,u]) 
          - (sum_{k not i,j} a[i,k]b[u,p[k]] + a[i,i]b[u,u] + a[i,j]b[u,v])
        
        Similarly for f_j.
        
        And the other terms in the total cost sum?
        C = sum_k f_k(p[k]).
        Only f_i and f_j change?
        No, f_k for k not i,j also changes because p[i] and p[j] changed, so the term b[p[k], p[i]] and b[p[k], p[j]] in f_k changes.
        
        f_k(p[k]) = sum_l a[k,l] b[p[k], p[l]].
        If k not in {i,j}, p[k] is unchanged. But p[i] and p[j] are swapped.
        So the terms in f_k involving l=i and l=j change.
        
        Delta f_k = a[k,i] (b[p[k], v] - b[p[k], u]) + a[k,j] (b[p[k], u] - b[p[k], v])
                  = (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
        
        So Total Delta = (f_i(v) - f_i(u)) + (f_j(u) - f_j(v)) + sum_{k not i,j} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
        
        This is O(n) to compute if we have the vectors.
        
        To make it fast, we can precompute:
        For each pair (u, v) of locations? No, too many.
        
        Let's implement a helper that computes the delta for a specific swap (i, j) in O(n).
        Since n=150, O(n) is very fast (150 ops).
        """
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        
        # We need:
        # 1. sum_{k not i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
        # 2. sum_{k not i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
        # 3. sum_{k not i,j} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
        # 4. Self terms:
        #    a[i,i](b[v,v]-b[u,u]) + a[i,j](b[v,u]-b[u,v])
        #    a[j,j](b[u,u]-b[v,v]) + a[j,i](b[u,v]-b[v,u])
        
        # Let's compute the sums using numpy vectorization on the subset of k.
        # mask = np.ones(n, dtype=bool); mask[i]=False; mask[j]=False
        # k_indices = np.where(mask)[0]
        
        # This creates overhead. Let's just loop or use full sums and subtract i,j terms.
        
        # Full sums:
        # S_i_v = sum_k a[i,k] b[v, p[k]]
        # S_i_u = sum_k a[i,k] b[u, p[k]]
        
        # b_row_v = b[v, p] -> shape (n,)
        # b_row_u = b[u, p]
        
        b_v = b[v, p]
        b_u = b[u, p]
        
        # S_i_v = dot(a[i, :], b_v)
        # S_i_u = dot(a[i, :], b_u)
        
        S_i_v = np.dot(a[i], b_v)
        S_i_u = np.dot(a[i], b_u)
        
        S_j_v = np.dot(a[j], b_v)
        S_j_u = np.dot(a[j], b_u)
        
        # These include k=i and k=j terms. We need to subtract them or handle them separately.
        # It's easier to compute the full change and subtract the "internal" parts that are double counted or handled separately?
        
        # Let's use the formula:
        # Delta = (S_i_v - S_i_u) + (S_j_u - S_j_v) + sum_{k not i,j} (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
        #        + Correction for the fact that S_i and S_j include the i,j terms which change differently?
        
        # Let's verify.
        # C = sum_k f_k.
        # Delta C = f'_i - f_i + f'_j - f_j + sum_{k not i,j} (f'_k - f_k)
        
        # f_i = S_i_u (with p[i]=u)
        # f'_i = sum_k a[i,k] b[v, p'_k]
        # p'_i = v, p'_j = u, p'_k = p[k] for k!=i,j.
        # f'_i = a[i,i]b[v,v] + a[i,j]b[v,u] + sum_{k not i,j} a[i,k]b[v,p[k]]
        
        # S_i_v = sum_k a[i,k]b[v, p[k]] = a[i,i]b[v,u] + a[i,j]b[v,v] + sum_{k not i,j} a[i,k]b[v,p[k]]
        # Note: p[i]=u, p[j]=v. So b[v, p[i]] = b[v,u], b[v, p[j]] = b[v,v].
        
        # So f'_i = a[i,i]b[v,v] + a[i,j]b[v,u] + (S_i_v - a[i,i]b[v,u] - a[i,j]b[v,v])
        # f'_i = S_i_v + a[i,i](b[v,v]-b[v,u]) + a[i,j](b[v,u]-b[v,v])
        # f'_i = S_i_v + (a[i,i]-a[i,j])(b[v,v]-b[v,u])
        
        # Similarly:
        # f_j = S_j_v (with p[j]=v)
        # f'_j = sum_k a[j,k] b[u, p'_k]
        # f'_j = a[j,j]b[u,u] + a[j,i]b[u,v] + sum_{k not i,j} a[j,k]b[u,p[k]]
        # S_j_u = sum_k a[j,k]b[u, p[k]] = a[j,i]b[u,v] + a[j,j]b[u,u] + sum_{k not i,j} a[j,k]b[u,p[k]]
        # Wait, p[i]=u, p[j]=v.
        # b[u, p[i]] = b[u,u]. b[u, p[j]] = b[u,v].
        # So S_j_u = a[j,i]b[u,v] + a[j,j]b[u,u] + sum...
        # So f'_j = S_j_u.
        
        # Let's re-evaluate f'_i.
        # f'_i uses p'_i=v, p'_j=u.
        # Term k=i: a[i,i] b[v, p'_i] = a[i,i] b[v,v]
        # Term k=j: a[i,j] b[v, p'_j] = a[i,j] b[v,u]
        # Term k!=i,j: a[i,k] b[v, p[k]]
        
        # S_i_v uses p_i=u, p_j=v.
        # Term k=i: a[i,i] b[v, p[i]] = a[i,i] b[v,u]
        # Term k=j: a[i,j] b[v, p[j]] = a[i,j] b[v,v]
        # Term k!=i,j: a[i,k] b[v, p[k]]
        
        # So f'_i - S_i_v = a[i,i](b[v,v]-b[v,u]) + a[i,j](b[v,u]-b[v,v])
        # = (a[i,i]-a[i,j])(b[v,v]-b[v,u])
        
        # Now for f_j.
        # f_j = S_j_v?
        # f_j = sum_k a[j,k] b[v, p[k]] (since p[j]=v)
        # Term k=i: a[j,i] b[v, u]
        # Term k=j: a[j,j] b[v, v]
        # Term k!=i,j: a[j,k] b[v, p[k]]
        # This matches S_j_v.
        
        # f'_j uses p'_j=u, p'_i=v.
        # f'_j = sum_k a[j,k] b[u, p'_k]
        # Term k=i: a[j,i] b[u, p'_i] = a[j,i] b[u, v]
        # Term k=j: a[j,j] b[u, p'_j] = a[j,j] b[u, u]
        # Term k!=i,j: a[j,k] b[u, p[k]]
        
        # S_j_u uses p_i=u, p_j=v.
        # Term k=i: a[j,i] b[u, p[i]] = a[j,i] b[u, u]
        # Term k=j: a[j,j] b[u, p[j]] = a[j,j] b[u, v]
        # Term k!=i,j: a[j,k] b[u, p[k]]
        
        # f'_j - S_j_u = a[j,i](b[u,v]-b[u,u]) + a[j,j](b[u,u]-b[u,v])
        # = (a[j,j]-a[j,i])(b[u,u]-b[u,v])
        
        # Now for k not in {i,j}.
        # f_k = sum_l a[k,l] b[p[k], p[l]]
        # f'_k = sum_l a[k,l] b[p[k], p'_l]
        # Only l=i and l=j change.
        # Delta f_k = a[k,i](b[p[k], v] - b[p[k], u]) + a[k,j](b[p[k], u] - b[p[k], v])
        # = (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
        
        # Sum over k not in {i,j}:
        # Let's compute full sum over all k:
        # Sum_all_k (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
        # For k=i: (a[i,i]-a[i,j]) (b[u, v] - b[u, u])
        # For k=j: (a[j,i]-a[j,j]) (b[v, v] - b[v, u])
        
        # So Sum_{k not i,j} = Sum_all_k - Term_i - Term_j
        
        # Total Delta = (f'_i - f_i) + (f'_j - f_j) + Sum_{k not i,j} (f'_k - f_k)
        # f'_i - f_i = (f'_i - S_i_v) + (S_i_v - S_i_u) + (S_i_u - f_i)
        # Note f_i = S_i_u.
        # So f'_i - f_i = (a[i,i]-a[i,j])(b[v,v]-b[v,u]) + S_i_v - S_i_u
        
        # f'_j - f_j = (f'_j - S_j_u) + (S_j_u - S_j_v) + (S_j_v - f_j)
        # Note f_j = S_j_v.
        # So f'_j - f_j = (a[j,j]-a[j,i])(b[u,u]-b[u,v]) + S_j_u - S_j_v
        
        # Sum_{k not i,j} = Sum_all_k - Term_i - Term_j
        
        # Term_i (from Sum_all_k) = (a[i,i]-a[i,j]) (b[u, v] - b[u, u])
        # Term_j (from Sum_all_k) = (a[j,i]-a[j,j]) (b[v, v] - b[v, u])
        
        # Let's group terms.
        
        b_vv = b[v,v]
        b_uu = b[u,u]
        b_vu = b[v,u]
        b_uv = b[u,v]
        
        # S_i_v - S_i_u = dot(a[i], b_v) - dot(a[i], b_u)
        # S_j_u - S_j_v = dot(a[j], b_u) - dot(a[j], b_v)
        
        # Sum_all_k = dot( (a[:,i] - a[:,j]), (b[:,v][p] - b[:,u][p]) ) ?
        # Let diff_a = a[:,i] - a[:,j]
        # Let diff_b = b[:,v] - b[:,u]
        # We need sum_k diff_a[k] * (b[p[k], v] - b[p[k], u])
        # = sum_k diff_a[k] * diff_b[p[k]]
        
        diff_a = a[:, i] - a[:, j]
        diff_b = b[:, v] - b[:, u]
        
        # Vectorized dot product with permutation
        sum_all_k = np.dot(diff_a, diff_b[p])
        
        term_i = (a[i,i] - a[i,j]) * (b_uv - b_uu)
        term_j = (a[j,i] - a[j,j]) * (b_vv - b_vu)
        
        sum_not_ij = sum_all_k - term_i - term_j
        
        delta_i = (a[i,i] - a[i,j]) * (b_vv - b_vu) + (S_i_v - S_i_u)
        delta_j = (a[j,j] - a[j,i]) * (b_uu - b_uv) + (S_j_u - S_j_v)
        
        return int(delta_i + delta_j + sum_not_ij)

    def or_opt_move(p, i, j, k):
        """
        Move facility i to location p[j], shifting j..k-1 down?
        This is complex to compute delta for.
        Let's stick to Swap for SA, but use a better cooling schedule and multiple restarts.
        Or implement Or-opt delta.
        
        Given time constraints, Swap is O(1) with precomputed vectors?
        My delta_swap above is O(n) due to dot products.
        For n=150, O(n) is ~150 ops. Very fast.
        """
        pass

    def local_search_2opt(p, max_time):
        """
        Greedy 2-opt local search.
        """
        current_cost = calc_cost(p)
        improved = True
        last_check = time.time()
        
        while improved and time.time() < max_time:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # To speed up, we can sample or use a priority queue?
            # For n=150, checking all pairs is 150*149/2 ~ 11k swaps.
            # Each swap is O(n) = 150 ops.
            # Total 1.6M ops per pass.
            # In Python, this might be slow. 1.6M * (numpy overhead)
            # Let's try to vectorize the delta calculation for all pairs?
            # That's hard.
            
            # Alternative: Sample random swaps?
            # No, greedy 2-opt is good for polishing.
            # Let's limit the number of passes or use a timeout.
            
            for i in range(n):
                for j in range(i + 1, n):
                    d = delta_swap(p, i, j)
                    if d < best_delta:
                        best_delta = d
                        best_i, best_j = i, j
                        
            if best_i != -1 and best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                
        return p, current_cost

    def simulated_annealing(p_init, max_time, T_start=5000.0, T_end=10.0, alpha=0.999):
        """
        Simulated Annealing with Swap neighborhood.
        """
        rng = np.random.default_rng(42)
        p = p_init.copy()
        current_cost = calc_cost(p)
        best_p = p.copy()
        best_cost = current_cost
        
        T = T_start
        step = 0
        
        while time.time() < max_time and T > T_end:
            # Pick random swap
            i, j = rng.choice(n, size=2, replace=False)
            i, j = int(i), int(j)
            
            d = delta_swap(p, i, j)
            
            # Acceptance
            if d < 0 or (T > 0 and rng.random() < np.exp(-d / T)):
                p[i], p[j] = p[j], p[i]
                current_cost += d
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
            
            T *= alpha
            step += 1
            
            # Periodic cooling adjustment?
            # The alpha 0.999 means it takes ~4600 steps to go from 5000 to 10.
            # 4600 steps is fast.
            
        return best_p, best_cost

    # Strategy:
    # 1. Generate a good initial solution using a fast heuristic.
    #    - Random start + 2-opt polish.
    # 2. Run Simulated Annealing from this start for a fixed time.
    # 3. Restart with a different random seed/perturbation if time allows.
    
    best_p_global = None
    best_cost_global = np.inf
    
    # Phase 1: Initial Heuristic
    # Try a few random starts, polish with 2-opt, keep the best.
    t_limit_phase1 = start_time + 20.0
    for _ in range(5):
        if time.time() > t_limit_phase1:
            break
        p_rand = np.arange(n)
        np.random.shuffle(p_rand)
        p_polished, c_polished = local_search_2opt(p_rand, t_limit_phase1)
        if c_polished < best_cost_global:
            best_cost_global = c_polished
            best_p_global = p_polished.copy()
            
    if best_p_global is None:
        best_p_global = np.arange(n)
        best_cost_global = calc_cost(best_p_global)

    # Phase 2: Simulated Annealing
    # Use the best solution found so far.
    # Run SA for the remaining time.
    t_limit_phase2 = deadline - 5.0
    
    # We can run multiple SA chains if time permits, or one long chain.
    # One long chain is usually better for escaping local minima if T schedule is good.
    # Let's try one long SA.
    
    p_sa, c_sa = simulated_annealing(best_p_global, t_limit_phase2, T_start=best_cost_global/100.0, T_end=1.0, alpha=0.9995)
    
    if c_sa < best_cost_global:
        best_cost_global = c_sa
        best_p_global = p_sa

    # Phase 3: Final Polish
    if time.time() < deadline - 2.0:
        p_final, c_final = local_search_2opt(best_p_global, deadline - 1.0)
        if c_final < best_cost_global:
            best_cost_global = c_final
            best_p_global = p_final

    return best_p_global.tolist()

# EVOLVE-BLOCK-END
