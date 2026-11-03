"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precomputation for efficient delta calculation
    # We want to minimize sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Let's precompute A_sym = a + a.T to handle the symmetric nature of the cost change terms
    A_sym = a + a.T
    
    # 1. Initialization: Rank matching (Heavy flows to short distances)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_sum, kind="stable")
    # Sort locations by ascending distance sum (central locations)
    central_order = np.argsort(b_sum, kind="stable")
    
    # Initial permutation: p[facility] = location
    p = np.zeros(n, dtype=int)
    for i in range(n):
        p[heavy_order[i]] = central_order[i]
        
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # T_start should be large enough to accept bad moves, T_end small to converge
    # Heuristic: T_start ~ 0.1 * initial_cost / n? Or just a fixed large number.
    # Let's estimate typical delta magnitude.
    # Cost is ~ 8e6. n=150.
    # Let's start T at 1000 and decay exponentially.
    T_start = 1000.0
    T_end = 0.1
    # Number of iterations to estimate decay rate
    # We will run until time runs out, but we need a temperature schedule.
    # Instead of fixed iterations, we can use a time-based cooling or just a long fixed number 
    # if each step is fast enough.
    # With vectorized full scan, each step is O(n^2) vector ops. 
    # n=150, n^2=22500. Vector ops are fast.
    # Let's aim for ~2000-5000 iterations.
    # If we do 3000 iterations, alpha = (T_end/T_start)^(1/3000)
    
    # We'll use a dynamic loop that checks time.
    # To make the schedule work without knowing total iterations, we can use a fixed 
    # number of "planned" iterations and stop early if time runs out, 
    # or simply use a very slow cooling rate and rely on the time limit.
    # Better: Estimate max iterations based on time.
    # Let's assume ~1ms per iteration (optimistic for Python+NumPy overhead on n=150 full scan).
    # 115s / 1ms = 115,000 iterations. That's plenty.
    # Let's be conservative: 10ms per iteration -> 11,500 iterations.
    # Let's set max_iter to a high number and break on time.
    # But we need T to decay.
    # Let's use a fixed number of iterations for the schedule, say 5000, 
    # and if we finish early, we continue with T=T_end (greedy) or restart.
    # Actually, let's just run as many as possible with a cooling schedule 
    # that assumes a target of 5000 iterations. If we finish faster, we can 
    # do a second pass or just let it run.
    
    target_iters = 5000
    alpha = (T_end / T_start) ** (1.0 / target_iters)
    T = T_start
    
    # Precompute row sums of A_sym? No, we need specific interactions.
    # For a fixed i, we want to find j that minimizes delta.
    # Delta(i, j) = 2(a[i,i]-a[j,j])(b[p[j],p[j]]-b[p[i],p[i]]) + 
    #               2(a[i,j]-a[j,i])(b[p[j],p[i]]-b[p[i],p[j]]) +
    #               sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[p[j],p[k]] - b[p[i],p[k]])
    
    # Let u = p[i], v = p[j].
    # Term 3 is the dominant one.
    # sum_{k} (A_sym[i,k] - A_sym[j,k]) (b[v, p[k]] - b[u, p[k]])
    # = sum_k A_sym[i,k](b[v, p[k]] - b[u, p[k]]) - sum_k A_sym[j,k](b[v, p[k]] - b[u, p[k]])
    
    # Let V_k = b[v, p[k]] - b[u, p[k]]
    # This depends on v (which is p[j]) and u (which is p[i]).
    # For a fixed i (and thus u), we can compute the contribution for all j.
    
    # Let's vectorize the inner loop over j for a fixed i.
    # For fixed i, u=p[i]:
    # We want to compute Delta(i, j) for all j != i.
    # v = p[j]
    
    # Term 1: 2 * (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
    # Let C1_i = 2 * a[i,i] * (-b[u,u])  # constant part for i
    # Let C1_j = 2 * a[j,j] * b[v,v]     # variable part for j
    # Term 1 = C1_i + C1_j? No.
    # Term 1 = 2*a[i,i]*b[v,v] - 2*a[i,i]*b[u,u] - 2*a[j,j]*b[v,v] + 2*a[j,j]*b[u,u]
    #         = b[v,v]*(2*a[i,i] - 2*a[j,j]) + b[u,u]*(2*a[j,j] - 2*a[i,i])
    #         = (2*a[i,i] - 2*a[j,j]) * (b[v,v] - b[u,u])
    
    # Term 2: 2 * (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
    # This is hard to vectorize over j because a[i,j] and b[v,u] depend on j.
    # a[i, :] is a row. b[:, u] is a column.
    # b[p, u] is a vector where element j is b[p[j], u] = b[v_j, u].
    # b[u, p] is a vector where element j is b[u, p[j]] = b[u, v_j].
    # So b[v,u] - b[u,v] for all j is: b[p, u] - b[u, p].
    # Let Diff_B_u = b[p, u] - b[u, p]  (vector of length n, index j gives b[p[j], u] - b[u, p[j]])
    # Term 2_j = 2 * (a[i, j] - a[j, i]) * Diff_B_u[j]
    # = 2 * a[i, :] * Diff_B_u - 2 * A_sym[1, j]?? No.
    # a[i, j] - a[j, i] is the skew-symmetric part of A row i.
    # Let A_skew_i = a[i, :] - a[:, i]
    # Term 2 = 2 * A_skew_i * Diff_B_u
    
    # Term 3: sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[v,p[k]] - b[u,p[k]])
    # Let's ignore the k=i,j correction for the main vectorized part and handle it separately,
    # or include it in the full sum and subtract.
    # Full sum S(i,j) = sum_k (A_sym[i,k] - A_sym[j,k]) (b[p[j], p[k]] - b[p[i], p[k]])
    # Let V_j_k = b[p[j], p[k]] - b[p[i], p[k]]
    # This is (b[p[j], :] - b[p[i], :]) @ p? No.
    # b[p[j], p[k]] is element (p[j], p[k]) of b.
    # Let Bp = b[p][:, p]  # Bp[x, y] = b[p[x], p[y]]
    # Then b[p[j], p[k]] = Bp[j, k]
    # b[p[i], p[k]] = Bp[i, k]
    # So V_j_k = Bp[j, k] - Bp[i, k]
    # S(i,j) = sum_k (A_sym[i,k] - A_sym[j,k]) (Bp[j,k] - Bp[i,k])
    # = sum_k A_sym[i,k] Bp[j,k] - sum_k A_sym[i,k] Bp[i,k] - sum_k A_sym[j,k] Bp[j,k] + sum_k A_sym[j,k] Bp[i,k]
    
    # Let's precompute:
    # M1 = A_sym @ Bp  # M1[i, j] = sum_k A_sym[i,k] Bp[j,k]
    # M2 = Bp @ A_sym.T? No.
    # Term 2: sum_k A_sym[i,k] Bp[i,k] = sum_k A_sym[i,k] Bp[i,k] (diagonal of A_sym * Bp? No, elementwise)
    # Let D_Asym_Bp = np.sum(A_sym * Bp, axis=1)  # D_Asym_Bp[i] = sum_k A_sym[i,k] Bp[i,k]
    
    # Term 3: sum_k A_sym[j,k] Bp[j,k] = D_Asym_Bp[j]
    # Term 4: sum_k A_sym[j,k] Bp[i,k] = (A_sym @ Bp.T)[j, i]? 
    # (A_sym @ Bp.T)[j, i] = sum_k A_sym[j, k] Bp.T[k, i] = sum_k A_sym[j, k] Bp[i, k]. Yes.
    # Let M4 = A_sym @ Bp.T  # M4[j, i] = sum_k A_sym[j, k] Bp[i, k]
    
    # So S(i,j) = M1[i, j] - D_Asym_Bp[i] - D_Asym_Bp[j] + M4[j, i]
    
    # Total Delta(i, j) = Term 1 + Term 2 + S(i, j) - Correction(i, j)
    # Correction for k=i, j in S(i,j):
    # The formula S(i,j) summed over ALL k.
    # The true sum is over k != i, j.
    # We need to subtract the terms where k=i and k=j.
    # Term k=i in S(i,j): (A_sym[i,i] - A_sym[j,i]) (Bp[j,i] - Bp[i,i])
    # Term k=j in S(i,j): (A_sym[i,j] - A_sym[j,j]) (Bp[j,j] - Bp[i,j])
    
    # Let's compute the main vectorized part for all j given i.
    # We update Bp whenever p changes.
    # Bp = b[p][:, p]
    
    Bp = b[p][:, p]
    M1 = A_sym @ Bp
    M4 = A_sym @ Bp.T
    D_Asym_Bp = np.sum(A_sym * Bp, axis=1)
    
    # Precompute constants for Term 1 and Term 2 parts that don't depend on j in a simple way?
    # Term 1 depends on j via a[j,j] and b[p[j], p[j]] = Bp[j,j].
    # Term 2 depends on j via A_skew_i and Diff_B_u.
    
    no_improve_count = 0
    max_no_improve = 200  # Restart/cooling adjustment
    
    while time.time() < deadline:
        # Check time
        if time.time() > deadline - 1.0:
            break
            
        # Update Temperature
        if T > T_end:
            T *= alpha
            if T < T_end:
                T = T_end
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # We scan all i? That is O(n^3) per step if we do full vectorization for each i.
        # n=150. n^3 = 3.3M ops. Vectorized, this is very fast (<1ms).
        # Let's do a full neighborhood scan to find the best move (or a good random move).
        # Finding the *best* move is greedy. SA usually picks a *random* move and accepts/rejects.
        # Greedy with SA acceptance is "Best Improvement SA" or just "Greedy" if T=0.
        # Standard SA picks a random neighbor. 
        # However, "Best Improvement" is often faster to converge if we can afford the scan.
        # Let's try: Find the best delta. If delta < 0, accept. 
        # If delta > 0, accept with probability exp(-delta/T).
        # This is "Best-Improvement Simulated Annealing".
        
        # To make it faster, we can just pick a random i and scan j.
        # Or scan all i, j.
        # Let's scan all i, j to find the single best move. This is robust.
        
        # Vectorize Delta calculation for all i, j?
        # Delta(i, j) is a matrix of size n x n.
        
        # Term 1: (2*a[i,i] - 2*a[j,j]) * (Bp[j,j] - Bp[i,i])
        # Let diag_A = 2 * np.diag(a)
        # Let diag_Bp = np.diag(Bp)
        # T1 = (diag_A[:, None] - diag_A[None, :]) * (diag_Bp[None, :] - diag_Bp[:, None])
        # Note: T1[i, j] corresponds to swapping i and j.
        
        # Term 2: 2 * (a[i, j] - a[j, i]) * (b[p[j], p[i]] - b[p[i], p[j]])
        # b[p[j], p[i]] is Bp[j, i]
        # b[p[i], p[j]] is Bp[i, j]
        # So Diff_B = Bp.T - Bp  # Diff_B[i, j] = Bp[j, i] - Bp[i, j]
        # A_skew = a - a.T
        # T2 = 2 * A_skew * Diff_B
        
        # Term 3 (S(i,j)): M1[i, j] - D_Asym_Bp[i] - D_Asym_Bp[j] + M4[j, i]
        # M4[j, i] is element (j, i) of M4. So M4.T[i, j] = M4[j, i].
        # T3 = M1 - D_Asym_Bp[:, None] - D_Asym_Bp[None, :] + M4.T
        
        # Correction:
        # C1 = (A_sym[i,i] - A_sym[j,i]) * (Bp[j,i] - Bp[i,i])
        # C2 = (A_sym[i,j] - A_sym[j,j]) * (Bp[j,j] - Bp[i,j])
        # This is harder to vectorize fully as it involves specific elements.
        # However, the correction terms are small compared to the main sum for large n.
        # For n=150, the error might be significant though.
        # Let's ignore the correction for the *selection* of the best move, 
        # and then compute the *exact* delta for the selected (i, j) before accepting.
        # This is a common approximation in heuristic searches.
        
        # Approximate Delta Matrix:
        diag_A = 2 * np.diag(a)
        diag_Bp = np.diag(Bp)
        T1 = (diag_A[:, None] - diag_A[None, :]) * (diag_Bp[None, :] - diag_Bp[:, None])
        
        A_skew = a - a.T
        Diff_B = Bp.T - Bp
        T2 = 2 * A_skew * Diff_B
        
        T3 = M1 - D_Asym_Bp[:, None] - D_Asym_Bp[None, :] + M4.T
        
        Delta_approx = T1 + T2 + T3
        
        # Set diagonal to inf (no swap with self)
        np.fill_diagonal(Delta_approx, np.inf)
        
        # Find best move
        min_val = np.min(Delta_approx)
        min_idx = np.unravel_index(np.argmin(Delta_approx), Delta_approx.shape)
        i, j = min_idx
        
        if i == j:
            continue
            
        # Calculate EXACT delta for (i, j)
        u = p[i]
        v = p[j]
        
        # Exact formula:
        # t1 = 2 * (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
        # t2 = 2 * (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        # t3 = sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[v,p[k]] - b[u,p[k]])
        
        t1 = 2 * (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
        t2 = 2 * (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        
        V = b[v, p] - b[u, p]
        W = A_sym[i] - A_sym[j]
        Sum_all = np.dot(W, V)
        
        D_i = b[v,u] - b[u,u]
        D_j = b[v,v] - b[u,v]
        
        Correction = (A_sym[i,i] - A_sym[j,i]) * D_i + (A_sym[i,j] - A_sym[j,j]) * D_j
        Sum_rest = Sum_all - Correction
        
        Delta_exact = t1 + t2 + Sum_rest
        
        # Acceptance criterion
        accept = False
        if Delta_exact < 0:
            accept = True
        else:
            # Metropolis criterion
            # Avoid overflow in exp
            if T > 0:
                prob = np.exp(-Delta_exact / T)
                if np.random.random() < prob:
                    accept = True
        
        if accept:
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += int(Delta_exact)
            
            # Update Bp and derived matrices
            # Only rows/cols i and j in Bp change?
            # Bp[x, y] = b[p[x], p[y]]
            # Changing p[i] and p[j] affects:
            # Row i: Bp[i, :] = b[p[i], p] -> b[v, p]
            # Row j: Bp[j, :] = b[p[j], p] -> b[u, p]
            # Col i: Bp[:, i] = b[p, p[i]] -> b[p, v]
            # Col j: Bp[:, j] = b[p, p[j]] -> b[p, u]
            
            # Swap rows i and j in Bp
            Bp[[i, j], :] = Bp[[j, i], :]
            # Swap cols i and j in Bp
            Bp[:, [i, j]] = Bp[:, [j, i]]
            
            # Update M1, M4, D_Asym_Bp
            # M1 = A_sym @ Bp
            # M4 = A_sym @ Bp.T
            # D_Asym_Bp = sum(A_sym * Bp, axis=1)
            
            # We can update these incrementally or recompute.
            # Recomputing is O(n^3) ~ 3M ops. Fast enough.
            M1 = A_sym @ Bp
            M4 = A_sym @ Bp.T
            D_Asym_Bp = np.sum(A_sym * Bp, axis=1)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # If stuck, perturb
        if no_improve_count > max_no_improve:
            # Random restart or large perturbation
            for _ in range(10):
                ri = np.random.randint(n)
                rj = np.random.randint(n)
                if ri != rj:
                    p[ri], p[rj] = p[rj], p[ri]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            
            # Reset Bp
            Bp = b[p][:, p]
            M1 = A_sym @ Bp
            M4 = A_sym @ Bp.T
            D_Asym_Bp = np.sum(A_sym * Bp, axis=1)
            
            no_improve_count = 0
            # Increase temperature to escape
            T = min(T * 2.0, T_start)

    return best_p.tolist()
# EVOLVE-BLOCK-END
