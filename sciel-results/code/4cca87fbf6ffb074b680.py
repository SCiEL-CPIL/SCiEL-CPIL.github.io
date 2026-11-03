# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    
    # 1. Initialization: Rank-matching heuristic
    # Heavy flows (high row sum in A) to Central locations (low row sum in B)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    def calc_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # 2. Local Search (2-opt)
    # We perform a full pass to find the best improving swap.
    # If found, apply it and repeat.
    # If no improving swap found, we are at a local optimum.
    
    while (time.time() - start_time) < deadline:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Precompute row sums of A for potential optimization, but direct delta is safer
        # For n=100, O(n^3) per pass is ~10^6 ops, which is very fast in numpy/python if vectorized or simple loops.
        # However, pure python loops for n^2 pairs * n ops might be slow.
        # Let's vectorize the delta calculation for a fixed i against all j.
        
        # M[i, j] = sum_k a[i, k] b[p[i], p[k]] is NOT correct for delta.
        # Let's use the explicit formula.
        
        for i in range(n):
            Li = p[i]
            # Vector for b[Li, p[k]] for all k
            # b[Li, p] is a vector of length n
            
            # We want Delta(i, j) for all j > i
            # Delta(i, j) = sum_{k != i, j} a[i, k] (b[Lj, pk] - b[Li, pk]) + 
            #               sum_{k != i, j} a[j, k] (b[Li, pk] - b[Lj, pk]) +
            #               a[i, i] (b[Lj, Lj] - b[Li, Li]) + a[j, j] (b[Li, Li] - b[Lj, Lj])
            
            # Let's compute terms for all j at once?
            # This is tricky because a[j, k] depends on j.
            
            # Given n=100, a simple nested loop with numpy vectorization for the inner sum is likely fastest.
            # Or just precompute C[i][j] = sum_k a[i, k] b[p[i], p[k]]? No, that's not the delta.
            
            # Let's stick to a robust O(n^3) approach with numpy vectorization for the sum over k.
            # For each i, for each j > i:
            #   delta = np.dot(a[i, [k for k not in i,j]], b[p[j], p[k not in i,j]] - b[p[i], p[k not in i,j]])
            #   ... this masking is slow.
            
            # Better:
            # Let S_i = sum_k a[i, k] b[p[i], p[k]]
            # This is not directly helpful.
            
            # Let's use the property:
            # Delta(i, j) = 
            #   (sum_{k != i, j} a[i, k] b[p[j], p[k]] - sum_{k != i, j} a[i, k] b[p[i], p[k]]) +
            #   (sum_{k != i, j} a[j, k] b[p[i], p[k]] - sum_{k != i, j} a[j, k] b[p[j], p[k]]) +
            #   a[i, i] * (b[p[j], p[j]] - b[p[i], p[i]]) +
            #   a[j, j] * (b[p[i], p[i]] - b[p[j], p[j]])
            
            # Let X_i = sum_k a[i, k] b[p[i], p[k]]
            # Let Y_i_j = sum_k a[i, k] b[p[j], p[k]]
            
            # We can precompute a matrix D where D[i, j] = sum_k a[i, k] b[p[j], p[k]].
            # D = A @ B_perm where B_perm[i, j] = b[p[i], p[j]]?
            # Let B_p be the matrix with B_p[i, j] = b[p[i], p[j]].
            # Then (A @ B_p)[i, j] = sum_k a[i, k] b[p[k], p[j]].
            # This is close, but indices are swapped in the second term.
            # We need sum_k a[i, k] b[p[j], p[k]].
            # Let B_pT be transpose of B_p. Since B is symmetric, B_p is symmetric.
            # So sum_k a[i, k] b[p[j], p[k]] = sum_k a[i, k] b[p[k], p[j]] = (A @ B_p)[i, j].
            
            # So, let M = A @ B_p.
            # M[i, j] = sum_k a[i, k] b[p[k], p[j]] = sum_k a[i, k] b[p[j], p[k]].
            
            # Then:
            # sum_{k != i, j} a[i, k] b[p[j], p[k]] = M[i, j] - a[i, i]b[p[i], p[j]] - a[i, j]b[p[j], p[j]]
            # sum_{k != i, j} a[i, k] b[p[i], p[k]] = M[i, i] - a[i, i]b[p[i], p[i]] - a[i, j]b[p[j], p[i]]
            
            # Since B is symmetric, b[p[i], p[j]] = b[p[j], p[i]].
            
            # Term 1 (i part):
            # T1 = (M[i, j] - a[i, i]b[p[i], p[j]] - a[i, j]b[p[j], p[j]]) - (M[i, i] - a[i, i]b[p[i], p[i]] - a[i, j]b[p[i], p[j]])
            # T1 = M[i, j] - M[i, i] - a[i, i]b[p[i], p[j]] - a[i, j]b[p[j], p[j]] + a[i, i]b[p[i], p[i]] + a[i, j]b[p[i], p[j]]
            # T1 = M[i, j] - M[i, i] + a[i, i](b[p[i], p[i]] - b[p[i], p[j]]) + a[i, j](b[p[i], p[j]] - b[p[j], p[j]])
            
            # Term 2 (j part):
            # T2 = (M[j, i] - M[j, j] + a[j, j](b[p[j], p[j]] - b[p[j], p[i]]) + a[j, i](b[p[j], p[i]] - b[p[i], p[i]]))
            
            # Term 3 (diagonal corrections):
            # T3 = a[i, i] * (b[p[j], p[j]] - b[p[i], p[i]]) + a[j, j] * (b[p[i], p[i]] - b[p[j], p[j]])
            
            # Total Delta = T1 + T2 + T3
            
            # Let's simplify T1 + T3:
            # T1 has: a[i, i](b[p[i], p[i]] - b[p[i], p[j]])
            # T3 has: a[i, i](b[p[j], p[j]] - b[p[i], p[i]])
            # Sum: a[i, i](b[p[j], p[j]] - b[p[i], p[j]])
            
            # Similarly for j in T2 + T3:
            # T2 has: a[j, j](b[p[j], p[j]] - b[p[j], p[i]])
            # T3 has: a[j, j](b[p[i], p[i]] - b[p[j], p[j]])
            # Sum: a[j, j](b[p[i], p[i]] - b[p[j], p[i]])
            
            # So Delta = M[i, j] - M[i, i] + M[j, i] - M[j, j] + 
            #           a[i, j](b[p[i], p[j]] - b[p[j], p[j]]) + a[j, i](b[p[j], p[i]] - b[p[i], p[i]]) +
            #           a[i, i](b[p[j], p[j]] - b[p[i], p[j]]) + a[j, j](b[p[i], p[i]] - b[p[j], p[i]])
            
            # Since A and B are symmetric:
            # a[i, j] = a[j, i]
            # b[p[i], p[j]] = b[p[j], p[i]]
            
            # Delta = M[i, j] + M[j, i] - M[i, i] - M[j, j] + 
            #           2 * a[i, j] * (b[p[i], p[j]] - b[p[j], p[j]]) + # Wait, check signs
            #           a[i, i](b[p[j], p[j]] - b[p[i], p[j]]) + a[j, j](b[p[i], p[i]] - b[p[j], p[i]])
            
            # Let's re-verify T1:
            # T1 = M[i, j] - M[i, i] + a[i, i](b[p[i], p[i]] - b[p[i], p[j]]) + a[i, j](b[p[i], p[j]] - b[p[j], p[j]])
            # T3_i = a[i, i](b[p[j], p[j]] - b[p[i], p[i]])
            # T1 + T3_i = M[i, j] - M[i, i] + a[i, i](b[p[i], p[i]] - b[p[i], p[j]] + b[p[j], p[j]] - b[p[i], p[i]]) + a[i, j](b[p[i], p[j]] - b[p[j], p[j]])
            # = M[i, j] - M[i, i] + a[i, i](b[p[j], p[j]] - b[p[i], p[j]]) + a[i, j](b[p[i], p[j]] - b[p[j], p[j]])
            
            # Similarly for j:
            # T2 + T3_j = M[j, i] - M[j, j] + a[j, j](b[p[i], p[i]] - b[p[j], p[i]]) + a[j, i](b[p[j], p[i]] - b[p[i], p[i]])
            
            # Total Delta = M[i, j] + M[j, i] - M[i, i] - M[j, j] + 
            #               a[i, i](b[p[j], p[j]] - b[p[i], p[j]]) + a[i, j](b[p[i], p[j]] - b[p[j], p[j]]) +
            #               a[j, j](b[p[i], p[i]] - b[p[j], p[i]]) + a[j, i](b[p[j], p[i]] - b[p[i], p[i]])
            
            # Grouping terms with a[i, j]:
            # a[i, j] (b[p[i], p[j]] - b[p[j], p[j]]) + a[j, i] (b[p[j], p[i]] - b[p[i], p[i]])
            # = a[i, j] (b[p[i], p[j]] - b[p[j], p[j]] + b[p[j], p[i]] - b[p[i], p[i]])
            # = a[i, j] (2 b[p[i], p[j]] - b[p[i], p[i]] - b[p[j], p[j]])
            
            # Grouping diagonal terms:
            # a[i, i](b[p[j], p[j]] - b[p[i], p[j]]) + a[j, j](b[p[i], p[i]] - b[p[j], p[i]])
            
            # So Delta = M[i, j] + M[j, i] - M[i, i] - M[j, j] + 
            #           a[i, j] * (2 * b[p[i], p[j]] - b[p[i], p[i]] - b[p[j], p[j]]) +
            #           a[i, i] * (b[p[j], p[j]] - b[p[i], p[j]]) +
            #           a[j, j] * (b[p[i], p[i]] - b[p[j], p[i]])
            
            # This formula is O(1) per pair if M is precomputed!
            # M = A @ B_p.
            # B_p[i, j] = b[p[i], p[j]].
            
            # We can precompute M once per pass.
            # Then iterate i, j to find min Delta.
            
            pass # Placeholder for logic below
            
        # Recompute M for current p
        B_p = b[np.ix_(p, p)]
        M = a @ B_p
        
        found_improvement = False
        for i in range(n):
            Li = p[i]
            for j in range(i + 1, n):
                Lj = p[j]
                
                # Calculate Delta using derived formula
                # Note: M[i, j] and M[j, i] are floats/ints
                delta = (M[i, j] + M[j, i] - M[i, i] - M[j, j] + 
                         a[i, j] * (2 * b[Li, Lj] - b[Li, Li] - b[Lj, Lj]) +
                         a[i, i] * (b[Lj, Lj] - b[Li, Lj]) +
                         a[j, j] * (b[Li, Li] - b[Lj, Lj]))
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_delta < 0:
            # Apply swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            found_improvement = True
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
            # Check time
            if time.time() - start_time > deadline:
                break
        else:
            # No improvement found
            break

    # 3. Simulated Annealing with remaining time
    if time.time() - start_time < deadline - 5.0:
        T = 5000.0 
        T_end = 1.0
        alpha = 0.995
        
        p = best_p.copy()
        current_cost = best_cost
        
        while T > T_end and (time.time() - start_time) < deadline:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
                
            # Calculate delta for swap i, j
            Li = p[i]
            Lj = p[j]
            
            # Reuse the O(1) delta formula?
            # We need M[i, j], M[j, i], M[i, i], M[j, j].
            # M changes after every swap.
            # Updating M is O(n) for a swap?
            # M = A @ B_p.
            # Swapping i, j in p swaps rows/cols i, j in B_p.
            # This updates M rows/cols i, j.
            # It is easier to just calculate delta directly for SA steps since we only do 1 step at a time.
            # Direct O(n) delta calculation:
            
            # Delta = sum_{k != i, j} a[i, k] (b[Lj, pk] - b[Li, pk]) + 
            #         sum_{k != i, j} a[j, k] (b[Li, pk] - b[Lj, pk]) +
            #         a[i, i] (b[Lj, Lj] - b[Li, Li]) + a[j, j] (b[Li, Li] - b[Lj, Lj])
            
            # Vectorized:
            # diff_B = b[Lj, p] - b[Li, p]
            # diff_B[i] = 0, diff_B[j] = 0
            # sum_ik = np.dot(a[i, :], diff_B)
            # sum_jk = np.dot(a[j, :], diff_B)
            # diag_corr = a[i, i] * (b[Lj, Lj] - b[Li, Li]) + a[j, j] * (b[Li, Li] - b[Lj, Lj])
            
            diff_B = b[Lj, p] - b[Li, p]
            diff_B[i] = 0
            diff_B[j] = 0
            
            delta = np.dot(a[i, :], diff_B) + np.dot(a[j, :], diff_B) + \
                    a[i, i] * (b[Lj, Lj] - b[Li, Li]) + a[j, j] * (b[Li, Li] - b[Lj, Lj])
            
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    
            T *= alpha
            
            if time.time() - start_time > deadline:
                break
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
