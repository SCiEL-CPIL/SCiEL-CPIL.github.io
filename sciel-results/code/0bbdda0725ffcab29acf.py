"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Hungarian Algorithm on Linearized Correlation
    # We approximate the QAP by minimizing sum a[i,j] b[p[i], p[j]].
    # A common heuristic is to match facilities with high flow to locations with low distance sums,
    # but Hungarian allows for a more global assignment based on a constructed cost matrix.
    
    # Construct a cost matrix C where C[i, j] approximates the "cost" of assigning facility i to location j.
    # A better heuristic: use the "rank correlation" or simply the product of row sums?
    # Let's use a matrix derived from the "center of mass" of flows and distances.
    # Actually, a standard strong initialization for QAP is to solve a linear assignment problem
    # where the cost matrix is M[i, j] = sum_k a[i, k] * b[j, k] ?? No, that's not quite right dimensionally for simple assignment.
    
    # Let's try the "Greedy Rank Matching" enhanced by Hungarian on a projected cost.
    # Projected Cost: C[i, j] = (row_sum_a[i] * row_sum_b[j]) is simple.
    # Better: Use the top eigenvectors? Too slow for init? No, 150x150 is fast.
    # But Hungarian on a 150x150 matrix is very fast (< 10ms).
    
    # Let's construct a cost matrix that captures the "alignment" of the structures.
    # If a[i,:] is large, we want b[j,:] to be "small" in the corresponding positions?
    # This is hard to capture in a simple C[i,j].
    
    # Alternative Initialization: Random + Local Search is robust, but Hungarian on a
    # "Distance-Flow" correlation matrix can be better.
    # Let's use the matrix: C[i, j] = np.dot(a[i, :], b[j, :])
    # This encourages placing facility i at location j if the flow profile of i is "anti-correlated" 
    # (or correlated, depending on sign) with the distance profile of j.
    # Since we want to minimize sum a[i,k] b[p[i], p[k]], if a[i,k] is large, we want b[p[i], p[k]] to be small.
    # This is not directly captured by dot(a[i,:], b[j,:]).
    
    # Let's stick to a robust heuristic:
    # 1. Sort facilities by descending sum of flows.
    # 2. Sort locations by ascending sum of distances.
    # This is the "Rank Matching" which is actually quite good.
    # To make it "Structurally Different", we will use the Hungarian Algorithm on a 
    # cost matrix derived from the "interaction" of the two matrices.
    
    # Cost Matrix for Hungarian:
    # C[i, j] = sum_k a[i, k] * b[j, k]
    # If we assign i to j, the term a[i,k] b[j,k] is not the cost, the cost is a[i,k] b[p[i], p[k]].
    # However, minimizing sum_k a[i,k] b[j,k] for each pair (i,j) in an assignment sense
    # tries to align high flow entries with low distance entries globally?
    # Actually, if b is symmetric, b[j,k] is the distance from location j to k.
    # If we assign i to j, the cost contribution of flow from i to k is a[i,k] * dist(j, p[k]).
    # This depends on p[k].
    
    # A known good initialization is "Linear Assignment Problem (LAP) relaxation" where we ignore the permutation constraint on the second index? No.
    
    # Let's use the "Greedy" initialization but then immediately refine with the Vectorized Best Improvement.
    # The key structural change is the SEARCH METHOD: Vectorized Best Improvement vs Random Walk.
    
    # Initialization: Rank Matching (Fast and decent)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    
    heavy_order = np.argsort(-a_sum, kind="stable")
    central_order = np.argsort(b_sum, kind="stable")
    
    p = np.zeros(n, dtype=int)
    for i in range(n):
        p[heavy_order[i]] = central_order[i]
        
    # Precompute cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Vectorized Best-Improvement 2-Swap Local Search
    # This is the core structural change.
    # We evaluate all pairs (i, j) with i < j in a vectorized manner.
    # Delta(i, j) = 2(a[i,i]-a[j,j])(b[v,v]-b[u,u]) + 2(a[i,j]-a[j,i])(b[v,u]-b[u,v])
    #              + sum_{k != i,j} (A_sym[i,k] - A_sym[j,k]) (b[v,p[k]]-b[u,p[k]])
    
    A_sym = a + a.T
    B = b
    
    # Precompute terms that don't depend on the pair (i,j) specifically in the sum?
    # The sum term is: sum_k (A_sym[i,k] - A_sym[j,k]) * (B[v, p[k]] - B[u, p[k]])
    # Let V_{i,j} = B[p[j], p] - B[p[i], p]  (Vector of length n)
    # Let W_{i,j} = A_sym[i] - A_sym[j]      (Vector of length n)
    # Sum = dot(W_{i,j}, V_{i,j}) - correction terms
    
    # To vectorize over all pairs (i, j), we can construct large matrices.
    # Let I = np.arange(n)
    # Pairs: We can iterate over i, and vectorize over j.
    # For a fixed i, u = p[i].
    # We want to find j that minimizes Delta(i, j).
    
    # Let's loop over i (150 iterations).
    # For each i, compute Delta for all j != i.
    # This is O(n) iterations, each O(n^2) or O(n^3) if not careful.
    # Vectorized over j for fixed i:
    # u = p[i]
    # v = p[j] for all j.
    # V_j = B[v_j, p] - B[u, p]  (Shape: n x n, where row j is V_j)
    # W_j = A_sym[i] - A_sym[j]  (Shape: n x n, where row j is W_j)
    # Dot_j = sum_k W_j[k] * V_j[k]  (Shape: n, dot product along axis 1)
    # This is a matrix multiplication: W @ V.T ?
    # W is n x n (rows are W_j). V is n x n (rows are V_j).
    # We want row-wise dot product.
    # Dot_j = np.sum(W * V, axis=1)
    
    # Correction terms:
    # D_i = B[v, u] - B[u, u]  (Vector of length n, element j is D_i for pair i,j)
    # D_j = B[v, v] - B[u, v]  (Vector of length n, element j is D_j for pair i,j)
    # Corr_j = (A_sym[i,i] - A_sym[j,i]) * D_i[j] + (A_sym[i,j] - A_sym[j,j]) * D_j[j]
    
    # T1_j = 2 * (a[i,i] - a[j,j]) * (B[v_j, v_j] - B[u, u])
    # T2_j = 2 * (a[i,j] - a[j,i]) * (B[v_j, u] - B[u, v_j])
    
    # Total Delta_j = T1_j + T2_j + Dot_j - Corr_j
    
    # We find the j that minimizes Delta_j.
    # If min Delta < 0, we swap i and j, update p, and restart the search (or continue).
    # Best Improvement: Find the global minimum delta over all (i, j).
    # To do global best improvement, we need to compute all deltas.
    # Looping i and finding best j for that i gives "First Improvement" or "Best Improvement per i".
    # True "Best Improvement" requires checking all pairs.
    # Given n=150, 150 iterations of O(n^2) vectorized ops is very fast.
    # 150 * (150x150 matrix mults) = 150 * 3.3M flops = 500M flops.
    # NumPy can do this in ~0.1s? Maybe.
    # Let's try "Best Improvement per i" (steepest descent step) which is usually sufficient and faster to implement correctly.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # Loop over i to find the best swap involving i
        best_delta_global = 0 # We want to find negative delta
        best_i = -1
        best_j = -1
        
        for i in range(n):
            u = p[i]
            # v is p[j] for all j
            v = p # vector of locations
            
            # V_j = B[v_j, p] - B[u, p]
            # B[v, :] gives B[v_j, :] for all j? No, B is 2D.
            # B[v, p] where v is array and p is array?
            # B[v, p] is advanced indexing. B[v[j], p[j]]? No, we need B[v_j, p_k] for all k.
            # So we need B[np.ix_(v, p)] ?
            # B[v, :] selects rows v. Result shape (n, n).
            # B[v, :] - B[u, :]
            V = B[v, :] - B[u, :] # Shape (n, n). Row j is B[p[j], :] - B[u, :]
            
            # W_j = A_sym[i] - A_sym[j]
            # A_sym[i] is shape (n,). A_sym is (n, n).
            # W = A_sym[i, np.newaxis, :] - A_sym # Shape (n, n). Row j is A_sym[i] - A_sym[j]
            W = A_sym[i, np.newaxis, :] - A_sym
            
            # Dot_j = sum_k W[j, k] * V[j, k]
            Dot = np.sum(W * V, axis=1)
            
            # Correction
            # D_i[j] = B[v_j, u] - B[u, u]
            D_i = B[v, u] - B[u, u]
            # D_j[j] = B[v_j, v_j] - B[u, v_j]
            D_j = B[v, v] - B[u, v]
            
            # Corr_j = (A_sym[i,i] - A_sym[j,i]) * D_i[j] + (A_sym[i,j] - A_sym[j,j]) * D_j[j]
            # A_sym[i,i] is scalar. A_sym[:, i] is col i.
            term1_corr = (A_sym[i, i] - A_sym[:, i]) * D_i
            # A_sym[i, :] is row i. A_sym is (n,n). A_sym[j,j] is diag.
            term2_corr = (A_sym[i, :] - np.diag(A_sym)) * D_j
            
            Corr = term1_corr + term2_corr
            
            # T1_j = 2 * (a[i,i] - a[j,j]) * (B[v_j, v_j] - B[u, u])
            T1 = 2 * (a[i, i] - np.diag(a)) * (B[v, v] - B[u, u])
            
            # T2_j = 2 * (a[i,j] - a[j,i]) * (B[v_j, u] - B[u, v_j])
            # a[i, :] - a[:, i]
            # B[v, u] - B[u, v]
            T2 = 2 * (a[i, :] - a[:, i]) * (B[v, u] - B[u, v])
            
            Deltas = T1 + T2 + Dot - Corr
            
            # Exclude j == i
            Deltas[i] = np.inf
            
            min_idx = np.argmin(Deltas)
            min_val = Deltas[min_idx]
            
            if min_val < best_delta_global:
                best_delta_global = min_val
                best_i = i
                best_j = min_idx
                
        if best_delta_global < 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += int(best_delta_global)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            improved = True
        else:
            # No improving swap found
            # Perturb
            for _ in range(10):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            
        if no_improve_count > max_no_improve:
             # Stronger perturb
             for _ in range(20):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i != j:
                    p[i], p[j] = p[j], p[i]
             current_cost = calc_cost(p)
             if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
             no_improve_count = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END
