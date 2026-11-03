"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Spectral Initialization
    # The QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]] = Tr(P^T A P B)
    # Spectral heuristic: Map facilities to locations based on similarity of their "spectral signatures".
    # We use the top eigenvector of A and B.
    # Note: For minimization, we often look at the largest magnitude eigenvalues.
    # If A and B have similar structures, aligning their top eigenvectors helps.
    
    try:
        # Use symmetric part for stability if matrices are not perfectly symmetric (though QAPLIB usually is)
        # A and B are assumed symmetric.
        w_a, v_a = np.linalg.eigh(a)
        w_b, v_b = np.linalg.eigh(b)
        
        # Take the eigenvector corresponding to the largest magnitude eigenvalue
        # Usually the last one if sorted ascending by eigh
        # Let's pick the one with max absolute eigenvalue to be safe, or just the largest algebraic?
        # For QAP, the "dominant" mode is often the largest positive or largest negative.
        # Let's try the one with the largest absolute eigenvalue.
        
        idx_a = np.argmax(np.abs(w_a))
        idx_b = np.argmax(np.abs(w_b))
        
        vec_a = v_a[:, idx_a]
        vec_b = v_b[:, idx_b]
        
        # We want to assign facility i to location j such that the "directions" align?
        # Or such that the product is minimized?
        # If we treat vec_a and vec_b as coordinates, we want to match similar "types".
        # A common spectral heuristic for QAP is to sort facilities by the value of the top eigenvector
        # and sort locations by the value of the top eigenvector, then match rank to rank.
        # However, the sign matters. If one is positive and one negative, we might need to flip.
        # Let's compute the dot product of the two global vectors. If negative, flip one.
        if np.dot(vec_a, vec_b) < 0:
            vec_b = -vec_b
            
        # Rank matching based on spectral values
        # Sort facilities by vec_a values
        # Sort locations by vec_b values
        # Match i-th smallest facility to i-th smallest location?
        
        order_a = np.argsort(vec_a, kind="stable")
        order_b = np.argsort(vec_b, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        # Facility at rank k in A gets Location at rank k in B
        p[order_a] = order_b
        
    except Exception:
        # Fallback to rank matching if spectral fails
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        heavy_order = np.argsort(-a_sum, kind="stable")
        central_order = np.argsort(b_sum, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order

    # Refine with Hungarian algorithm on a linearized approximation?
    # The linear part of the cost for facility i is sum_j a[i,j] b[p[i], p[j]]
    # This is hard to linearize directly without fixing others.
    # However, we can use the spectral solution as the start for local search.
    
    def calc_cost(p):
        b_perm = b[p][:, p]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Local Search: 2-opt swaps with Tabu
    tabu_size = 30
    tabu_list = {} # (i, j) -> expiry step
    current_step = 0
    no_improve_count = 0
    max_no_improve = 5000 # Increased patience
    
    # Precompute a for faster access? a is already numpy.
    
    while time.time() < deadline:
        # Check for restart
        if no_improve_count > max_no_improve:
            # Perturb: random restart or heavy perturbation
            # Let's try a random restart mixed with spectral
            if np.random.rand() < 0.5:
                 # Random permutation
                 np.random.shuffle(p)
            else:
                 # Spectral with noise
                 try:
                    w_a, v_a = np.linalg.eigh(a)
                    w_b, v_b = np.linalg.eigh(b)
                    idx_a = np.argmax(np.abs(w_a))
                    idx_b = np.argmax(np.abs(w_b))
                    vec_a = v_a[:, idx_a] + 0.1 * np.random.randn(n)
                    vec_b = v_b[:, idx_b] + 0.1 * np.random.randn(n)
                    if np.dot(vec_a, vec_b) < 0:
                        vec_b = -vec_b
                    order_a = np.argsort(vec_a, kind="stable")
                    order_b = np.argsort(vec_b, kind="stable")
                    p = np.empty(n, dtype=np.int64)
                    p[order_a] = order_b
                 except:
                    np.random.shuffle(p)
            
            current_cost = calc_cost(p)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            no_improve_count = 0
            tabu_list.clear()
            continue
            
        # Find best move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Optimization: Precompute Bp = b[:, p]
        Bp = b[:, p]
        
        # We can vectorize the delta calculation for a fixed i against all j > i
        # delta(i, j) = (a[i] - a[j]) dot (Bp[p[j]] - Bp[p[i]])
        # = a[i] dot Bp[p[j]] - a[i] dot Bp[p[i]] - a[j] dot Bp[p[j]] + a[j] dot Bp[p[i]]
        
        # Let's stick to the loop but optimize inner operations.
        # For each i, precompute:
        # ai_dot_Bp_u = np.dot(a[i], Bp[p[i]])  (constant for fixed i)
        # ai_dot_Bp = a[i] @ Bp  (vector of length n, dot with each location)
        # Wait, Bp is (n, n) where Bp[k, m] = b[k, p[m]].
        # We need Bp[v, k] for k in 0..n-1.
        
        for i in range(n):
            u = p[i]
            ai = a[i]
            
            # Precompute terms for i
            # term1 = ai dot Bp[v] for all v? No, v is p[j].
            # We want delta(i, j) for j > i.
            # delta = ai dot Bp[p[j]] - ai dot Bp[u] - a[j] dot Bp[p[j]] + a[j] dot Bp[u]
            
            # Let's compute ai_dot_Bp_u once
            ai_dot_u = np.dot(ai, Bp[u])
            
            # Compute ai_dot_Bp_pj for all j?
            # ai @ Bp gives a vector of length n where element k is ai dot Bp[:, k] = ai dot b[:, p[k]]
            # So (ai @ Bp)[j] = ai dot b[:, p[j]]
            ai_dot_all_locs = ai @ Bp
            
            # Now for each j > i:
            # delta = ai_dot_all_locs[j] - ai_dot_u - (a[j] dot Bp[p[j]]) + (a[j] dot Bp[u])
            
            # We need a[j] dot Bp[p[j]] and a[j] dot Bp[u] for all j.
            # This is hard to vectorize over j without a loop or heavy matrix ops.
            # a[j] dot Bp[p[j]] is the diagonal of A @ Bp @ P? No.
            # Let's just loop j but use precomputed ai_dot_all_locs.
            
            for j in range(i + 1, n):
                key = (i, j)
                if key in tabu_list and tabu_list[key] > current_step:
                    continue
                
                v = p[j]
                aj = a[j]
                
                # delta = (ai - aj) dot (Bp[v] - Bp[u])
                # = ai dot Bp[v] - ai dot Bp[u] - aj dot Bp[v] + aj dot Bp[u]
                
                # ai dot Bp[v] is ai_dot_all_locs[j] because Bp[v] is the column of Bp corresponding to location v?
                # No. Bp = b[:, p].
                # Bp[k, m] = b[k, p[m]].
                # So Bp[:, m] = b[:, p[m]].
                # We want dot with Bp[v, :] ? No, delta formula uses Bp[v, k] where k is index in 0..n-1.
                # Wait, the delta formula derived earlier:
                # sum_{k != i,j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
                # Note: b[p[j], p[k]] is the element at row p[j], col p[k] in b.
                # In Bp = b[:, p], the element Bp[row, col] = b[row, p[col]].
                # So Bp[p[j], k] = b[p[j], p[k]].
                # So we need dot of (a[i] - a[j]) with (Bp[p[j], :] - Bp[p[i], :]).
                
                # Let's re-evaluate the vectorization.
                # diff_A = a[i] - a[j]
                # diff_B_row = Bp[p[j]] - Bp[p[i]]
                # delta = np.dot(diff_A, diff_B_row)
                
                # This is 1 dot product of length n.
                # My previous vectorization attempt `ai @ Bp` computed ai dot Bp[:, k] for all k.
                # That is ai dot b[:, p[k]].
                # But we need ai dot Bp[p[j], :] which is ai dot b[p[j], p[0..n-1]].
                # These are different. ai @ Bp gives row vector.
                # We need column vectors from Bp.
                
                # So we can't easily precompute "ai dot Bp[v]" for all v using a single matrix multiply
                # unless we transpose.
                # (Bp.T @ ai) gives a vector where element v is Bp[v, :] dot ai = b[p[v], p[0..n-1]] dot ai.
                # Yes!
                
                # Let's precompute for fixed i:
                # ai_dot_Bp_rows = Bp.T @ ai  (length n vector)
                # ai_dot_Bp_rows[v] = sum_k Bp[v, k] a[i, k] = sum_k b[v, p[k]] a[i, k]
                
                # Then delta(i, j) = ai_dot_Bp_rows[p[j]] - ai_dot_Bp_rows[p[i]] 
                #                    - (a[j] dot Bp[p[j], :]) + (a[j] dot Bp[p[i], :])
                
                # The term (a[j] dot Bp[p[j], :]) is still dependent on j.
                # Let's just use the simple dot product in the inner loop, it's likely fast enough for n=150
                # if we don't do too much overhead.
                
                # delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                
                # To speed up, we can precompute Bp[v] - Bp[u] ? No, u changes with i, v with j.
                
                # Let's try a slightly different vectorization for the inner loop:
                # For fixed i, we want best j.
                # delta(i, j) = (ai - a[j]) dot (Bp[p[j]] - Bp[u])
                
                # Let's just run the loop. n=150 is small.
                
                # Optimization: Check if we can skip if ai is very similar to aj? No.
                
                delta = np.dot(ai - a[j], Bp[v] - Bp[u])
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            tabu_list.clear()
            continue
            
        # Perform swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += int(best_delta)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        tabu_list[(best_i, best_j)] = current_step + tabu_size
        tabu_list[(best_j, best_i)] = current_step + tabu_size
        
        current_step += 1
        
        if current_step % 100 == 0:
            keys_to_remove = [k for k, v in tabu_list.items() if v <= current_step]
            for k in keys_to_remove:
                del tabu_list[k]

    return best_p.tolist()
# EVOLVE-BLOCK-END
