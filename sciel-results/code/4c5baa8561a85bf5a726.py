"""Spectral Initialization + Vectorized Greedy Local Search for QAPLIB wil100."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # 1. Spectral Initialization (Karmarkar-Vannucci)
    # We use the eigenvectors of A and B to create an initial assignment.
    # This is often better than simple row-sum matching for structured instances.
    
    # Use the largest eigenvector (or a few) to define the "shape" of the solution.
    # For symmetric matrices, we can use eigh.
    try:
        # Get the largest eigenvector of A and B
        # We want to align high-flow facilities with low-distance locations in a spectral sense.
        # Actually, the standard spectral approach minimizes ||A - B P^T P|| or similar.
        # A simpler heuristic: Project facilities and locations onto the principal components.
        
        # Compute eigenvectors. We only need the top few to capture structure.
        # For n=100, eigh is fast enough.
        _, va = np.linalg.eigh(a)
        _, vb = np.linalg.eigh(b)
        
        # Take the last few eigenvectors (largest eigenvalues)
        k = 10
        va_top = va[:, -k:]
        vb_top = vb[:, -k:]
        
        # We want to assign facility i to location j such that va_top[i] is close to vb_top[j]
        # Or rather, we want to maximize the correlation.
        # Let's compute a cost matrix based on the negative dot product of the spectral features.
        # C[i, j] = - dot(va_top[i], vb_top[j])
        # Minimizing C is equivalent to maximizing the alignment.
        
        # However, the standard spectral method for QAP is:
        # Solve min tr(A P^T B P) ~ min ||A^{1/2} P B^{1/2} P^T||_F^2
        # This leads to matching the eigenvectors.
        
        # Let's try a simpler spectral heuristic:
        # Compute the "spectral signature" of each facility and location.
        # And use Hungarian on the distance between these signatures.
        
        # Distance in spectral space
        # sig_a = va_top (n x k)
        # sig_b = vb_top (n x k)
        # dist[i, j] = ||sig_a[i] - sig_b[j]||^2
        
        # Actually, for QAP, we usually want to map high flow to low distance.
        # The spectral method aligns the "structure".
        # Let's compute the cost matrix for Hungarian as:
        # C[i, j] = - sum_m (va[i, m] * vb[j, m])  (Maximize correlation)
        # But we must be careful with signs. Eigenvectors are defined up to sign.
        # To fix signs, we can force the largest component to be positive, or align them.
        
        # Let's align the signs of va and vb so that their dot product is positive.
        # For each column m, if sum(va[:, m] * vb[:, m]) < 0, flip vb[:, m].
        for m in range(k):
            if np.sum(va_top[:, m] * vb_top[:, m]) < 0:
                vb_top[:, m] = -vb_top[:, m]
                
        # Now compute cost matrix
        # We want to minimize sum a[i,j] b[p[i], p[j]].
        # The spectral approximation suggests matching similar spectral profiles.
        # Cost C[i, j] = ||va_top[i] - vb_top[j]||^2
        # This is a valid assignment problem.
        
        # Compute squared Euclidean distance
        # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u.v
        va_norm_sq = np.sum(va_top**2, axis=1)
        vb_norm_sq = np.sum(vb_top**2, axis=1)
        va_vb_dot = va_top @ vb_top.T
        
        cost_matrix = va_norm_sq[:, np.newaxis] + vb_norm_sq[np.newaxis, :] - 2 * va_vb_dot
        
        # Solve Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
    except Exception:
        # Fallback to simple row-sum matching if spectral fails
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        cost_matrix = np.outer(flow_sums, dist_sums)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Vectorized Greedy Local Search
    # We compute the delta for swapping i and j for all j, and pick the best improvement.
    # Delta(i, j) = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]])
    # Let u = p[i], v = p[j].
    # Delta(i, j) = 2 * ( (a[i] - a[j]) @ (b[v, p] - b[u, p]) - correction terms )
    
    # Precompute for speed?
    # In the loop, we will iterate i from 0 to n-1.
    # For a fixed i, we can compute the delta for all j > i at once?
    # Delta(i, j) depends on p[i] and p[j].
    # Let u = p[i].
    # term_j = (a[i] - a[j]) @ (b[p[j], p] - b[u, p])
    # This is not easily vectorizable over j because b[p[j], p] changes.
    
    # However, we can vectorize the inner loop for a fixed i.
    # Let's stick to a semi-vectorized approach or just optimize the Python loop.
    # Given n=100, n^2/2 = 5000 swaps per pass.
    # 120s / 5000 swaps = 24ms per swap. We can do much more.
    # Python loop overhead is the bottleneck.
    
    # Let's try to vectorize the delta calculation for a fixed i against all j.
    # For fixed i (location u=p[i]):
    # We want to find j that minimizes Delta(i, j).
    # Delta(i, j) = 2 * [ sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[u, p[k]]) - corrections ]
    # Let d_a_j = a[i] - a[j]
    # Let d_b_j = b[p[j], p] - b[u, p]
    # Dot product d_a_j @ d_b_j
    
    # This is still hard to fully vectorize over j without memory blowup or complex indexing.
    # Let's use a simpler vectorized update:
    # Compute the gradient of the cost with respect to moving facility i to location v.
    # If we move i from u to v (and j from v to u), the change is:
    # Delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
    
    # Let's implement a fast first-improvement search.
    
    start_time = time.time()
    timeout = 115.0
    
    # Precompute row sums for a and b? No, we need full vectors.
    
    # Optimize the local search by checking pairs in a random order or specific order.
    # Random order helps escape symmetries.
    
    while time.time() - start_time < timeout:
        improved = False
        # Iterate over all pairs
        # To speed up, we can break early if no improvement is found in a full pass?
        # No, we want to keep improving.
        
        # Let's try to vectorize the delta calculation for a fixed i.
        # For a fixed i, u = p[i].
        # We want to find the best j.
        # delta(i, j) = 2 * ( (a[i] - a[j]) @ (b[p[j], p] - b[u, p]) - term_i - term_j )
        
        # Let's compute the base term for all j at once?
        # Let M = b[p, p] (the current permuted B)
        # Then b[p[j], p] is M[j, :]
        # b[u, p] is M[p.index(u), :] ... wait, p is the permutation.
        # b[p[j], p[k]] = M[j, k] where M = b[np.ix_(p, p)]
        
        # So for fixed i (u=p[i]), we want to find j such that:
        # (a[i] - a[j]) @ (M[j, :] - M[i, :]) is minimized.
        # Note: p[k] is the location of facility k.
        # M[j, k] = b[p[j], p[k]].
        
        # Let's precompute M = b[np.ix_(p, p)]
        # This takes O(n^2) time.
        
        # Inside the loop:
        for i in range(n):
            u = p[i]
            # We need M[i, :] which is b[u, p]
            # And we need to compare with M[j, :] for all j != i.
            
            # Let's compute the delta for all j at once.
            # diff_a = a[i] - A (n x n matrix where row j is a[i] - a[j])
            # This is a[i][np.newaxis, :] - a
            diff_a = a[i][np.newaxis, :] - a
            
            # diff_b = M[j, :] - M[i, :] for all j
            # M is n x n.
            # diff_b = M - M[i][np.newaxis, :]
            
            # dot = diff_a @ diff_b.T ? No.
            # We want sum_k (a[i,k] - a[j,k]) * (M[j,k] - M[i,k])
            # This is element-wise product summed over k.
            # Let P = diff_a * (M - M[i][np.newaxis, :])
            # Then sum_k P[j, k] gives the sum for each j.
            
            # But we must exclude k=i and k=j.
            # And we must exclude j=i.
            
            # Let's compute the full sum first.
            # S_j = sum_k (a[i,k] - a[j,k]) * (M[j,k] - M[i,k])
            
            # This is O(n^3) if done naively for each i.
            # n=100, n^3 = 1,000,000 operations per i.
            # 100 * 1,000,000 = 100,000,000 operations per pass.
            # This is too slow in Python/NumPy if not optimized.
            
            # Let's stick to the O(n^2) pass with optimized Python loops or small vectorization.
            # Given the time limit, we might only get a few passes.
            
            # Let's try a different vectorization:
            # For fixed i, compute delta for all j.
            # M is precomputed.
            # term_k = (a[i,k] - a[j,k]) * (M[j,k] - M[i,k])
            
            # Let's just use the standard loop but with minimal overhead.
            u_val = p[i]
            for j in range(i + 1, n):
                v_val = p[j]
                
                # Calculate delta
                # delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v_val, p[k]] - b[u_val, p[k]])
                
                # Use precomputed M?
                # M[j, k] = b[p[j], p[k]] = b[v_val, p[k]]
                # M[i, k] = b[p[i], p[k]] = b[u_val, p[k]]
                
                # So we can use M.
                # But M changes after every swap.
                # Updating M after a swap is O(n).
                
                # Let's maintain M in the loop.
                # If we swap i and j:
                # p[i], p[j] swap.
                # M[i, :], M[j, :] swap.
                # M[:, i], M[:, j] swap.
                # M[i, i], M[j, j] stay same (diagonal).
                # M[i, j], M[j, i] stay same (symmetric).
                
                # This is efficient!
                
                # Let's restructure the local search to maintain M.
        
        # Break out and implement the M-maintaining version.
        pass

    # Re-implementation with M maintenance
    M = b[np.ix_(p, p)]
    current_cost = int(np.sum(a * M))
    best_cost = current_cost
    best_p = p.copy()
    best_M = M.copy()
    
    start_time = time.time()
    timeout = 115.0
    
    while time.time() - start_time < timeout:
        improved = False
        # First improvement or Best improvement?
        # Best improvement is slower but better.
        # Let's try First Improvement with random restarts or just sequential.
        
        # To make it faster, we can randomize the order of i, j.
        # Or just do sequential.
        
        for i in range(n):
            u = p[i]
            # We can break if time is critical
            if time.time() - start_time > timeout - 2:
                break
                
            for j in range(i + 1, n):
                v = p[j]
                
                # Calculate delta using M
                # delta = 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (M[j,k] - M[i,k])
                
                # Vectorized calculation for this pair
                # diff_a = a[i] - a[j]
                # diff_M = M[j] - M[i]
                # sum_all = np.sum(diff_a * diff_M)
                
                # Corrections:
                # k=i: (a[i,i] - a[j,i]) * (M[j,i] - M[i,i])
                # k=j: (a[i,j] - a[j,j]) * (M[j,j] - M[i,j])
                
                # Note: M is symmetric, a is symmetric.
                # M[j,i] = M[i,j]
                # M[i,i] = b[u,u]
                # M[j,j] = b[v,v]
                
                # Let's just compute the full sum and subtract the two terms.
                # This is O(n) per pair.
                
                diff_a = a[i] - a[j]
                diff_M = M[j] - M[i]
                
                sum_all = np.dot(diff_a, diff_M)
                
                term_i = (a[i, i] - a[j, i]) * (M[j, i] - M[i, i])
                term_j = (a[i, j] - a[j, j]) * (M[j, j] - M[i, j])
                
                delta = 2 * (sum_all - term_i - term_j)
                
                if delta < 0:
                    # Accept
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    
                    # Update M
                    # Swap rows/cols i and j in M
                    M[[i, j], :] = M[[j, i], :]
                    M[:, [i, j]] = M[:, [j, i]]
                    
                    # Break inner loop? No, continue to find more improvements.
                    # But if we break, we do more passes.
                    # Let's continue.
        
        if not improved:
            # Local optimum reached.
            # Perturb
            if time.time() - start_time > timeout - 5:
                break
                
            # Random swap perturbation
            num_swaps = 10
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j:
                    continue
                # Calculate delta
                u = p[i]
                v = p[j]
                diff_a = a[i] - a[j]
                diff_M = M[j] - M[i]
                sum_all = np.dot(diff_a, diff_M)
                term_i = (a[i, i] - a[j, i]) * (M[j, i] - M[i, i])
                term_j = (a[i, j] - a[j, j]) * (M[j, j] - M[i, j])
                delta = 2 * (sum_all - term_i - term_j)
                
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                M[[i, j], :] = M[[j, i], :]
                M[:, [i, j]] = M[:, [j, i]]
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            best_M = M.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
