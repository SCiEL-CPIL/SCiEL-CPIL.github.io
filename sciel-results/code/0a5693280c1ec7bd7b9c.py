"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # 1. Spectral Initialization
    # Compute eigenvectors of flow matrix A and distance matrix B
    # We use the top eigenvector to map facilities to locations.
    try:
        # Symmetrize for eigendecomposition if not already (QAP matrices are usually symmetric)
        A_sym = (a + a.T) / 2.0
        B_sym = (b + b.T) / 2.0
        
        # Use float64 for spectral methods
        A_float = A_sym.astype(np.float64)
        B_float = B_sym.astype(np.float64)
        
        # Get top eigenvectors
        # For n=150, full eigendecomposition is fast enough
        eigvals_A, eigvecs_A = np.linalg.eigh(A_float)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_float)
        
        # Sort eigenvectors by eigenvalue (eigh returns ascending, so reverse)
        idx_A = np.argsort(eigvals_A)[::-1]
        idx_B = np.argsort(eigvals_B)[::-1]
        
        v_A = eigvecs_A[:, idx_A[0]]
        v_B = eigvecs_B[:, idx_B[0]]
        
        # Rank matching based on spectral coordinates
        # High flow facility -> High distance location? 
        # QAP minimization: usually align signs or ranks appropriately.
        # Let's try matching the ranks of the top eigenvectors.
        rank_A = np.argsort(v_A)
        rank_B = np.argsort(v_B)
        
        p = np.empty(n, dtype=np.int64)
        # Facility i (ranked by v_A) goes to location j (ranked by v_B)
        # p[facility] = location
        # So if facility i is the k-th largest in v_A, it should go to the k-th largest in v_B?
        # Let's map the sorted indices.
        # facilities_sorted = np.argsort(v_A)
        # locations_sorted = np.argsort(v_B)
        # p[facilities_sorted[k]] = locations_sorted[k]
        
        fac_sorted = np.argsort(v_A)
        loc_sorted = np.argsort(v_B)
        
        p[fac_sorted] = loc_sorted
        
    except Exception:
        # Fallback to rank matching if spectral fails
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_idx = np.argsort(-a_sums, kind="stable")
        central_idx = np.argsort(b_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_idx] = central_idx

    # Calculate initial cost
    cost = int(np.sum(a * b[p]))
    
    best_cost = cost
    best_p = p.copy()

    # Maintain interaction sums: c[i] = sum_j a[i,j] * b[p[i], p[j]]
    # c[i] represents the contribution of facility i to the total cost, 
    # specifically the sum of flows from i to all j multiplied by distances from p[i] to p[j].
    # Total cost = sum_i c[i] / 2 ? No.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let c[i] = sum_j a[i,j] b[p[i], p[j]]
    # Then Cost = sum_i c[i]. Wait, if i!=j, the term a[i,j]b[p[i],p[j]] is counted in c[i] and a[j,i]b[p[j],p[i]] in c[j].
    # Since a and b are symmetric, a[i,j]=a[j,i] and b[p[i],p[j]]=b[p[j],p[i]].
    # So term (i,j) appears in c[i] as a[i,j]b[p[i],p[j]] and in c[j] as a[j,i]b[p[j],p[i]].
    # Thus sum_i c[i] = 2 * Cost (excluding diagonal? No, diagonal is a[i,i]b[p[i],p[i]]).
    # Actually, sum_i c[i] = sum_i sum_j a[i,j] b[p[i], p[j]] = Cost.
    # So Cost = sum(c).
    
    def compute_c(p, a, b):
        """Compute c[i] = sum_j a[i,j] * b[p[i], p[j]] for all i"""
        c = np.zeros(n, dtype=np.int64)
        for i in range(n):
            c[i] = int(np.dot(a[i], b[p[i], p]))
        return c

    c = compute_c(p, a, b)
    
    # Verify cost calculation
    # cost should be sum(c)
    # Let's double check with direct sum occasionally or just trust it.
    # Direct sum is O(n^2) vectorized, which is fast.
    # c computation is O(n^2) in Python loop, which is slow (150 * 150 dot products).
    # We can vectorize c computation:
    # c[i] = dot(a[i], b[p[i], p])
    # This is hard to fully vectorize without advanced indexing which creates large arrays.
    # However, we can update c efficiently during swaps.
    
    # Delta for swap (i, j):
    # Let u = p[i], v = p[j].
    # Old terms involving i or j:
    # In c[i]: sum_k a[i,k] b[u, p[k]]
    # In c[j]: sum_k a[j,k] b[v, p[k]]
    # Other c[k] for k != i,j:
    # c[k] = sum_m a[k,m] b[p[k], p[m]]
    # Terms involving u or v in p[m]:
    # If m=i, term is a[k,i] b[p[k], u]. In new, p[i]=v, so a[k,i] b[p[k], v].
    # If m=j, term is a[k,j] b[p[k], v]. In new, p[j]=u, so a[k,j] b[p[k], u].
    # So delta for c[k] (k!=i,j) is:
    # a[k,i] (b[p[k], v] - b[p[k], u]) + a[k,j] (b[p[k], u] - b[p[k], v])
    # = (a[k,i] - a[k,j]) (b[p[k], v] - b[p[k], u])
    
    # Delta for c[i]:
    # Old c[i] = sum_k a[i,k] b[u, p[k]]
    # New c[i]' = sum_k a[i,k] b[v, p'[k]] where p'[j]=u, p'[i]=v, p'[k]=p[k] for k!=i,j.
    # c[i]' = a[i,i] b[v,v] + a[i,j] b[v,u] + sum_{k!=i,j} a[i,k] b[v, p[k]]
    # Old c[i] = a[i,i] b[u,u] + a[i,j] b[u,v] + sum_{k!=i,j} a[i,k] b[u, p[k]]
    # Note b[v,u] = b[u,v].
    # Delta c[i] = a[i,i](b[v,v]-b[u,u]) + a[i,j](b[u,v]-b[u,v]) + sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
    # = a[i,i](b[v,v]-b[u,u]) + sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
    
    # Similarly for c[j]:
    # Delta c[j] = a[j,j](b[u,u]-b[v,v]) + sum_{k!=i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
    
    # Total Delta Cost = Delta c[i] + Delta c[j] + sum_{k!=i,j} Delta c[k]
    
    # This is still O(n) per swap evaluation if we compute the sum over k.
    # With n=150, O(n) is 150 operations.
    # Vectorizing the sum over k:
    # diff_a_k = a[k, i] - a[k, j] for k != i,j
    # diff_b_k = b[p[k], v] - b[p[k], u] for k != i,j
    # sum_k diff_a_k * diff_b_k
    
    # We can precompute b_u = b[u, :] and b_v = b[v, :]
    # Then b[p[k], v] - b[p[k], u] is b_v[p[k]] - b_u[p[k]]? No.
    # b[p[k], v] is element (p[k], v) of b.
    # b[p[k], u] is element (p[k], u) of b.
    # So diff_b_k = b[p[k], v] - b[p[k], u].
    # This is vectorized as b[:, v][p] - b[:, u][p].
    
    def get_delta_ij(i, j, p, a, b, c):
        u = p[i]
        v = p[j]
        
        # Diagonal terms
        delta_diag_i = a[i, i] * (b[v, v] - b[u, u])
        delta_diag_j = a[j, j] * (b[u, u] - b[v, v])
        
        # Off-diagonal terms for c[i] and c[j] involving each other are 0 as shown above.
        # Wait, let's re-verify.
        # Old c[i] has a[i,j] b[u,v]. New c[i]' has a[i,j] b[v,u]. Same.
        # Old c[j] has a[j,i] b[v,u]. New c[j]' has a[j,i] b[u,v]. Same.
        # So cross terms cancel.
        
        # Sum over k != i, j
        # diff_a = a[:, i] - a[:, j]
        # diff_b = b[:, v][p] - b[:, u][p]
        # mask = np.ones(n, dtype=bool)
        # mask[i] = False; mask[j] = False
        # sum_k = np.dot(diff_a[mask], diff_b[mask])
        
        # Optimization: Include i and j in the sum and subtract their contributions?
        # For k=i: diff_a[i] = a[i,i] - a[i,j]. diff_b[i] = b[u, v] - b[u, u].
        # For k=j: diff_a[j] = a[j,i] - a[j,j]. diff_b[j] = b[v, u] - b[v, v].
        
        # Let's just compute the full dot product and subtract the i,j terms.
        diff_a = a[:, i] - a[:, j]
        # b[:, v] is column v of b. b[:, u] is column u of b.
        # We need b[p[k], v] which is b.T[v, p[k]]? No, b is symmetric usually, but let's be safe.
        # b[p[k], v] is the element at row p[k], col v.
        # This is b.T[v, p[k]] if b is not symmetric? No, b.T[v, p[k]] is b[p[k], v]. Yes.
        # So diff_b = b.T[v, p] - b.T[u, p]
        # Since b is symmetric, b.T = b. So diff_b = b[v, p] - b[u, p].
        
        diff_b = b[v, p] - b[u, p]
        
        total_dot = np.dot(diff_a, diff_b)
        
        # Subtract terms for k=i and k=j
        # k=i: diff_a[i] * diff_b[i]
        # diff_a[i] = a[i, i] - a[i, j]
        # diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        term_i = (a[i, i] - a[i, j]) * (b[v, u] - b[u, u])
        
        # k=j: diff_a[j] * diff_b[j]
        # diff_a[j] = a[j, i] - a[j, j]
        # diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        term_j = (a[j, i] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_k = total_dot - term_i - term_j
        
        delta = delta_diag_i + delta_diag_j + sum_k
        return delta

    # Local Search: Best Improvement 2-opt
    # To speed up, we can sample pairs or use random order.
    # With n=150, n(n-1)/2 = 11175 pairs.
    # Each delta is O(n) = 150 ops.
    # Total ops per full scan: 11175 * 150 ~ 1.7M ops.
    # In Python, this is too slow (seconds per scan).
    # We need to vectorize the delta calculation for ALL pairs at once?
    # That would be a 150x150 matrix of deltas.
    # Delta(i,j) depends on p[i], p[j].
    # This is complex to fully vectorize.
    
    # Alternative: Use random sampling of pairs in each iteration.
    # Or reduce n? No.
    
    # Let's try a simpler local search: 
    # Pick random pairs, compute delta, swap if negative.
    # This is stochastic local search.
    
    # To improve efficiency, we can batch the delta calculations.
    # But given the time limit, let's stick to a loop but optimize the inner part.
    
    # Actually, let's use the vectorized delta for a batch of random pairs.
    
    max_iterations = 10000
    iter_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Pick a batch of random pairs
        batch_size = 500
        if n < 10:
            batch_size = n*(n-1)//2
            
        # Generate random pairs
        # i < j
        i_idx = np.random.randint(0, n, size=batch_size)
        j_idx = np.random.randint(0, n, size=batch_size)
        # Ensure i != j
        mask = i_idx == j_idx
        i_idx[mask] = (i_idx[mask] + 1) % n
        # Ensure i < j for consistency? Not strictly necessary for swap, but avoids duplicates.
        # Let's just use i, j as is.
        
        # Compute deltas for these pairs
        # We can loop over the batch, which is 500 iterations of O(n) vectorized ops.
        # 500 * 150 = 75k ops. This is fast in NumPy.
        
        deltas = np.zeros(batch_size)
        for k in range(batch_size):
            i, j = i_idx[k], j_idx[k]
            if i == j:
                continue
            deltas[k] = get_delta_ij(i, j, p, a, b, c)
            
        # Find the best improvement
        min_delta = np.min(deltas)
        
        if min_delta < 0:
            # Find which pair it was
            best_k = np.argmin(deltas)
            i, j = i_idx[best_k], j_idx[best_k]
            
            # Apply swap
            u, v = p[i], p[j]
            p[i], p[j] = v, u
            cost += int(min_delta)
            improved = True
            
            # Update c
            # c[i] += delta_c_i
            # c[j] += delta_c_j
            # c[k] += delta_c_k for k != i,j
            # We can just recompute c? O(n^2) = 22500 ops.
            # Or update incrementally.
            # Let's recompute c to be safe and simple, as it's only O(n^2) and n=150.
            # Actually, recomputing c in a Python loop is O(n^2) Python ops, which is slow.
            # Let's vectorize c computation.
            # c[i] = dot(a[i], b[p[i], p])
            # c = np.array([np.dot(a[i], b[p[i], p]) for i in range(n)])
            # This is still a Python loop of 150 dot products.
            # 150 * 150 = 22500 multiplications. Fast in C, slow in Python loop overhead?
            # Let's try to vectorize the whole thing.
            # c = (a @ b) ? No.
            # c[i] = sum_j a[i,j] b[p[i], p[j]]
            # This is not a standard matrix multiplication because of the p indexing.
            # However, if we create a matrix B_p where B_p[i, j] = b[p[i], p[j]],
            # then c = a @ B_p? No, c[i] = sum_j a[i,j] B_p[i,j]. This is element-wise product then sum.
            # c = (a * B_p).sum(axis=1)
            # Constructing B_p:
            # B_p = b[np.ix_(p, p)]
            # This creates a 150x150 matrix. Fast in NumPy.
            
            B_p = b[np.ix_(p, p)]
            c = (a * B_p).sum(axis=1).astype(np.int64)
            
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
                
        iter_count += 1
        if iter_count > 5000: # Safety break
             break

    return best_p.tolist()

# EVOLVE-BLOCK-END
