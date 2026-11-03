"""QAP solver for tho150 using vectorized local search with efficient swap evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    """Solve QAP: assign facilities to locations to minimize sum a[i,j]*b[p[i],p[j]].
    
    Args:
        a: flow matrix (n x n) int64
        b: distance matrix (n x n) int64
    Returns:
        p: permutation list where facility i goes to location p[i]
    """
    n = len(a)
    deadline = time.time() + 112  # Safety margin before 150s kill
    
    # Convert to float for faster arithmetic
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    
    # Precompute row sums for initial heuristics
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    def compute_cost(p):
        """Compute exact cost of permutation p."""
        M = b[np.ix_(p, p)]
        return int(np.sum(a * M))
    
    def compute_all_deltas(p, M):
        """Compute all pairwise swap deltas in a vectorized manner.
        
        Returns a matrix D where D[i,j] = cost change when swapping positions i and j.
        Uses the identity:
        delta(i,j) = (row_ai - row_aj) @ (M[j,:] - b[pi,p]) 
                   + (col_ai - col_aj) @ (M[:,j] - b[p,pi])
                   + (a[i,i] - a[j,j]) * (b[pj,pj] - b[pi,pi])
        where pi = p[i], pj = p[j], and b[pi,p] = b_f[pi, p], etc.
        
        We can rewrite this more compactly. Let:
        R[i] = a_f[i, :] (row i of a)
        C[i] = a_f[:, i] (col i of a)
        B_pi = b_f[p[i], :] (row p[i] of b)
        B_pi_col = b_f[:, p[i]] (col p[i] of b)
        
        For swap (i,j):
        delta = R[i]@(M[j,:] - B_pi_row) + C[i]@(M[:,j] - B_pi_col)
              + R[j]@(B_pi_row - M[j,:]) + C[j]@(B_pi_col - M[:,j])
              + (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
              
        = (R[i]-R[j])@(M[j,:]-B_pi_row) + (C[i]-C[j])@(M[:,j]-B_pi_col)
              + (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        
        But this still needs per-pair computation. Let's use a different approach:
        
        Actually, let's precompute for each position i:
        row_contrib[i] = R[i] @ M  (shape: n x n, row i is R[i] @ M)
        col_contrib[i] = C[i] @ M  (shape: n x n, row i is C[i] @ M)
        
        Wait, that's O(n^3). Instead, let's compute:
        
        For the row part: R[i] @ (M[j,:] - b_f[p[i], p])
        = R[i] @ M[j,:] - R[i] @ b_f[p[i], p]
        
        R[i] @ M[j,:] = sum_k R[i,k] * M[j,k] = sum_k a[i,k] * b[p[j], p[k]]
        R[i] @ b_f[p[i], p] = sum_k a[i,k] * b[p[i], p[k]]
        
        Let's define:
        A_M[i,j] = sum_k a[i,k] * b[p[j], p[k]] = a_f[i,:] @ b_f[p[j], p]
        This is (a_f @ b_f[np.ix_(p,p)]) but transposed... 
        Actually A_M = a_f @ b_f[np.ix_(p,p)] doesn't work directly.
        
        Let M = b_f[np.ix_(p,p)] so M[j,k] = b[p[j], p[k]]
        Then A_M[i,j] = sum_k a[i,k] * M[j,k] = (a_f @ M.T)[i,j]
        
        Similarly for col part:
        C[i] @ M[:,j] = sum_k a[k,i] * M[k,j] = (M.T @ a_f)[i,j]... 
        Wait: M[k,j] = b[p[k], p[j]], so sum_k a[k,i]*b[p[k],p[j]]
        = (a_f.T @ b_f[np.ix_(p,p)] @ ... ) hmm.
        
        Let me just compute:
        A_row[i,j] = a_f[i,:] @ M[j,:]  -- this is a_f[i,:] dot M[j,:]
        We can compute the full matrix: A_row = a_f @ M.T  (since (a_f @ M.T)[i,j] = sum_k a_f[i,k]*M.T[k,j] = sum_k a_f[i,k]*M[j,k])
        
        A_col[i,j] = a_f[:,i] @ M[:,j] = sum_k a_f[k,i]*M[k,j]
        = (M @ a_f)[j,i]... let's see: (M @ a_f)[j,i] = sum_k M[j,k]*a_f[k,i] = sum_k b[p[j],p[k]]*a[k,i]
        That's not quite right. We want sum_k a[k,i]*b[p[k],p[j]] = sum_k a_f[k,i]*M[k,j]
        = (a_f.T @ M)[i,j]... (a_f.T @ M)[i,j] = sum_k a_f.T[i,k]*M[k,j] = sum_k a_f[k,i]*M[k,j]. Yes!
        
        So:
        A_row = a_f @ M.T   # A_row[i,j] = sum_k a[i,k]*b[p[j],p[k]]
        A_col = a_f.T @ M   # A_col[i,j] = sum_k a[k,i]*b[p[k],p[j]]
        
        Now for swap (i,j) with pi=p[i], pj=p[j]:
        B_pi_row = b_f[pi, p]  # length n
        B_pi_col = b_f[p, pi]  # length n
        
        Row part for i: R[i]@(M[j,:] - B_pi_row) = A_row[i,j] - R[i]@B_pi_row
        R[i]@B_pi_row = sum_k a[i,k]*b[pi,p[k]] = A_row[i, idx_of_pi] where idx_of_pi = position where p[idx]=pi, i.e., idx_of_pi = i itself!
        So R[i]@B_pi_row = A_row[i,i]
        
        Similarly: R[i]@B_pi_col = sum_k a[i,k]*b[p[k],pi] = A_col[i, idx_of_pi] = A_col[i,i]
        
        Wait, B_pi_col[k] = b[p[k], pi], so R[i]@B_pi_col = sum_k a[i,k]*b[p[k],pi]
        And A_col[i,j] = sum_k a[k,i]*b[p[k],p[j]], which is different.
        
        Let me redo. Let me define:
        For a given location loc, define:
        b_loc_row[p] = b[loc, p[k]] for all k  -- this is b_f[loc, p]
        b_loc_col[p] = b[p[k], loc] for all k  -- this is b_f[p, loc]
        
        For swap (i,j):
        delta_row_i = sum_k a[i,k]*(b[pj,p[k]] - b[pi,p[k]])
        = sum_k a[i,k]*b[pj,p[k]] - sum_k a[i,k]*b[pi,p[k]]
        = A_row[i,j] - A_row[i,i]
        
        delta_col_i = sum_k a[k,i]*(b[p[k],pj] - b[p[k],pi])
        = sum_k a[k,i]*b[p[k],pj] - sum_k a[k,i]*b[p[k],pi]
        = A_col[i,j] - A_col[i,i]
        
        Similarly for j:
        delta_row_j = sum_k a[j,k]*(b[pi,p[k]] - b[pj,p[k]])
        = A_row[j,i] - A_row[j,j]
        
        delta_col_j = sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
        = A_col[j,i] - A_col[j,j]
        
        Diagonal: (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        
        So: delta(i,j) = (A_row[i,j] - A_row[i,i]) + (A_col[i,j] - A_col[i,i])
                       + (A_row[j,i] - A_row[j,j]) + (A_col[j,i] - A_col[j,j])
                       + (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        
        This is fully vectorizable!
        """
        # M = b_f[np.ix_(p, p)] already passed in
        A_row = a_f @ M.T   # A_row[i,j] = sum_k a[i,k]*b[p[j],p[k]]
        A_col = a_f.T @ M   # A_col[i,j] = sum_k a[k,i]*b[p[k],p[j]]
        
        # Diagonal terms
        a_diag = a_f.diagonal()  # a[i,i]
        b_diag_p = b_f[np.ix_(p, p)].diagonal()  # b[p[i],p[i]]
        
        # delta(i,j) = (A_row[i,j] - A_row[i,i]) + (A_col[i,j] - A_col[i,i])
        #            + (A_row[j,i] - A_row[j,j]) + (A_col[j,i] - A_col[j,j])
        #            + (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        
        # Let's compute the matrix of deltas
        # Part 1: A_row[i,j] - A_row[i,i] for all i,j
        # A_row[i,i] is the diagonal of A_row
        diag_row = np.diag(A_row)
        diag_col = np.diag(A_col)
        
        # D1[i,j] = A_row[i,j] - diag_row[i]
        D1 = A_row - diag_row[:, np.newaxis]
        
        # D2[i,j] = A_col[i,j] - diag_col[i]
        D2 = A_col - diag_col[:, np.newaxis]
        
        # D3[i,j] = A_row[j,i] - diag_row[j] = D1.T adjusted... 
        # D1.T[i,j] = D1[j,i] = A_row[j,i] - diag_row[j]
        D3 = D1.T
        
        # D4[i,j] = A_col[j,i] - diag_col[j]
        D4 = D2.T
        
        # D5[i,j] = (a[i,i]-a[j,j])*(b[pj,pj]-b[pi,pi])
        # = a[i,i]*b[pj,pj] - a[i,i]*b[pi,pi] - a[j,j]*b[pj,pj] + a[j,j]*b[pi,pi]
        # Let's compute as outer product differences
        # = (a_diag[:,np.newaxis]*b_diag_p[np.newaxis,:] + a_diag[np.newaxis,:]*b_diag_p[:,np.newaxis]) 
        #   - (a_diag[:,np.newaxis]*b_diag_p[:,np.newaxis] + a_diag[np.newaxis,:]*b_diag_p[np.newaxis,:])
        # Hmm, let's just do it directly:
        # D5[i,j] = (a_diag[i] - a_diag[j]) * (b_diag_p[j] - b_diag_p[i])
        
        D5 = (a_diag[:, np.newaxis] - a_diag[np.newaxis, :]) * (b_diag_p[np.newaxis, :] - b_diag_p[:, np.newaxis])
        
        D = D1 + D2 + D3 + D4 + D5
        
        # Set diagonal to 0 (no self-swap)
        np.fill_diagonal(D, 0)
        
        return D
    
    def local_search(p, deadline):
        """Steepest descent local search. Returns improved permutation and cost."""
        cost = compute_cost(p)
        M = b_f[np.ix_(p, p)]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_delta = 0
            best_i, best_j = -1, -1
            
            # Compute all pairwise deltas at once
            D = compute_all_deltas(p, M)
            
            # Find best (i,j) with i < j
            # Mask upper triangle
            triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            D_masked = np.where(triu_mask, D, np.inf)
            
            flat_idx = np.argmin(D_masked)
            best_i, best_j = divmod(flat_idx, n)
            best_delta = D[best_i, best_j]
            
            if best_delta < 0:
                # Perform swap
                p[best_i], p[best_j] = p[best_j], p[best_i]
                cost += int(round(best_delta))
                # Update M
                M = b_f[np.ix_(p, p)]
                improved = True
        
        return p, cost
    
    def make_initial_solution(method, rng):
        """Create an initial permutation using different strategies."""
        if method == 0:
            # Rank matching: heavy flow -> central location
            heavy = np.argsort(-a_row_sums, kind='stable')
            central = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 1:
            # Reverse rank
            heavy = np.argsort(a_row_sums, kind='stable')
            central = np.argsort(-b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        elif method == 2:
            # Random
            p = rng.permutation(n).astype(np.int64)
        elif method == 3:
            # Sort by a_row_sums / b_row_sums ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(b_row_sums > 0, a_row_sums / np.maximum(b_row_sums, 1), 0)
            fac_order = np.argsort(-ratio, kind='stable')
            loc_order = np.argsort(b_row_sums, kind='stable')
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
        elif method == 4:
            # Spectral-like: use eigenvectors of a and b
            try:
                _, eig_a = np.linalg.eigh(a_f)
                _, eig_b = np.linalg.eigh(b_f)
                # Use last eigenvector (largest eigenvalue) for ordering
                order_a = np.argsort(-np.abs(eig_a[:, -1]))
                order_b = np.argsort(-np.abs(eig_b[:, -1]))
                p = np.empty(n, dtype=np.int64)
                p[order_a] = order_b
            except:
                p = rng.permutation(n).astype(np.int64)
        else:
            p = rng.permutation(n).astype(np.int64)
        return p
    
    # Multiple initializations with ILS
    best_p = None
    best_cost = np.inf
    
    rng = np.random.default_rng(42)
    
    # Try several different initial solutions
    n_init = 10
    for init_idx in range(n_init):
        if time.time() > deadline:
            break
        
        method = init_idx % 5
        p = make_initial_solution(method, rng)
        
        # Local search
        p, cost = local_search(p, deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
        
        # ILS: perturb and search again if time permits
        if time.time() < deadline - 25:
            for ils_iter in range(8):
                if time.time() > deadline:
                    break
                # Perturb: random swaps with adaptive size
                n_swaps = rng.integers(10, 35)
                p_perturbed = p.copy()
                for _ in range(n_swaps):
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        p_perturbed[i], p_perturbed[j] = p_perturbed[j], p_perturbed[i]
                
                p_perturbed, cost_p = local_search(p_perturbed, deadline)
                
                if cost_p < best_cost:
                    best_cost = cost_p
                    best_p = p_perturbed.copy()
                elif cost_p < cost:
                    p = p_perturbed.copy()
                    cost = cost_p
    
    # Final verification
    if best_p is None:
        best_p = make_initial_solution(0, rng)
    
    final_cost = compute_cost(best_p)
    if final_cost < best_cost:
        best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
