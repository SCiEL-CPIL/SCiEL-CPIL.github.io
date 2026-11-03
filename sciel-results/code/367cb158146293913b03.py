"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.linalg import eigsh

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    # Ensure arrays are float64 for faster arithmetic and eigsh compatibility
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    def compute_cost(p):
        """Compute the QAP cost for permutation p."""
        # cost = sum_i sum_j a[i,j] * b[p[i], p[j]]
        # Vectorized: for each i, sum_j a[i,j] * b[p[i], j]
        # This is equivalent to: trace(A * B_perm) where B_perm is permuted
        # Efficient way: sum of dot products of rows
        total = 0.0
        for i in range(n):
            total += np.dot(a[i, :], b[p[i], :])
        return int(total)

    def compute_swap_delta(i, j, p):
        """
        Compute the change in cost when swapping locations of facilities i and j.
        p[i] and p[j] are the current locations.
        """
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0

        # Terms that change:
        # 1. Diagonal: a[i,i]*b[li,li] -> a[i,i]*b[lj,lj]
        #    a[j,j]*b[lj,lj] -> a[j,j]*b[li,li]
        # 2. Cross i-j: a[i,j]*b[li,lj] -> a[i,j]*b[lj,li]
        #    a[j,i]*b[lj,li] -> a[j,i]*b[li,lj]
        # 3. For all k not in {i,j}:
        #    a[i,k]*b[li,p[k]] -> a[i,k]*b[lj,p[k]]
        #    a[k,i]*b[p[k],li] -> a[k,i]*b[p[k],lj]
        #    a[j,k]*b[lj,p[k]] -> a[j,k]*b[li,p[k]]
        #    a[k,j]*b[p[k],lj] -> a[k,j]*b[p[k],li]

        delta = 0.0

        # Diagonal terms
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])

        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])

        # Terms with other facilities k (vectorized)
        # Create mask excluding i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]

        if len(ks) > 0:
            pk = p[ks]

            # a[i,k] * (b[lj,pk] - b[li,pk])
            delta += np.sum(a[i, ks] * (b[lj, pk] - b[li, pk]))
            # a[k,i] * (b[pk,lj] - b[pk,li])
            delta += np.sum(a[ks, i] * (b[pk, lj] - b[pk, li]))
            # a[j,k] * (b[li,pk] - b[lj,pk])
            delta += np.sum(a[j, ks] * (b[li, pk] - b[lj, pk]))
            # a[k,j] * (b[pk,li] - b[pk,lj])
            delta += np.sum(a[ks, j] * (b[pk, li] - b[pk, lj]))

        return int(delta)

    def local_search(p, max_time=None):
        """
        Steepest descent local search with 2-opt swaps.
        Returns the best permutation found.
        """
        if max_time is None:
            max_time = deadline

        cost = compute_cost(p)
        improved = True
        no_improve_count = 0

        while improved and time.time() < max_time:
            improved = False
            best_delta = 0
            best_pair = None

            # Find the best improving swap
            for i in range(n):
                if time.time() >= max_time:
                    break
                for j in range(i + 1, n):
                    if time.time() >= max_time:
                        break
                    delta = compute_swap_delta(i, j, p)
                    if delta < best_delta:
                        best_delta = delta
                        best_pair = (i, j)

            if best_pair is not None:
                i, j = best_pair
                p[i], p[j] = p[j], p[i]
                cost += best_delta
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 5: # Break early if stuck for a few iterations? 
                    # Actually steepest descent will find the best, so if no improvement, it's a local opt.
                    break

        return p, cost

    def perturb(p, strength=10):
        """
        Perturb the permutation by performing 'strength' random swaps.
        Used for ILS (Iterated Local Search).
        """
        p = p.copy()
        rng = np.random.RandomState(int(time.time() * 1000) % (2**32))
        for _ in range(strength):
            i, j = rng.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
        return p

    def spectral_initialization():
        """
        Generate initial permutations using spectral decomposition.
        We look for the top eigenvectors of A and B.
        """
        initializations = []
        
        # 1. Rank matching (baseline)
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p_rank = np.empty(n, dtype=np.int64)
        p_rank[heavy] = central
        initializations.append(p_rank)

        # 2. Spectral Matching
        try:
            # Get top k eigenvectors
            k = min(10, n)
            
            # Eigenvectors of A (Flow)
            # eigsh returns eigenvalues and eigenvectors
            # We want the ones corresponding to largest eigenvalues (for minimization of cost, 
            # we often align large flows with small distances, or vice versa depending on signs).
            # Let's just take the top k components.
            val_a, vec_a = eigsh(a, k=k, which='LM')
            # Sort by eigenvalue magnitude
            idx_a = np.argsort(np.abs(val_a))[::-1]
            vec_a = vec_a[:, idx_a]

            val_b, vec_b = eigsh(b, k=k, which='LM')
            idx_b = np.argsort(np.abs(val_b))[::-1]
            vec_b = vec_b[:, idx_b]

            # Project facilities and locations onto these eigenvectors
            # facility_scores[i, m] = vec_a[i, m]
            # location_scores[l, m] = vec_b[l, m]
            
            # We want to assign facility i to location l such that 
            # sum_m vec_a[i, m] * vec_b[l, m] is maximized? 
            # Or minimized?
            # Cost = sum a[i,j] b[p[i], p[j]].
            # If A and B are positive definite, large eigenvalues correspond to "dominant" structures.
            # Aligning the dominant structures usually yields a good heuristic.
            
            # Let's try aligning the top eigenvectors.
            # Score matrix S = vec_a @ vec_b.T
            # We want to find a permutation p that maximizes sum_i S[i, p[i]]?
            # This is an Assignment Problem.
            
            from scipy.optimize import linear_sum_assignment
            
            # Try different combinations of eigenvectors
            for m in range(min(3, k)): # Use top 3 principal components
                # Project onto m-th eigenvector
                fa = vec_a[:, m]
                fb = vec_b[:, m]
                
                # If we assume "high flow" should go to "central" location, 
                # and eigenvectors capture "centrality" or "activity", 
                # we might want to match high fa with high fb.
                
                # Let's create a cost matrix for the assignment problem.
                # We want to minimize the QAP cost. 
                # A common heuristic is to minimize the sum of products of deviations.
                
                # Let's try to match the signs/magnitudes.
                # If A and B have positive correlations, we match high with high.
                # If we just use linear_sum_assignment on -fa * fb, we maximize the sum of products.
                
                # Cost for assignment: - (fa[i] * fb[j])
                # We want to maximize sum fa[i] * fb[p[i]]
                # So cost matrix = - outer(fa, fb)
                
                cost_mat = -np.outer(fa, fb)
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                p_spec = np.empty(n, dtype=np.int64)
                p_spec[row_ind] = col_ind
                initializations.append(p_spec)
                
                # Also try inverted (match high with low)
                cost_mat_inv = np.outer(fa, fb)
                row_ind, col_ind = linear_sum_assignment(cost_mat_inv)
                p_spec_inv = np.empty(n, dtype=np.int64)
                p_spec_inv[row_ind] = col_ind
                initializations.append(p_spec_inv)

        except Exception:
            pass # Fallback to rank if spectral fails

        return initializations

    # Generate diverse initial solutions
    inits = spectral_initialization()
    
    best_p = None
    best_cost = float('inf')
    current_p = None
    current_cost = float('inf')
    
    rng = np.random.RandomState(42)

    # Process each initialization
    for p_init in inits:
        if time.time() > deadline - 10:
            break
            
        p_ls, cost_ls = local_search(p_init.copy(), deadline - 5)
        
        if cost_ls < best_cost:
            best_p = p_ls.copy()
            best_cost = cost_ls
            current_p = p_ls.copy()
            current_cost = cost_ls

    # If we have time, do ILS from the best solution found
    if best_p is not None and time.time() < deadline - 20:
        iterations = 0
        while time.time() < deadline - 5:
            iterations += 1
            # Perturb
            strength = 5 + int(10 * (1 - time.time() / deadline))
            strength = max(3, min(20, strength))
            perturbed_p = perturb(current_p, strength)

            # Local search from perturbed solution
            p_ls, cost_ls = local_search(perturbed_p, deadline - 5)

            # Accept if better
            if cost_ls < best_cost:
                best_p = p_ls.copy()
                best_cost = cost_ls
                current_p = p_ls.copy()
                current_cost = cost_ls
            else:
                # Sometimes accept worse solutions to escape local optima (Simulated Annealing style)
                # Simple random acceptance
                if rng.random() < 0.1:
                    current_p = p_ls.copy()
                    current_cost = cost_ls

    # Fallback if something went wrong
    if best_p is None:
        best_p = np.arange(n)

    return best_p.tolist()

# EVOLVE-BLOCK-END
