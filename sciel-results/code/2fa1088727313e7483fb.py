"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Pre-compute some things if needed, but a and b are constant.
    # We will use vectorized delta calculation.
    
    def compute_cost_fast(p):
        """Compute cost using vectorized matrix multiplication."""
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    def compute_swap_delta(i, j, p, a_rows, b_cols, b_rows, a_cols):
        """Compute cost delta when swapping facilities at indices i and j in permutation p."""
        li = p[i]
        lj = p[j]
        
        if li == lj:
            return 0
        
        # Diagonal terms
        delta = a[i, i] * (b[lj, lj] - b[li, li])
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        
        # Cross terms between i and j
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        
        # Terms with other facilities k
        # For performance, we precompute the rows/cols we need
        # a[i, ks] where ks are all k != i, j
        # This is a[i, :] with a[i, i] and a[i, j] removed
        # Similarly for other terms
        
        # We'll use slicing to avoid creating masks
        # Get all indices except i and j
        if i < j:
            ks_list = list(range(i)) + list(range(i+1, j)) + list(range(j+1, n))
        else:
            ks_list = list(range(j)) + list(range(j+1, i)) + list(range(i+1, n))
        
        ks = np.array(ks_list, dtype=np.int64)
        pk = p[ks]
        
        if len(ks) > 0:
            # Vectorized sums
            # a[i, ks] * (b[lj, pk] - b[li, pk])
            delta += np.dot(a[i, ks], b[lj, pk] - b[li, pk])
            # a[ks, i] * (b[pk, lj] - b[pk, li])
            delta += np.dot(a[ks, i], b[pk, lj] - b[pk, li])
            # a[j, ks] * (b[li, pk] - b[lj, pk])
            delta += np.dot(a[j, ks], b[li, pk] - b[lj, pk])
            # a[ks, j] * (b[pk, li] - b[pk, lj])
            delta += np.dot(a[ks, j], b[pk, li] - b[pk, lj])
            
        return delta

    def fast_local_search(p, current_cost, end_time):
        """
        Perform First Improvement 2-swap Local Search.
        Randomizes the order of swaps to avoid bias.
        Returns best_p, best_cost found during the search.
        """
        curr_p = p.copy()
        curr_cost = current_cost
        best_p = p.copy()
        best_cost = current_cost
        
        # Pre-generate random permutations of indices for shuffling
        # We will shuffle the list of pairs (i, j)
        rng = np.random.RandomState(12345)
        
        while time.time() < end_time:
            improved = False
            # Create list of all pairs (i, j) with i < j
            # To be efficient, we can just iterate i, j and use a random start offset or shuffle i
            # Shuffling all pairs is O(n^2) memory/time, which is small for n=150.
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            
            for i, j in pairs:
                if time.time() > end_time:
                    break
                
                delta = compute_swap_delta(i, j, curr_p, None, None, None, None)
                
                if delta < 0:
                    # Accept move
                    curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                    curr_cost += delta
                    improved = True
                    
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_p = curr_p.copy()
            
            if not improved:
                break
                
        return best_p, best_cost

    def simulated_annealing(p, best_cost, end_time):
        """
        Perform Simulated Annealing to escape local minima.
        """
        curr_p = p.copy()
        curr_cost = best_cost
        best_p = p.copy()
        best_cost_sa = best_cost
        
        rng = np.random.RandomState(42)
        
        # Estimate initial temperature
        deltas = []
        for _ in range(20):
            i, j = rng.choice(n, 2, replace=False)
            d = compute_swap_delta(i, j, curr_p, None, None, None, None)
            deltas.append(abs(d))
        avg_delta = np.mean(deltas) if deltas else 1.0
        if avg_delta == 0:
            avg_delta = 1.0
            
        T = avg_delta * 15.0
        cooling_rate = 0.9995
        min_T = 1e-5
        
        while time.time() < end_time:
            if T < min_T:
                break
                
            # Generate random move
            i, j = rng.choice(n, 2, replace=False)
            
            delta = compute_swap_delta(i, j, curr_p, None, None, None, None)
            
            # Acceptance criterion
            if delta < 0:
                accept = True
            else:
                # Avoid overflow in exp
                if T > 0:
                    prob = np.exp(-delta / T)
                else:
                    prob = 0.0
                accept = rng.random_sample() < prob
            
            if accept:
                curr_p[i], curr_p[j] = curr_p[j], curr_p[i]
                curr_cost += delta
                
                if curr_cost < best_cost_sa:
                    best_cost_sa = curr_cost
                    best_p = curr_p.copy()
            
            T *= cooling_rate
        
        return best_p, best_cost_sa

    # 1. Spectral + Hungarian Initialization
    # Spectral initialization: use eigenvectors of a and b to get a good starting point
    def spectral_init():
        """Use spectral decomposition to get an initial assignment."""
        try:
            # Compute eigenvectors of a (symmetrize first if needed)
            a_sym = (a + a.T) / 2.0
            b_sym = (b + b.T) / 2.0
            w_a, v_a = np.linalg.eigh(a_sym)
            w_b, v_b = np.linalg.eigh(b_sym)
            # Use the top few eigenvectors
            k = min(10, n)
            Va = v_a[:, :k]
            Vb = v_b[:, :k]
            # Assign based on correlation
            C_spec = Va @ Vb.T
            row_ind, col_ind = linear_sum_assignment(-C_spec)
            p_spec = np.empty(n, dtype=np.int64)
            p_spec[row_ind] = col_ind
            return p_spec
        except:
            return None
    
    p_spec = spectral_init()
    
    # Hungarian on the linearized cost
    C = a @ b.T
    row_ind, col_ind = linear_sum_assignment(-C)
    p_hung = np.empty(n, dtype=np.int64)
    p_hung[row_ind] = col_ind
    
    # Choose the better initial solution
    cost_hung = compute_cost_fast(p_hung)
    if p_spec is not None:
        cost_spec = compute_cost_fast(p_spec)
        if cost_spec < cost_hung:
            best_p = p_spec.copy()
            best_cost = cost_spec
        else:
            best_p = p_hung.copy()
            best_cost = cost_hung
    else:
        best_p = p_hung.copy()
        best_cost = cost_hung
    
    # 2. Polish Hungarian solution with Local Search
    end_time_lk = deadline - 20.0
    p_lk, cost_lk = fast_local_search(best_p, best_cost, end_time_lk)
    if cost_lk < best_cost:
        best_p = p_lk
        best_cost = cost_lk
        
    # 3. Try Random Initialization + Local Search
    if time.time() < deadline - 15.0:
        rng_init = np.random.RandomState(123)
        p_rand = rng_init.permutation(n)
        cost_rand = compute_cost_fast(p_rand)
        
        end_time_lk2 = deadline - 10.0
        p_lk2, cost_lk2 = fast_local_search(p_rand, cost_rand, end_time_lk2)
        if cost_lk2 < best_cost:
            best_p = p_lk2
            best_cost = cost_lk2
            
    # 4. Perturb Best and Local Search
    if time.time() < deadline - 5.0:
        rng_perturb = np.random.RandomState(456)
        p_pert = best_p.copy()
        # Swap 5 random pairs to perturb
        for _ in range(5):
            i, j = rng_perturb.choice(n, 2, replace=False)
            p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
        cost_pert = compute_cost_fast(p_pert)
        end_time_lk3 = deadline - 2.0
        p_lk3, cost_lk3 = fast_local_search(p_pert, cost_pert, end_time_lk3)
        if cost_lk3 < best_cost:
            best_p = p_lk3
            best_cost = cost_lk3

    # 5. Final SA Polish if time remains
    if time.time() < deadline - 1.0:
        p_sa, cost_sa = simulated_annealing(best_p, best_cost, deadline - 0.5)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa

    return best_p.tolist()
# EVOLVE-BLOCK-END
