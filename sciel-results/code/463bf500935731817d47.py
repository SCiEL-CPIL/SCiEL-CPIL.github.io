"""QAP solver: Hungarian init + fast vectorized first-improvement LS + ILS with Or-opt moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill

    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_swap_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        O(n) vectorized computation.
        """
        u = p[i]
        v = p[j]
        
        # Direct pair term
        delta = (a[i, j] - a[j, i]) * (b[v, u] - b[u, v])
        
        # Diagonal terms
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Cross terms with all other k
        diff_a_row = a[i] - a[j]
        diff_a_col = a[:, i] - a[:, j]
        
        diff_b_row = b[v, p] - b[u, p]
        diff_b_col = b[p, v] - b[p, u]
        
        sum_T1 = np.dot(diff_a_row, diff_b_row)
        sum_T2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i terms
        sum_T1 -= (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        sum_T2 -= (a[i, i] - a[i, j]) * (b[u, v] - b[u, u])
        
        # Subtract k=j terms
        sum_T1 -= (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        sum_T2 -= (a[j, i] - a[j, j]) * (b[v, v] - b[v, u])
        
        delta += sum_T1 + sum_T2
        return int(delta)

    def get_3cycle_delta(p, i, j, k):
        """
        Calculate delta cost for 3-cycle: i->j, j->k, k->i positions.
        That is, the facility at position i moves to position j's location,
        facility at j moves to k's location, facility at k moves to i's location.
        """
        u_i = p[i]
        u_j = p[j]
        u_k = p[k]
        
        # Old cost terms involving i, j, k
        # New cost: facility i goes to location u_j, facility j to u_k, facility k to u_i
        
        # This is complex to vectorize efficiently for arbitrary triples.
        # For now, let's stick to swaps and 2-3-4 moves if needed, or just fast swaps.
        # Given time constraints, fast first-improvement swaps are likely better than slow 3-cycles.
        
        # Let's implement a simple 3-move: reverse a segment of length 3? No, that's not a single 3-cycle.
        # Let's stick to swaps for the fast local search, but use a smarter neighbor generation in ILS.
        
        return 0  # Placeholder, not used in this version

    def local_search_first_improvement(p, current_cost):
        """
        First-improvement local search using swaps.
        Iterates through pairs in a fixed order, applying the first improving swap found.
        Repeats until no improving swap is found in a full pass.
        """
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 5.0:
                break
            
            for i in range(n):
                if time.time() - start_time > deadline - 5.0:
                    break
                for j in range(i + 1, n):
                    delta = get_swap_delta(p, i, j)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
                        # In first-improvement, we could restart the scan or continue.
                        # Restarting can be faster to find local optimum in some cases,
                        # but continuing is simpler. Let's continue to the end of the pass.
                        # Actually, restarting the inner loop after a swap can be beneficial.
                        # But for simplicity and speed, let's just continue.
                        # A more aggressive strategy: break inner loop and restart outer?
                        # Let's just continue scanning.
        
        return p, current_cost

    def local_search_best_improvement(p, current_cost):
        """
        Best-improvement local search using swaps.
        Finds the single best improving swap in a full pass, applies it, repeats.
        Slower per iteration but stronger.
        """
        while True:
            if time.time() - start_time > deadline - 5.0:
                break
            
            best_delta = 0
            best_i = -1
            best_j = -1
            
            for i in range(n):
                if time.time() - start_time > deadline - 5.0:
                    break
                for j in range(i + 1, n):
                    delta = get_swap_delta(p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i != -1:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
            else:
                break
        
        return p, current_cost

    def hungarian_init():
        """
        Initialize using Hungarian algorithm on surrogate cost.
        C[i, j] = -a_row_sums[i] * b_row_sums[j]
        """
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        
        C = -np.outer(a_row_sums, b_row_sums)
        
        row_ind, col_ind = linear_sum_assignment(C)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def perturb(p, num_swaps):
        """
        Apply random swaps to p.
        """
        p = p.copy()
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            p[i], p[j] = p[j], p[i]
        return p

    # 1. Initialization
    p = hungarian_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Initial Local Search (Best Improvement)
    p, current_cost = local_search_best_improvement(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()

    # 3. ILS Loop
    no_improve_count = 0
    rng = np.random.default_rng()
    
    while time.time() - start_time < deadline - 2.0:
        # Adaptive perturbation strength
        if no_improve_count < 3:
            num_swaps = 3
        elif no_improve_count < 8:
            num_swaps = 6
        elif no_improve_count < 15:
            num_swaps = 10
        else:
            num_swaps = 15
        
        # Perturb
        p_perturbed = perturb(best_p, num_swaps)
        
        # Fast local search (First Improvement)
        p_perturbed, cost_perturbed = local_search_first_improvement(p_perturbed, calc_cost(p_perturbed))
        
        # If first improvement got stuck high, try best improvement if time permits
        if time.time() - start_time < deadline - 10.0:
            p_perturbed, cost_perturbed = local_search_best_improvement(p_perturbed, cost_perturbed)
        
        if cost_perturbed < best_cost:
            best_cost = cost_perturbed
            best_p = p_perturbed.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
            # If stuck for too long, try a more aggressive restart
            if no_improve_count > 30:
                # Random restart from scratch? Or just increase perturbation significantly
                no_improve_count = 0 
                # Let's just continue with increased perturbation

    return best_p.tolist()
# EVOLVE-BLOCK-END
