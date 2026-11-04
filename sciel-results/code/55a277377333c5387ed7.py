"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))
    
    def get_swap_delta(i, j, p):
        """Compute delta for swapping facilities i and j. Assumes symmetric a and b."""
        u = p[i]
        v = p[j]
        ai = a[i]
        aj = a[j]
        bv = b[v]
        bu = b[u]
        diff_a = ai - aj
        sum_all = np.sum(diff_a * (bv[p] - bu[p]))
        term_i = (ai[i] - aj[i]) * (bv[u] - bu[u])
        term_j = (ai[j] - aj[j]) * (bv[v] - bu[u])
        self_term = 2 * (ai[i] - aj[j]) * (bv[v] - bu[u])
        delta = 2 * (sum_all - term_i - term_j) + self_term
        return int(delta)
    
    def get_3cycle_delta(i, j, k, p):
        """Cost delta for 3-cycle: i->p[j], j->p[k], k->p[i]."""
        u, v, w = p[i], p[j], p[k]
        # New assignments: i gets v, j gets w, k gets u
        # Internal 3x3
        internal = (a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,w]-b[u,v]) + a[i,k]*(b[v,u]-b[u,w])
                  + a[j,i]*(b[w,u]-b[v,u]) + a[j,j]*(b[w,w]-b[v,v]) + a[j,k]*(b[w,u]-b[v,w])
                  + a[k,i]*(b[u,v]-b[w,u]) + a[k,j]*(b[u,w]-b[w,v]) + a[k,k]*(b[u,u]-b[w,w]))
        # Cross terms with all other facilities
        mask = np.ones(n, dtype=bool)
        mask[[i, j, k]] = False
        others = p[mask]
        # For facility x in {i,j,k} moving from old_loc to new_loc:
        # contribution = a[x, y] * (b[new_loc, p[y]] - b[old_loc, p[y]]) for y not in {i,j,k}
        # plus symmetric: a[y, x] * (b[p[y], new_loc] - b[p[y], old_loc])
        # Since a and b are symmetric, these are equal, so factor of 2.
        cross = 0
        # i: u -> v
        cross += 2 * np.dot(a[i, mask], b[v, others] - b[u, others])
        # j: v -> w
        cross += 2 * np.dot(a[j, mask], b[w, others] - b[v, others])
        # k: w -> u
        cross += 2 * np.dot(a[k, mask], b[u, others] - b[w, others])
        return int(internal + cross)
    
    def local_search_2swap(p, current_cost, max_iters=50):
        """Greedy 2-swap local search."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for i in range(n):
                for j in range(i + 1, n):
                    delta = get_swap_delta(i, j, p)
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost
    
    def tabu_3cycle_search(p, current_cost, tabu_tenure=7, max_iters=200, time_limit=None):
        """Tabu search with 3-cycle moves."""
        tabu_list = {}  # (i, j) -> iteration when tabu expires
        best_p = p.copy()
        best_cost = current_cost
        current_p = p.copy()
        current_c = current_cost
        rng = np.random.RandomState(12345)
        
        for it in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            best_delta = 0
            best_move = None
            best_aspiration = False
            
            # Sample a subset of 3-cycles for efficiency
            num_samples = min(500, n * (n-1) // 3)
            sampled = set()
            for _ in range(num_samples):
                idx = rng.choice(n, size=3, replace=False)
                key = tuple(sorted(idx))
                if key in sampled:
                    continue
                sampled.add(key)
                i, j, k = key
                
                # Check tabu: move is tabu if (i,j) or (j,k) or (i,k) was recently used
                is_tabu = False
                for pair in [(i,j), (j,k), (i,k)]:
                    if pair in tabu_list and tabu_list[pair] > it:
                        is_tabu = True
                        break
                
                delta = get_3cycle_delta(i, j, k, current_p)
                
                if is_tabu:
                    # Aspiration: accept if it improves best
                    if delta < best_cost - current_c:
                        if delta < best_delta or best_aspiration:
                            best_delta = delta
                            best_move = (i, j, k)
                            best_aspiration = True
                else:
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j, k)
                        best_aspiration = False
            
            if best_move is None:
                # Try random 3-cycle to escape
                idx = rng.choice(n, size=3, replace=False)
                i, j, k = idx
                delta = get_3cycle_delta(i, j, k, current_p)
                best_move = (i, j, k)
                best_delta = delta
                best_aspiration = True
            
            i, j, k = best_move
            # Apply 3-cycle: i->p[j], j->p[k], k->p[i]
            u, v, w = current_p[i], current_p[j], current_p[k]
            current_p[i] = v
            current_p[j] = w
            current_p[k] = u
            current_c += best_delta
            
            # Update tabu
            tabu_list[(i, j)] = it + tabu_tenure
            tabu_list[(j, k)] = it + tabu_tenure
            tabu_list[(i, k)] = it + tabu_tenure
            
            if current_c < best_cost:
                best_cost = current_c
                best_p = current_p.copy()
        
        return best_p, best_cost, current_p, current_c
    
    def hungarian_init():
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        cost_matrix = np.outer(flow_sums, dist_sums)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    def random_init():
        return np.random.permutation(n).astype(np.int64)
    
    def spectral_init():
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        fs_norm = flow_sums / (np.max(flow_sums) + 1e-10)
        ds_norm = dist_sums / (np.max(dist_sums) + 1e-10)
        cost_matrix = np.outer(fs_norm, ds_norm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    
    # Try multiple initializations
    candidates = [hungarian_init(), spectral_init(), random_init()]
    best_p = None
    best_cost = float('inf')
    for cand in candidates:
        c = calc_cost(cand)
        if c < best_cost:
            best_cost = c
            best_p = cand.copy()
    
    p = best_p.copy()
    current_cost = best_cost
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Phase 1: Polish initial solution with 2-swap local search
    p, current_cost = local_search_2swap(p, current_cost, max_iters=30)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # Phase 2: Tabu search with 3-cycles, multiple restarts
    rng = np.random.RandomState(42)
    tabu_tenure = 7
    
    while time.time() < deadline - 5:
        # Perturb the current best
        perturbed = best_p.copy()
        num_swaps = 4 + rng.randint(6)
        for _ in range(num_swaps):
            i = rng.randint(n)
            j = rng.randint(n)
            if i == j:
                continue
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        # Quick 2-swap polish on perturbed
        perturbed, pert_cost = local_search_2swap(perturbed, calc_cost(perturbed), max_iters=10)
        
        # Tabu search from perturbed
        remaining = deadline - 5 - time.time()
        if remaining < 5:
            break
        
        # Scale max_iters based on remaining time
        max_iters = int(min(300, max(20, remaining * 50)))
        
        tabu_p, tabu_cost, _, _ = tabu_3cycle_search(
            perturbed, pert_cost, 
            tabu_tenure=tabu_tenure,
            max_iters=max_iters,
            time_limit=deadline - 3
        )
        
        if tabu_cost < best_cost:
            best_cost = tabu_cost
            best_p = tabu_p.copy()
        
        # Occasionally adjust tabu tenure
        if rng.random() < 0.2:
            tabu_tenure = max(3, min(15, tabu_tenure + rng.randint(-2, 3)))
    
    # Final polish
    if time.time() < deadline - 2:
        best_p, final_cost = local_search_2swap(best_p, best_cost, max_iters=10)
        if final_cost < best_cost:
            best_cost = final_cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
