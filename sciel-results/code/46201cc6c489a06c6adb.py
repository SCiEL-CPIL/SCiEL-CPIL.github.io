"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """
    Hybrid GRASP + Simulated Annealing for QAP.
    Uses permuted-B caching for fast swap deltas and Hungarian init.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    # Cache: b_perm = b[p, p] so that b_perm[i,j] = b[p[i], p[j]]
    b_perm = None

    def refresh_b_perm(p):
        nonlocal b_perm
        b_perm = b[np.ix_(p, p)].copy()

    def compute_cost(p):
        """Compute the QAP cost for a given permutation p."""
        B_p = b[np.ix_(p, p)]
        return int(np.sum(a * B_p))

    def swap_delta_cached(p, i, j):
        """
        Compute delta for swapping positions i and j in p, using cached b_perm.
        b_perm[i,j] == b[p[i], p[j]]
        """
        li = p[i]
        lj = p[j]

        # Self terms
        delta = a[i, i] * (b_perm[j, j] - b_perm[i, i]) + a[j, j] * (b_perm[i, i] - b_perm[j, j])

        # Cross terms (i,j) and (j,i)
        delta += a[i, j] * (b_perm[j, i] - b_perm[i, j])
        delta += a[j, i] * (b_perm[i, j] - b_perm[j, i])

        # Terms with k != i, j: use b_perm rows/cols
        # sum_k a[k,i] * (b_perm[k,j] - b_perm[k,i])
        # sum_k a[i,k] * (b_perm[j,k] - b_perm[i,k])
        # sum_k a[k,j] * (b_perm[k,i] - b_perm[k,j])
        # sum_k a[j,k] * (b_perm[i,k] - b_perm[j,k])
        S1 = np.dot(a[:, i], b_perm[:, j]) - np.dot(a[:, i], b_perm[:, i])
        S2 = np.dot(a[i, :], b_perm[j, :]) - np.dot(a[i, :], b_perm[i, :])
        S3 = np.dot(a[:, j], b_perm[:, i]) - np.dot(a[:, j], b_perm[:, j])
        S4 = np.dot(a[j, :], b_perm[i, :]) - np.dot(a[j, :], b_perm[j, :])

        # Subtract the k=i and k=j contributions already counted in self/cross
        S1_sub = a[i, i] * (b_perm[i, j] - b_perm[i, i]) + a[j, i] * (b_perm[j, i] - b_perm[j, j])
        S2_sub = a[i, i] * (b_perm[j, i] - b_perm[i, i]) + a[i, j] * (b_perm[j, j] - b_perm[i, j])
        S3_sub = a[i, j] * (b_perm[i, i] - b_perm[i, j]) + a[j, j] * (b_perm[j, i] - b_perm[j, j])
        S4_sub = a[j, i] * (b_perm[i, i] - b_perm[j, i]) + a[j, j] * (b_perm[i, j] - b_perm[j, j])

        delta += (S1 - S1_sub) + (S2 - S2_sub) + (S3 - S3_sub) + (S4 - S4_sub)
        return delta

    def do_swap(p, i, j):
        """Swap positions i,j in p and update b_perm incrementally."""
        nonlocal b_perm
        li = p[i]
        lj = p[j]
        p[i], p[j] = p[j], p[i]
        # Swap rows i and j in b_perm
        b_perm[[i, j], :] = b_perm[[j, i], :]
        # Swap cols i and j in b_perm
        b_perm[:, [i, j]] = b_perm[:, [j, i]]

    def local_search(p, current_cost, dl):
        """2-opt local search using cached b_perm."""
        p = p.copy()
        refresh_b_perm(p)
        improved = True
        while improved and time.time() < dl:
            improved = False
            for i in range(n):
                if time.time() >= dl:
                    break
                for j in range(i + 1, n):
                    delta = swap_delta_cached(p, i, j)
                    if delta < 0:
                        do_swap(p, i, j)
                        current_cost += delta
                        improved = True
        return p, current_cost

    def grasp_construct(alpha=0.3, dl=None):
        """
        Greedy Randomized construction (Vectorized).
        """
        if dl is None:
            dl = deadline
        n_facilities = list(range(n))
        n_locations = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        placed = []
        
        random.shuffle(n_facilities)

        for step in range(n):
            if time.time() >= dl:
                break
            idx = random.randint(0, len(n_facilities) - 1)
            i = n_facilities.pop(idx)

            if not placed:
                costs = a[i, i] * b[np.ix_(n_locations, n_locations)].diagonal()
            else:
                placed_arr = np.array(placed)
                p_placed = p[placed_arr]
                costs = a[i, i] * b[np.ix_(n_locations, n_locations)].diagonal()
                b_l_placed = b[np.ix_(n_locations, p_placed)]
                costs += b_l_placed @ a[i, placed_arr]
                b_placed_l = b[np.ix_(p_placed, n_locations)]
                costs += a[placed_arr, i] @ b_placed_l

            min_cost = np.min(costs)
            max_cost = np.max(costs)
            threshold = min_cost + alpha * (max_cost - min_cost)
            candidates = [loc_idx for loc_idx in range(len(n_locations)) if costs[loc_idx] <= threshold]

            chosen_loc_idx = random.choice(candidates)
            l = n_locations.pop(chosen_loc_idx)
            p[i] = l
            placed.append(i)

        return p

    def greedy_init():
        """
        Greedy construction: always pick the best location.
        """
        n_facilities = list(range(n))
        n_locations = list(range(n))
        p = np.full(n, -1, dtype=np.int64)
        placed = []
        
        random.shuffle(n_facilities)

        for i in n_facilities:
            if not placed:
                costs = a[i, i] * b[np.ix_(n_locations, n_locations)].diagonal()
            else:
                placed_arr = np.array(placed)
                p_placed = p[placed_arr]
                costs = a[i, i] * b[np.ix_(n_locations, n_locations)].diagonal()
                b_l_placed = b[np.ix_(n_locations, p_placed)]
                costs += b_l_placed @ a[i, placed_arr]
                b_placed_l = b[np.ix_(p_placed, n_locations)]
                costs += a[placed_arr, i] @ b_placed_l
            
            best_loc_idx = np.argmin(costs)
            l = n_locations.pop(best_loc_idx)
            p[i] = l
            placed.append(i)
            
        return p

    def simulated_annealing(p_init, current_cost, dl, t_start, t_end, cooling):
        """
        Simulated Annealing with 2-swap moves using cached b_perm.
        """
        p = p_init.copy()
        refresh_b_perm(p)
        cost = current_cost
        t = t_start

        while t > t_end and time.time() < dl:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue

            delta = swap_delta_cached(p, i, j)

            if delta < 0:
                do_swap(p, i, j)
                cost += delta
            else:
                if random.random() < np.exp(-delta / t):
                    do_swap(p, i, j)
                    cost += delta

            t *= cooling

        return p, cost

    def hungarian_init():
        """
        Use Hungarian on a linearized cost matrix to get an initial assignment.
        For QAP: minimize sum_{i,j} a[i,j] b[p[i], p[j]].
        Linearize: for each facility i, define effective cost of placing at location l
        as sum_j a[i,j] * b[l, q[j]] where q is a reference ordering.
        We use the row-sum ordering as reference.
        """
        # Reference: sort locations by b row sums
        loc_order = np.argsort(b.sum(axis=1), kind="stable")
        # For facility i, cost of assigning to location l:
        # sum_j a[i,j] * b[l, loc_order[j]]
        # This is (a @ b[:, loc_order])[i, :] but we need a matrix M[i, l]
        # M[i, l] = sum_j a[i,j] * b[l, loc_order[j]] = (a @ b.T @ ???)
        # Actually: sum_j a[i,j] * b[l, loc_order[j]] = row i of (a @ B_ref) where B_ref[j, l] = b[l, loc_order[j]]
        # B_ref = b.T[:, loc_order] ... let's just compute directly
        # B_ref[l, j] = b[l, loc_order[j]]
        B_ref = b[:, loc_order]  # shape (n, n), B_ref[l, j] = b[l, loc_order[j]]
        # M[i, l] = sum_j a[i,j] * B_ref[l, j] = (a @ B_ref.T)[i, l]
        M = a @ B_ref.T
        # Also add diagonal/self terms: a[i,i]*b[l,l] is included in M[i,l] when j=i: a[i,i]*b[l,loc_order[i]]
        # That's not quite b[l,l] unless loc_order[i]==l. Close enough for init.
        row_ind, col_ind = linear_sum_assignment(M)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    # Main loop
    best_p = None
    best_cost = np.inf

    # Initialization 1: rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_init = np.empty(n, dtype=np.int64)
    p_init[heavy] = central
    cost_init = compute_cost(p_init)

    if cost_init < best_cost:
        best_cost = cost_init
        best_p = p_init.copy()

    # Initialization 2: Greedy
    if time.time() < deadline - 5:
        p_greedy = greedy_init()
        cost_greedy = compute_cost(p_greedy)
        if cost_greedy < best_cost:
            best_cost = cost_greedy
            best_p = p_greedy.copy()

    # Initialization 3: Hungarian
    if time.time() < deadline - 5:
        p_hung = hungarian_init()
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_cost = cost_hung
            best_p = p_hung.copy()

    # Local search on best init
    p_ls, cost_ls = local_search(best_p, best_cost, min(deadline, time.time() + 20))
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()

    # GRASP + SA iterations
    iteration = 0
    while time.time() < deadline - 10:
        iteration += 1

        # Construct a new solution using GRASP
        p_grasp = grasp_construct(alpha=0.2)
        cost_grasp = compute_cost(p_grasp)

        # Local search
        p_ls, cost_ls = local_search(p_grasp, cost_grasp, min(deadline - 5, time.time() + 15))

        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()

        # SA on the best solution
        if time.time() < deadline - 15:
            t_start = max(best_cost * 0.1, 1000.0)
            t_end = max(best_cost * 0.001, 1.0)
            cooling = 0.995

            p_sa, cost_sa = simulated_annealing(best_p, best_cost, min(deadline - 10, time.time() + 10), t_start, t_end, cooling)

            # Local search on SA result
            p_sa_ls, cost_sa_ls = local_search(p_sa, cost_sa, min(deadline - 5, time.time() + 5))

            if cost_sa_ls < best_cost:
                best_cost = cost_sa_ls
                best_p = p_sa_ls.copy()

        if time.time() >= deadline - 5:
            break

    # Final polish with local search
    p_final, cost_final = local_search(best_p, best_cost, deadline)

    return p_final.tolist()
# EVOLVE-BLOCK-END
