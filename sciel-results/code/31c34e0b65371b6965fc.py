"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def greedy_init(a, b, rng):
    """
    Greedy construction: place facilities one by one, choosing the location
    that minimizes incremental cost given already-placed facilities.
    Uses a randomized tie-breaking / perturbation to diversify.
    """
    n = len(a)
    p = np.empty(n, dtype=np.int64)
    unplaced = list(range(n))
    used_locs = set()

    # Precompute row sums of a for ordering
    a_row_sums = a.sum(axis=1)
    # Order facilities by descending flow row-sum (place high-flow facilities first)
    order = np.argsort(-a_row_sums, kind='stable')

    # Precompute b row sums for location desirability
    b_row_sums = b.sum(axis=1)

    for idx in order:
        best_loc = -1
        best_cost = float('inf')
        for loc in range(n):
            if loc in used_locs:
                continue
            # Incremental cost of placing facility idx at location loc
            # = sum over already-placed j: a[idx,j]*b[loc,p[j]] + a[j,idx]*b[p[j],loc]
            # Since a is symmetric in QAP, a[idx,j] == a[j,idx], but let's be general
            inc = 0
            for j in range(n):
                if j == idx or p[j] == -1:
                    continue
                inc += a[idx, j] * b[loc, p[j]]
            # Add small random perturbation for diversity
            inc += rng.random() * 1e-6 * abs(inc) if inc != 0 else rng.random() * 1e-6
            if inc < best_cost:
                best_cost = inc
                best_loc = loc
        p[idx] = best_loc
        used_locs.add(best_loc)

    return p


def two_opt_eval(a, b, p, i, j):
    """
    Compute the delta cost of swapping positions i and j in permutation p.
    Only valid if i and j are distinct indices in the permutation.
    """
    li = p[i]
    lj = p[j]
    if li == lj:
        return 0

    # Delta = sum over k != i,j of:
    #   a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li])
    # + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj])
    # + a[i,j]*(b[lj,lj] - b[li,li]) + a[j,i]*(b[lj,lj] - b[li,li])
    # - a[i,j]*(b[li,lj] - b[li,lj]) ... wait, let's be careful.

    # Before swap:
    #   term(i,j) = a[i,j]*b[li,lj]
    #   term(j,i) = a[j,i]*b[lj,li]
    #   For each k: term(i,k) = a[i,k]*b[li,p[k]], term(k,i) = a[k,i]*b[p[k],li]
    #   For each k: term(j,k) = a[j,k]*b[lj,p[k]], term(k,j) = a[k,j]*b[p[k],lj]
    #
    # After swap (i gets lj, j gets li):
    #   term(i,j) = a[i,j]*b[lj,li]
    #   term(j,i) = a[j,i]*b[li,lj]
    #   For each k: term(i,k) = a[i,k]*b[lj,p[k]], term(k,i) = a[k,i]*b[p[k],lj]
    #   For each k: term(j,k) = a[j,k]*b[li,p[k]], term(k,j) = a[k,j]*b[p[k],li]
    #
    # Delta = (after - before)
    # For the i,j pair:
    #   a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
    #   = (a[i,j] - a[j,i])*(b[lj,li] - b[li,lj])
    # If a is symmetric, this is 0. Let's keep it general.

    delta = (a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li]))

    # For each k != i, j:
    for k in range(len(p)):
        if k == i or k == j:
            continue
        lk = p[k]
        # i's contribution with k
        delta += a[i, k] * (b[lj, lk] - b[li, lk])
        delta += a[k, i] * (b[lk, lj] - b[lk, li])
        # j's contribution with k
        delta += a[j, k] * (b[li, lk] - b[lj, lk])
        delta += a[k, j] * (b[lk, li] - b[lk, lj])

    return int(delta)


def simulated_annealing(a, b, p, rng, time_limit):
    """
    Simulated Annealing with 2-opt neighborhood.
    """
    n = len(p)
    current_cost = compute_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost

    start_temp = 1000.0
    end_temp = 1.0
    cooling_rate = 0.9995

    temp = start_temp
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 5000

    while temp > end_temp:
        if time.time() - start_time > time_limit:
            break

        # Generate a random 2-opt move
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue

        delta = two_opt_eval(a, b, p, i, j)

        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            # Accept the move
            p[i], p[j] = p[j], p[i]
            current_cost += delta

            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count > max_no_improve:
                # Restart with a steeper cooling
                temp = max(temp * 0.95, end_temp)
                no_improve_count = 0

        temp *= cooling_rate

    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    rng = np.random.default_rng(42)

    best_p = None
    best_cost = float('inf')

    # Phase 1: Multiple greedy initializations
    num_greedy = 3
    for g in range(num_greedy):
        if time.time() - start_time > deadline - 30:
            break
        p_g = greedy_init(a, b, rng)
        cost_g = compute_cost(a, b, p_g)
        if cost_g < best_cost:
            best_cost = cost_g
            best_p = p_g.copy()

    # Phase 2: Simulated Annealing from best initialization
    if best_p is not None:
        sa_time = min(80.0, deadline - (time.time() - start_time) - 5)
        if sa_time > 5:
            best_p, best_cost = simulated_annealing(a, b, best_p, rng, sa_time)

    # Phase 3: Second SA run from a different init if time permits
    remaining = deadline - (time.time() - start_time) - 3
    if remaining > 10 and best_p is not None:
        p_alt = greedy_init(a, b, rng)
        cost_alt = compute_cost(a, b, p_alt)
        if cost_alt < best_cost:
            best_cost = cost_alt
            best_p = p_alt.copy()
        # SA from alt
        sa_time2 = min(remaining - 2, 50.0)
        if sa_time2 > 5:
            p2, cost2 = simulated_annealing(a, b, best_p.copy(), rng, sa_time2)
            if cost2 < best_cost:
                best_cost = cost2
                best_p = p2

    # Ensure valid permutation
    if best_p is None:
        best_p = np.arange(n, dtype=np.int64)

    return best_p.tolist()
# EVOLVE-BLOCK-END
