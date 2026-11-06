"""Combined: bitmask coprime checks + heuristic local search + MILP polish via scipy."""
import math
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 138.0  # safety margin before 150s kill
    N = 512

    # Precompute prime factor bitmasks for all numbers 1..511
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N, i):
                is_prime[j] = False

    primes = [i for i in range(2, N) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}

    prime_factors_mask = [0] * N
    for a in range(2, N):
        x = a
        mask = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                mask |= (1 << prime_index[p])
                while x % p == 0:
                    x //= p
        if x > 1:
            mask |= (1 << prime_index[x])
        prime_factors_mask[a] = mask
    prime_factors_mask[1] = 0

    def score(arr):
        return sum(1.0 / (N - a) for a in arr)

    # Conflict graph: two numbers conflict if not coprime
    # For MWIS: we want max-weight subset with no conflicts
    # Vertex a (index 0..511, value 1..511) has weight w[a] = 1/(512-a)
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    # Conflict adjacency: conflict_mask[a] is a bitmask of which other numbers conflict with a
    # Since we have 512 vertices, we use Python integers as bitsets
    conflict_mask = [0] * N
    for a in range(1, N):
        for b in range(a + 1, N):
            if (prime_factors_mask[a] & prime_factors_mask[b]) != 0:
                # They share a prime factor, so not coprime -> conflict
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)

    # Greedy from largest to smallest (baseline)
    def greedy_desc():
        chosen = []
        used_mask = 0
        for a in range(511, 0, -1):
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    best = greedy_desc()
    best_score = score(best)

    # Randomized greedy with multiple orderings
    rng = random.Random(42)
    all_nums = list(range(1, 512))

    def greedy_with_order(order):
        chosen = []
        used_mask = 0
        for a in order:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
        return chosen

    for trial in range(300):
        if time.time() > deadline:
            break
        # Biased random order
        weights_list = [1.0 / (N - a) for a in all_nums]
        remaining = list(all_nums)
        rem_weights = list(weights_list)
        order = []
        for _ in range(len(remaining)):
            total_w = sum(rem_weights)
            if total_w <= 0:
                order.extend(remaining)
                break
            r = rng.random() * total_w
            cumulative = 0
            for i, w in enumerate(rem_weights):
                cumulative += w
                if cumulative >= r:
                    order.append(remaining.pop(i))
                    rem_weights.pop(i)
                    break
            else:
                order.extend(remaining)
                break
        cand = greedy_with_order(order)
        s = score(cand)
        if s > best_score:
            best_score = s
            best = cand

    # Local search
    def local_search(chosen, max_iters=200):
        chosen_set = set(chosen)
        current_score = sum(1.0 / (N - a) for a in chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= prime_factors_mask[a]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1

            # Try adding a new element
            for a in range(511, 0, -1):
                if a in chosen_set:
                    continue
                if (prime_factors_mask[a] & used_mask) == 0:
                    chosen.append(a)
                    chosen_set.add(a)
                    used_mask |= prime_factors_mask[a]
                    current_score += 1.0 / (N - a)
                    improved = True
                    break

            if improved:
                continue

            # Try single removal + greedy refill
            sorted_chosen = sorted(chosen, key=lambda x: 1.0 / (N - x))
            for a in sorted_chosen:
                if time.time() > deadline:
                    break
                remaining = [x for x in chosen if x != a]
                rem_set = set(remaining)
                rem_mask = 0
                for x in remaining:
                    rem_mask |= prime_factors_mask[x]

                new_chosen = list(remaining)
                new_set = set(remaining)
                new_mask = rem_mask
                new_score = sum(1.0 / (N - x) for x in new_chosen)

                for c in range(511, 0, -1):
                    if c in new_set:
                        continue
                    if (prime_factors_mask[c] & new_mask) == 0:
                        new_chosen.append(c)
                        new_set.add(c)
                        new_mask |= prime_factors_mask[c]
                        new_score += 1.0 / (N - c)

                if new_score > current_score + 1e-12:
                    chosen = new_chosen
                    chosen_set = new_set
                    used_mask = new_mask
                    current_score = new_score
                    improved = True
                    break

            if improved:
                continue

            # Try double removal
            if len(chosen) >= 2:
                for i in range(len(sorted_chosen)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(sorted_chosen)):
                        a, b = sorted_chosen[i], sorted_chosen[j]
                        remaining = [x for x in chosen if x != a and x != b]
                        rem_set = set(remaining)
                        rem_mask = 0
                        for x in remaining:
                            rem_mask |= prime_factors_mask[x]

                        new_chosen = list(remaining)
                        new_set = set(remaining)
                        new_mask = rem_mask
                        new_score = sum(1.0 / (N - x) for x in new_chosen)
                        added_count = 0

                        for c in range(511, 0, -1):
                            if c in new_set:
                                continue
                            if (prime_factors_mask[c] & new_mask) == 0:
                                new_chosen.append(c)
                                new_set.add(c)
                                new_mask |= prime_factors_mask[c]
                                new_score += 1.0 / (N - c)
                                added_count += 1
                                if added_count >= 4:
                                    break

                        if new_score > current_score + 1e-12:
                            chosen = new_chosen
                            chosen_set = new_set
                            used_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break

        return chosen, current_score

    best, best_score = local_search(best)

    # Random restarts with local search
    for restart in range(150):
        if time.time() > deadline - 20:  # reserve time for B&B
            break

        rng_seed = 1000 + restart * 7919
        rng2 = random.Random(rng_seed)
        strategy = restart % 5

        if strategy == 0:
            seed_candidates = list(range(511, 300, -1))
        elif strategy == 1:
            seed_candidates = list(range(511, 200, -1))
        elif strategy == 2:
            seed_candidates = list(range(511, 100, -1))
        elif strategy == 3:
            seed_candidates = [a for a in range(511, 400, -1) if rng2.random() < 0.5]
        else:
            seed_candidates = list(range(511, 450, -1)) + [rng2.randint(1, 511) for _ in range(20)]

        rng2.shuffle(seed_candidates)
        num_seeds = 2 + (restart % 5)
        chosen = []
        used_mask = 0
        for a in seed_candidates:
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]
            if len(chosen) >= num_seeds:
                break

        if not chosen:
            continue

        for a in range(511, 0, -1):
            if a in chosen:
                continue
            if (prime_factors_mask[a] & used_mask) == 0:
                chosen.append(a)
                used_mask |= prime_factors_mask[a]

        chosen, s = local_search(chosen, max_iters=100)
        if s > best_score:
            best_score = s
            best = chosen

    # MILP polish: solve max-weight independent set via scipy.optimize.milp
    # Variables x_i in {0,1} for each vertex i (0..510, corresponding to a=i+1)
    # Maximize sum(w_i * x_i)
    # Subject to: x_i + x_j <= 1 for all conflicting pairs (i,j)
    #             x_i in {0,1}
    
    n_vars = 511  # vertices 0..510
    w0 = np.zeros(n_vars)
    for i in range(n_vars):
        w0[i] = 1.0 / (511 - i)  # weight for a=i+1: 1/(512-(i+1)) = 1/(511-i)
    
    # Build constraint matrix: for each conflicting pair (i,j), row with 1 at i and j
    # We need to enumerate all conflicting pairs
    conflict_pairs = []
    for i in range(n_vars):
        a = i + 1
        for j in range(i + 1, n_vars):
            b = j + 1
            if (prime_factors_mask[a] & prime_factors_mask[b]) != 0:
                conflict_pairs.append((i, j))
    
    n_constraints = len(conflict_pairs)
    
    if n_constraints > 0:
        # Build sparse constraint matrix
        rows = []
        cols = []
        vals = []
        for idx, (i, j) in enumerate(conflict_pairs):
            rows.extend([idx, idx])
            cols.extend([i, j])
            vals.extend([1.0, 1.0])
        
        A = csr_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))
        c = LinearConstraint(A, ub=np.ones(n_constraints))
    else:
        c = None
    
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Use the heuristic solution as an initial guess / warm start
    # scipy milp doesn't directly support warm starts, but we can set integrality
    integrality = np.ones(n_vars)  # all binary
    
    milp_start = time.time()
    milp_deadline = deadline - 5
    
    try:
        result = milp(
            c=-w0,  # minimize -sum(w_i * x_i) = maximize sum(w_i * x_i)
            constraints=c,
            integrality=integrality,
            bounds=bounds,
        )
        
        if result.status == 0 and result.x is not None:
            milp_solution = []
            milp_score = 0.0
            for i in range(n_vars):
                if result.x[i] > 0.5:
                    milp_solution.append(i + 1)
                    milp_score += w0[i]
            
            if milp_score > best_score + 1e-12:
                best = milp_solution
                best_score = milp_score
        
        milp_elapsed = time.time() - milp_start
        # If MILP finished quickly, we have remaining time for additional refinement
        if milp_elapsed < 30 and time.time() < milp_deadline:
            # Try additional local search on the MILP solution
            if best:
                best, final_score = local_search(best, max_iters=200)
                if final_score > best_score:
                    best_score = final_score
    
    except Exception:
        # If MILP fails, fall back to current best
        pass

    # Final polish: one more local search pass
    if time.time() < deadline - 2:
        best, final_score = local_search(best, max_iters=100)
        if final_score > best_score:
            best_score = final_score

    # Ensure we have a valid answer
    if not best:
        best = [1]  # 1 is coprime to everything

    return best

# EVOLVE-BLOCK-END
