"""Simulated annealing for covering design C(v,k,t)."""
import time
import itertools
import random
from math import comb


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)

    # Map t-subset tuple to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, precompute the bitmask of t-subsets it covers
    # n_tsub can be up to C(19,4)=3876, so we need ~61 64-bit words
    # Use Python big integers as bitsets
    coverage = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(blk, t):
            ti = tsub_to_idx[ts]
            mask |= (1 << ti)
        coverage[bi] = mask

    # Helper: count bits in an integer
    def popcount(x):
        return bin(x).count('1')

    # Initial solution: greedy set cover to get a starting point
    def greedy_init():
        uncovered = (1 << n_tsub) - 1
        chosen = []
        used = [False] * n_blocks
        while uncovered:
            best_gain = -1
            best_bi = -1
            for bi in range(n_blocks):
                if used[bi]:
                    continue
                gain = popcount(coverage[bi] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi == -1 or best_gain <= 0:
                break
            used[best_bi] = True
            chosen.append(best_bi)
            uncovered &= ~coverage[best_bi]
        return chosen

    initial = greedy_init()
    n_initial = len(initial)
    # Target: try to match or beat the best known (151 for C(19,7,4))
    # Use a target size that's slightly above best known to allow exploration
    target_size = max(n_initial, 151) if n_initial <= 151 else n_initial

    # Simulated annealing
    def run_annealing(seed, target_n, time_budget):
        rng = random.Random(seed)
        
        # Start from greedy solution, pad or trim to target_n
        sol = list(initial)
        if len(sol) > target_n:
            sol = sol[:target_n]
        elif len(sol) < target_n:
            # Add random blocks
            used_set = set(sol)
            for _ in range(target_n - len(sol)):
                bi = rng.randint(0, n_blocks - 1)
                if bi not in used_set:
                    sol.append(bi)
                    used_set.add(bi)
        
        # Compute current uncovered
        def compute_uncovered(sol):
            unc = (1 << n_tsub) - 1
            for bi in sol:
                unc &= ~coverage[bi]
            return unc

        current_uncovered = compute_uncovered(sol)
        current_uncovered_count = popcount(current_uncovered)
        best_sol = list(sol)
        best_uncovered_count = current_uncovered_count

        # Temperature schedule
        t_max = 1.0
        t_min = 0.001
        # Number of iterations based on time budget
        # Each iteration: pick random position, pick random replacement, compute gain
        # Estimate: ~0.0001s per iteration for n_blocks=50388
        # We'll just run until time budget is exhausted

        iter_count = 0
        while True:
            if time.time() > deadline - 2.0:
                break
            if time.time() > start_time + time_budget:
                break

            # Pick a random position in the solution
            pos = rng.randint(0, target_n - 1)
            old_bi = sol[pos]

            # Pick a random replacement block
            new_bi = rng.randint(0, n_blocks - 1)
            if new_bi == old_bi:
                continue

            # Compute change in uncovered count
            # Removing old_bi: adds back its coverage
            # Adding new_bi: removes its coverage
            # Net change in uncovered = popcount((coverage[old_bi] & current_uncovered)) 
            #                           - popcount((coverage[new_bi] & (current_uncovered | coverage[old_bi])))
            # But this is complex. Simpler: compute delta directly
            # uncovered_after = (current_uncovered | coverage[old_bi]) & ~coverage[new_bi]
            # delta = popcount(uncovered_after) - current_uncovered_count

            temp = t_max * ((t_min / t_max) ** (iter_count / max(1, int(time_budget * 5000))))
            
            # Quick check: if new block doesn't cover anything currently uncovered and old does, skip
            # For efficiency, compute the actual delta
            candidate_uncovered = (current_uncovered | coverage[old_bi]) & ~coverage[new_bi]
            delta = popcount(candidate_uncovered) - current_uncovered_count

            # Accept if delta <= 0 (improvement) or with probability exp(-delta/temp)
            if delta <= 0:
                sol[pos] = new_bi
                current_uncovered = candidate_uncovered
                current_uncovered_count += delta
                if current_uncovered_count < best_uncovered_count:
                    best_uncovered_count = current_uncovered_count
                    best_sol = list(sol)
            else:
                import math
                if rng.random() < math.exp(-delta / temp):
                    sol[pos] = new_bi
                    current_uncovered = candidate_uncovered
                    current_uncovered_count += delta

            iter_count += 1

        return best_sol, best_uncovered_count

    # Run multiple annealing attempts with different seeds and target sizes
    best_overall = None
    best_overall_count = n_tsub  # worst case

    # Try target sizes around the known best
    target_sizes = sorted(set([151, 152, 153, 150, 155, 160]))
    for target_n in target_sizes:
        if time.time() > deadline - 10.0:
            break
        remaining_time = deadline - time.time() - 5.0
        if remaining_time < 3.0:
            break
        # Allocate time proportionally
        time_per_target = max(2.0, remaining_time * 0.8 / max(1, len(target_sizes)))
        sol, unc_count = run_annealing(42 + target_n, target_n, time_per_target)
        if unc_count < best_overall_count:
            best_overall_count = unc_count
            best_overall = sol

    # If we still have uncovered, do a final greedy completion
    if best_overall is None:
        best_overall = initial
        best_overall_count = popcount(compute_uncovered(best_overall))

    if best_overall_count > 0:
        # Complete with greedy
        used_set = set(best_overall)
        unc = compute_uncovered(best_overall)
        while unc and time.time() < deadline - 1.0:
            best_gain = -1
            best_bi = -1
            for bi in range(n_blocks):
                if bi in used_set:
                    continue
                gain = popcount(coverage[bi] & unc)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi == -1 or best_gain <= 0:
                break
            best_overall.append(best_bi)
            used_set.add(best_bi)
            unc &= ~coverage[best_bi]

    # Convert to list of blocks
    final_blocks = [list(all_blocks[bi]) for bi in best_overall]

    # Ensure distinct and sorted
    seen = set()
    final = []
    for b in final_blocks:
        b_sorted = sorted(b)
        key = tuple(b_sorted)
        if key not in seen:
            seen.add(key)
            final.append(b_sorted)

    return final

