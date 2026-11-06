"""
Reciprocal-weight maximization for 3-AP-free subsets of [1,256].

Strategy:
  1. Precompute conflict structure: for each x, the set of pairs (y,z) that
     would form a 3-AP with x.
  2. Build a strong initial solution via greedy (small-to-large) plus a
     deterministic local-improvement pass.
  3. Run multi-start simulated annealing with bitmask acceleration to explore
     the space of valid 3-AP-free sets, accepting occasional worsening moves
     to escape local optima.
  4. Return the best valid set found, maximizing sum(1/x).
"""

import time
import random
from fractions import Fraction


# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s wall-clock

    n = 256

    # Precompute all 3-APs and conflict pairs
    ap_list = []
    for b in range(2, n):
        for a in range(1, b):
            c = 2 * b - a
            if c <= n:
                ap_list.append((a, b, c))

    # conflict_pairs[x] = list of (y, z) such that {x, y, z} is a 3-AP
    # If both y and z are in the set, x cannot be added (and vice versa).
    conflict_pairs = {i: [] for i in range(1, n + 1)}
    for a, b, c in ap_list:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))

    # Bitmask representation: element i is bit (i-1)
    # masks[i] = 1 << (i-1)
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        masks[i] = 1 << (i - 1)

    # For each x, precompute pair_masks: list of (mask_y, mask_z)
    pair_masks = {i: [] for i in range(1, n + 1)}
    for x in range(1, n + 1):
        for y, z in conflict_pairs[x]:
            pair_masks[x].append((masks[y], masks[z]))

    # weights[x] = 1.0 / x
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i

    # Greedy initialization: add elements in order of largest weight (smallest x)
    def greedy_init():
        S_mask = 0
        # Process in order of increasing x (largest reciprocal first)
        for x in range(1, n + 1):
            # Check if x can be added: no pair (y,z) with both y,z already in S
            ok = True
            for my, mz in pair_masks[x]:
                if (S_mask & my) and (S_mask & mz):
                    ok = False
                    break
            if ok:
                S_mask |= masks[x]
        return S_mask

    # Check if a set (as mask) is valid (no 3-APs)
    def is_valid(S_mask):
        for x in range(1, n + 1):
            if S_mask & masks[x]:
                for my, mz in pair_masks[x]:
                    if (S_mask & my) and (S_mask & mz):
                        return False
        return True

    # Compute score of a mask
    def score_mask(S_mask):
        total = 0.0
        for x in range(1, n + 1):
            if S_mask & masks[x]:
                total += weights[x]
        return total

    # Convert mask to sorted list
    def mask_to_list(S_mask):
        result = []
        for x in range(1, n + 1):
            if S_mask & masks[x]:
                result.append(x)
        return result

    # Deterministic local improvement: try all single-element swaps
    def local_improve(S_mask, max_passes=50):
        current_score = score_mask(S_mask)
        best_score = current_score
        best_mask = S_mask
        for _ in range(max_passes):
            improved = False
            # Try removing x and adding y (y < x for potential improvement)
            for x in range(1, n + 1):
                if not (S_mask & masks[x]):
                    continue
                for y in range(1, x):
                    if S_mask & masks[y]:
                        continue
                    # Try swap: remove x, add y
                    temp = S_mask & ~masks[x]
                    # Check if y can be added to temp
                    ok = True
                    for my, mz in pair_masks[y]:
                        if (temp & my) and (temp & mz):
                            ok = False
                            break
                    if ok:
                        new_mask = temp | masks[y]
                        new_score = current_score + weights[y] - weights[x]
                        if new_score > best_score:
                            best_score = new_score
                            best_mask = new_mask
                            S_mask = new_mask
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return best_mask, best_score

    # Simulated annealing
    def simulated_annealing(S_mask, t0, t_min, cooling, max_iter, time_limit):
        current_mask = S_mask
        current_score = score_mask(S_mask)
        best_mask = S_mask
        best_score = current_score
        T = t0
        step = 0
        while T > t_min and step < max_iter and time.time() < time_limit:
            step += 1
            # Random move: pick random element to toggle (add or remove)
            x = random.randint(1, n)
            if S_mask & masks[x]:
                # Remove x
                new_mask = S_mask & ~masks[x]
                delta = -weights[x]
            else:
                # Add x
                # Check validity
                ok = True
                for my, mz in pair_masks[x]:
                    if (S_mask & my) and (S_mask & mz):
                        ok = False
                        break
                if ok:
                    new_mask = S_mask | masks[x]
                    delta = weights[x]
                else:
                    continue
            new_score = current_score + delta
            if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
                S_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_mask = S_mask
            T *= cooling
        return best_mask, best_score

    # Multi-element removal + re-add strategy for escaping local optima
    def multi_readd(S_mask, k):
        """Remove k random elements, then greedily re-add best candidates."""
        if k == 0:
            return S_mask
        removed = []
        temp = S_mask
        for _ in range(k):
            # Pick a random element from temp
            candidates = []
            for x in range(1, n + 1):
                if temp & masks[x]:
                    candidates.append(x)
            if not candidates:
                break
            x = random.choice(candidates)
            temp &= ~masks[x]
            removed.append(x)
        # Now greedily try to add back elements (including removed ones)
        # Sort all elements by weight descending
        all_elems = sorted(range(1, n + 1), key=lambda i: -weights[i])
        for x in all_elems:
            if temp & masks[x]:
                continue
            ok = True
            for my, mz in pair_masks[x]:
                if (temp & my) and (temp & mz):
                    ok = False
                    break
            if ok:
                temp |= masks[x]
        return temp

    # Main search
    # Start with greedy
    best_overall_mask = greedy_init()
    best_overall_score = score_mask(best_overall_mask)

    # Deterministic local improvement
    best_overall_mask, best_overall_score = local_improve(best_overall_mask, max_passes=100)

    # Multi-start simulated annealing
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        # Perturb the best solution
        if r == 0:
            S_mask = best_overall_mask
        elif r % 3 == 1:
            # Remove 2-4 random elements and re-add
            S_mask = multi_readd(best_overall_mask, random.randint(2, 4))
        elif r % 3 == 2:
            # Fresh greedy with random ordering
            S_mask = 0
            order = list(range(1, n + 1))
            random.shuffle(order)
            # Process in random order but prefer small numbers
            # Actually, let's do: for x in sorted order by weight, but with random tie-breaking
            # Simpler: random permutation, but add if valid
            for x in order:
                ok = True
                for my, mz in pair_masks[x]:
                    if (S_mask & my) and (S_mask & mz):
                        ok = False
                        break
                if ok:
                    S_mask |= masks[x]
        else:
            # Start from greedy
            S_mask = greedy_init()

        # Local improve the starting point
        S_mask, _ = local_improve(S_mask, max_passes=20)

        # Run SA
        t0 = 2.0
        t_min = 0.01
        cooling = 0.995
        max_iter = 200000
        S_mask, sc = simulated_annealing(S_mask, t0, t_min, cooling, max_iter, deadline)

        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_mask = S_mask

    # Final local improvement
    best_overall_mask, best_overall_score = local_improve(best_overall_mask, max_passes=200)

    # Verify validity
    if not is_valid(best_overall_mask):
        # Fallback to greedy
        best_overall_mask = greedy_init()

    # Ensure nonempty
    if best_overall_mask == 0:
        best_overall_mask = masks[1]

    return mask_to_list(best_overall_mask)

# EVOLVE-BLOCK-END
