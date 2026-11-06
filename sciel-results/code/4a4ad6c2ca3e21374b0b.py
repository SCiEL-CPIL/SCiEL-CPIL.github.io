"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute weights: 1/(512 - a) for a in 1..511
    # We want to maximize sum of weights for a pairwise coprime set
    # a=511 has weight 1/1, a=510 has weight 1/2, etc.
    # So we prioritize larger a values (smaller 512-a)

    # Precompute gcd compatibility
    # conflict[a][b] = True if gcd(a,b) != 1
    # For efficiency, precompute for each a the set of b that conflict with it

    # Strategy: Branch and bound with greedy upper bound
    # Sort candidates by weight descending (a=511, 510, 509, ...)
    # At each step, try adding the current candidate or skipping it

    # Precompute coprime relationships using bitsets for speed
    # Represent each number's "compatible set" as a Python int bitmask
    # Bit i is set if number i is coprime with the current number

    N = 512
    # Numbers 1 to 511, index 0 to 510
    # weights[i] = 1/(512 - (i+1)) = 1/(511 - i)
    # Actually a ranges from 1 to 511, so index i corresponds to a = i+1
    # weight for a is 1/(512 - a)

    # Let's use 0-indexed: vertex i represents a = i + 1
    # weight[i] = 1/(512 - (i+1)) = 1/(511 - i)
    # For i=510 (a=511): weight = 1/1 = 1.0
    # For i=0 (a=1): weight = 1/511

    # Precompute coprime masks
    # coprime_mask[i] is a bitset where bit j is set if gcd(i+1, j+1) == 1
    coprime_mask = [0] * 511
    for i in range(511):
        a = i + 1
        mask = 0
        for j in range(511):
            b = j + 1
            if gcd(a, b) == 1:
                mask |= (1 << j)
        coprime_mask[i] = mask

    # weights in descending order of a (i.e., descending index)
    # We'll process vertices in order of decreasing weight: i=510, 510, ..., 0
    # weight[i] = 1/(511 - i)

    # Branch and bound
    best_score = Fraction(0)
    best_set = []

    # Greedy initial solution: take largest a values that are pairwise coprime
    greedy_set = []
    available = (1 << 511) - 1  # all bits set
    for i in range(510, -1, -1):
        if available & (1 << i):
            greedy_set.append(i)
            # Remove all vertices that conflict with i
            # Keep only those coprime with i
            available &= coprime_mask[i]

    # Compute greedy score
    greedy_score = sum(Fraction(1, 511 - i) for i in greedy_set)
    best_score = greedy_score
    best_set = list(greedy_set)

    # Now try to improve with local search / branch and bound
    # Use a recursive branch and bound with memoization on the available set
    # But 511 vertices is too many for full B&B. Use a smarter approach.

    # Alternative: Iterative improvement
    # 1. Start with greedy solution
    # 2. Try to swap or add/remove elements to improve
    # 3. Use multiple restarts with different greedy orders

    def compute_score(vertex_set):
        return sum(Fraction(1, 511 - i) for i in vertex_set)

    def is_valid_set(vertex_set):
        vset = set(vertex_set)
        for i in vset:
            for j in vset:
                if i != j and gcd(i+1, j+1) != 1:
                    return False
        return True

    def local_search(current_set, time_limit):
        """Try to improve the current set via add/remove/swap operations."""
        current_set = list(current_set)
        current_score = compute_score(current_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_set_set = set(current_set)

            # Try adding a single vertex
            for i in range(510, -1, -1):
                if i in current_set_set:
                    continue
                # Check if i is coprime with all in current_set
                can_add = True
                for j in current_set:
                    if gcd(i+1, j+1) != 1:
                        can_add = False
                        break
                if can_add:
                    new_set = current_set + [i]
                    new_score = current_score + Fraction(1, 511 - i)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        current_set_set = set(current_set)
                        improved = True
                        break

            if time.time() >= time_limit:
                break

            # Try removing a vertex and adding two others
            for remove_idx in range(len(current_set)):
                if time.time() >= time_limit:
                    break
                removed = current_set[remove_idx]
                remaining = current_set[:remove_idx] + current_set[remove_idx+1:]
                remaining_set = set(remaining)
                remaining_score = current_score - Fraction(1, 511 - removed)

                # Find candidates coprime with all in remaining
                candidates = []
                for i in range(510, -1, -1):
                    if i in remaining_set or i == removed:
                        continue
                    can_add = True
                    for j in remaining:
                        if gcd(i+1, j+1) != 1:
                            can_add = False
                            break
                    if can_add:
                        candidates.append(i)

                # Try adding up to 2 candidates
                # First, try adding 1
                for i in candidates:
                    if time.time() >= time_limit:
                        break
                    # Check coprime with removed? No, we're removing it
                    new_set = remaining + [i]
                    new_score = remaining_score + Fraction(1, 511 - i)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        current_set_set = set(current_set)
                        improved = True
                        break
                if improved:
                    break

                # Try adding 2
                for idx1 in range(len(candidates)):
                    if time.time() >= time_limit:
                        break
                    for idx2 in range(idx1+1, len(candidates)):
                        if time.time() >= time_limit:
                            break
                        i1 = candidates[idx1]
                        i2 = candidates[idx2]
                        if gcd(i1+1, i2+1) != 1:
                            continue
                        new_set = remaining + [i1, i2]
                        new_score = remaining_score + Fraction(1, 511 - i1) + Fraction(1, 511 - i2)
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            current_set_set = set(current_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return current_set, current_score

    # Run local search on the greedy solution
    time_limit = deadline - 10
    best_set, best_score = local_search(best_set, time_limit)

    # Try multiple greedy orders for diversity
    import random
    random.seed(42)

    for restart in range(20):
        if time.time() >= deadline - 15:
            break

        # Different greedy strategies
        if restart == 0:
            # Original greedy (largest first)
            order = list(range(510, -1, -1))
        elif restart == 1:
            # Random order
            order = list(range(511))
            random.shuffle(order)
        elif restart == 2:
            # Weighted random: higher weight more likely
            weights = [1.0 / (511 - i) for i in range(511)]
            total_w = sum(weights)
            order = []
            remaining = list(range(511))
            while remaining:
                w = [weights[i] for i in remaining]
                total = sum(w)
                r = random.random() * total
                cum = 0
                for idx, i in enumerate(remaining):
                    cum += w[idx]
                    if r <= cum:
                        order.append(i)
                        remaining.remove(i)
                        break
        else:
            # Partially random: keep top k, shuffle rest
            k = 511 - restart * 10
            k = max(k, 50)
            top = list(range(510, 510-k, -1))
            rest = list(range(510-k, -1, -1))
            random.shuffle(rest)
            order = top + rest

        # Greedy construction with this order
        greedy_set = []
        available = (1 << 511) - 1
        for i in order:
            if available & (1 << i):
                greedy_set.append(i)
                available &= coprime_mask[i]

        greedy_score = compute_score(greedy_set)
        if greedy_score > best_score:
            best_score = greedy_score
            best_set = list(greedy_set)

        # Local search from this starting point
        if time.time() < deadline - 15:
            ls_set, ls_score = local_search(greedy_set, deadline - 15)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set

    # Final polish: try single swaps on the best set
    if time.time() < deadline - 5:
        best_set_set = set(best_set)
        for i in range(510, -1, -1):
            if time.time() >= deadline - 5:
                break
            if i in best_set_set:
                continue
            # Check if i can be added
            can_add = all(gcd(i+1, j+1) == 1 for j in best_set)
            if can_add:
                new_score = best_score + Fraction(1, 511 - i)
                if new_score > best_score:
                    best_set.append(i)
                    best_set_set.add(i)
                    best_score = new_score

    # Convert to actual values a = i + 1
    result = [i + 1 for i in best_set]
    return result

# EVOLVE-BLOCK-END
