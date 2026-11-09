"""Rewritten program: optimize 32-element subset of 1..512 for 3-term AP differences."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512

    # We represent the set as a bitmask (Python int). Bit i set means i is in the set.
    # Values are 1..512, so we use bits 1..512.

    def score_mask(mask):
        """Count distinct d>0 such that a, a+d, a+2d are all in the set."""
        # For each pair (a, b) in the set with a < b, d = b - a
        # Check if a + 2d is in the set
        valid_diffs = set()
        # Extract elements
        elems = [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
        elem_set = set(elems)
        for i in range(len(elems)):
            a = elems[i]
            for j in range(i + 1, len(elems)):
                b = elems[j]
                d = b - a
                c = a + 2 * d
                if c in elem_set:
                    valid_diffs.add(d)
        return len(valid_diffs)

    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def count_bits(mask):
        return bin(mask).count('1')

    # Greedy initialization
    def greedy_init():
        # Start with a random seed set, then greedily add elements that maximize score
        best_score = 0
        best_mask = 0
        for trial in range(200):
            random.seed(trial * 12345 + 7)
            mask = 0
            # Pick a random starting point
            start = random.randint(1, MAX_VAL)
            mask |= (1 << start)
            # Greedily add elements
            for _ in range(N - 1):
                best_add = None
                best_add_score = -1
                for cand in range(1, MAX_VAL + 1):
                    if (mask >> cand) & 1:
                        continue
                    new_mask = mask | (1 << cand)
                    s = score_mask(new_mask)
                    if s > best_add_score:
                        best_add_score = s
                        best_add = cand
                if best_add is not None:
                    mask |= (1 << best_add)
                if count_bits(mask) >= N:
                    break
            if count_bits(mask) == N:
                s = score_mask(mask)
                if s > best_score:
                    best_score = s
                    best_mask = mask
        return best_mask, best_score

    # Simulated annealing
    def simulated_annealing(initial_mask, initial_score, deadline):
        mask = initial_mask
        score = initial_score
        best_mask = mask
        best_score = score

        # Parameters
        T_start = 10.0
        T_end = 0.1
        cooling_rate = 0.9995

        T = T_start
        steps = 0

        while time.time() < deadline:
            # Generate a move: remove 2, add 2
            elems = [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
            if len(elems) < N:
                break

            # Remove 2 random elements
            rem1_idx = random.randint(0, len(elems) - 1)
            rem2_idx = random.randint(0, len(elems) - 1)
            if rem2_idx == rem1_idx:
                rem2_idx = (rem2_idx + 1) % len(elems)
            rem1 = elems[rem1_idx]
            rem2 = elems[rem2_idx]

            new_mask = mask & ~(1 << rem1) & ~(1 << rem2)

            # Add 2 random elements not in set
            added = 0
            attempts = 0
            while added < 2 and attempts < 100:
                cand = random.randint(1, MAX_VAL)
                if not ((new_mask >> cand) & 1):
                    new_mask |= (1 << cand)
                    added += 1
                attempts += 1

            if added < 2:
                continue

            if count_bits(new_mask) != N:
                continue

            new_score = score_mask(new_mask)
            delta = new_score - score

            if delta >= 0:
                mask = new_mask
                score = new_score
                if score > best_score:
                    best_score = score
                    best_mask = mask
            else:
                # Accept with probability
                import math
                if T > 1e-9:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        mask = new_mask
                        score = new_score

            T *= cooling_rate
            steps += 1

            # Periodically try single-element swaps too
            if steps % 50 == 0:
                elems2 = [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
                if len(elems2) == N:
                    rem_idx = random.randint(0, N - 1)
                    rem_elem = elems2[rem_idx]
                    swap_mask = mask & ~(1 << rem_elem)
                    cand = random.randint(1, MAX_VAL)
                    if not ((swap_mask >> cand) & 1):
                        swap_mask |= (1 << cand)
                        if count_bits(swap_mask) == N:
                            swap_score = score_mask(swap_mask)
                            sdelta = swap_score - score
                            if sdelta >= 0:
                                mask = swap_mask
                                score = swap_score
                                if score > best_score:
                                    best_score = score
                                    best_mask = mask
                            else:
                                import math
                                if T > 1e-9:
                                    prob = math.exp(sdelta / T)
                                    if random.random() < prob:
                                        mask = swap_mask
                                        score = swap_score

        return best_mask, best_score

    # Try multiple random restarts
    best_overall_mask = 0
    best_overall_score = 0

    # First, greedy init
    g_mask, g_score = greedy_init()
    if g_score > best_overall_score:
        best_overall_score = g_score
        best_overall_mask = g_mask

    # Then simulated annealing from greedy start
    if time.time() < deadline:
        sa_mask, sa_score = simulated_annealing(g_mask, g_score, deadline)
        if sa_score > best_overall_score:
            best_overall_score = sa_score
            best_overall_mask = sa_mask

    # Try a few more random restarts if time permits
    random.seed(42)
    restarts = 0
    while time.time() < deadline - 10 and restarts < 5:
        # Random initial set
        rand_set = random.sample(range(1, MAX_VAL + 1), N)
        rand_mask = list_to_mask(rand_set)
        rand_score = score_mask(rand_mask)
        if rand_score > best_overall_score:
            best_overall_score = rand_score
            best_overall_mask = rand_mask

        # Short SA from this start
        sa_mask2, sa_score2 = simulated_annealing(rand_mask, rand_score, min(deadline - 10, time.time() + 15))
        if sa_score2 > best_overall_score:
            best_overall_score = sa_score2
            best_overall_mask = sa_mask2
        restarts += 1

    result = mask_to_list(best_overall_mask)
    # Ensure exactly 32 distinct integers in 1..512
    if len(result) != N:
        # Fallback
        result = list(range(1, N + 1))

    return result

# EVOLVE-BLOCK-END
