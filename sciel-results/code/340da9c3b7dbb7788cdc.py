"""Construct a large binary code of length 17 with minimum distance 6."""
import time
import random
from itertools import combinations

N = 17
MIN_DISTANCE = 6
DEADLINE_FRACTION = 0.90  # use 90% of the 60s budget for safety


def _hamming(a, b):
    return (a ^ b).bit_count()


def _is_compatible(candidate, selected):
    for s in selected:
        if _hamming(candidate, s) < MIN_DISTANCE:
            return False
    return True


def _greedy_extend(selected, candidates, rng):
    """Greedy local search: try to add more candidates, then swap to improve."""
    selected_set = set(selected)
    improved = True
    while improved:
        improved = False
        # Try to add any remaining candidate
        for c in candidates:
            if c not in selected_set:
                if _is_compatible(c, selected_set):
                    selected_set.add(c)
                    improved = True
        if not improved:
            # Try swaps: remove one, add two
            sel_list = list(selected_set)
            for i in range(len(sel_list)):
                for j in range(i + 1, len(sel_list)):
                    a, b = sel_list[i], sel_list[j]
                    temp_set = selected_set - {a, b}
                    found = 0
                    for c in candidates:
                        if c in temp_set:
                            continue
                        if _is_compatible(c, temp_set):
                            temp_set.add(c)
                            found += 1
                            if found >= 2:
                                break
                    if found >= 2:
                        selected_set = temp_set
                        improved = True
                        break
                if improved:
                    break
    return list(selected_set)


def _random_candidate(rng):
    """Generate a random 17-bit word with weight 8 or 9 (optimal weight classes)."""
    weight = rng.choice([8, 9])
    positions = rng.sample(range(N), weight)
    word = 0
    for p in positions:
        word |= (1 << p)
    return word


def _all_weight_words(weight):
    """Generate all 17-bit words of given weight."""
    words = []
    for positions in combinations(range(N), weight):
        word = 0
        for p in positions:
            word |= (1 << p)
        words.append(word)
    return words


def construct():
    start_time = time.time()
    deadline = start_time + 60.0 * DEADLINE_FRACTION

    rng = random.Random(42)

    # Start with the known 32-word Reed-Muller seed
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)

    # Also try to build a larger candidate pool from weight-8 and weight-9 words
    # But the full pool is C(17,8) + C(17,9) = 24310 + 24310 = 48620 words
    # That's too many for pairwise checks in 60s, so use a subset

    # Generate a random subset of weight-8 and weight-9 words
    all_w8 = _all_weight_words(8)
    all_w9 = _all_weight_words(9)
    rng.shuffle(all_w8)
    rng.shuffle(all_w9)

    # Use a manageable subset: first 5000 from each
    candidate_pool = all_w8[:5000] + all_w9[:5000]
    rng.shuffle(candidate_pool)

    # Greedy construction
    selected = set(words)
    for c in candidate_pool:
        if time.time() > deadline:
            break
        if _is_compatible(c, selected):
            selected.add(c)

    # Local improvement: try random additions and swaps
    while time.time() < deadline:
        # Pick a random candidate not in selected
        c = rng.choice(candidate_pool)
        if c in selected:
            continue
        if _is_compatible(c, selected):
            selected.add(c)
            continue
        # Try to remove one word to make room
        sel_list = list(selected)
        for _ in range(min(5, len(sel_list))):  # limit swaps per iteration
            a = rng.choice(sel_list)
            temp = selected - {a}
            if _is_compatible(c, temp):
                selected = temp
                selected.add(c)
                break

    # Final greedy extension pass
    selected_list = list(selected)
    for c in candidate_pool:
        if time.time() > deadline:
            break
        if c not in selected:
            if _is_compatible(c, selected):
                selected.add(c)

    return list(selected)
