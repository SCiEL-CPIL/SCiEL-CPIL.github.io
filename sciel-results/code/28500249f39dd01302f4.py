"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def is_sidon(subset):
        """Check if subset has all pair sums unique (including diagonals)."""
        sums = set()
        n = len(subset)
        for i in range(n):
            for j in range(i, n):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def max_sidon_size(arr):
        """Find the largest Sidon subset size in arr."""
        n = len(arr)
        # Try from largest to smallest
        for size in range(n, 0, -1):
            for subset in itertools.combinations(arr, size):
                if is_sidon(subset):
                    return size
        return 1

    def collision_score(arr):
        """Count number of pair-sum collisions (more = better for our purpose)."""
        sums = Counter()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                sums[arr[i] + arr[j]] += 1
        # Count how many sums appear more than once
        collisions = 0
        for count in sums.values():
            if count > 1:
                collisions += count - 1
        return collisions

    # Strategy: Greedy construction to maximize pair-sum collisions
    def greedy_construct(seed_offset=0):
        """Build a 12-element set greedily to maximize collisions."""
        chosen = []
        used = set()
        all_vals = list(range(seed_offset, min(seed_offset + 129, 129)))

        for _ in range(12):
            best_val = None
            best_collisions = -1
            for v in all_vals:
                if v in used:
                    continue
                # Count collisions this new element would create
                coll = 0
                for u in chosen:
                    s = v + u
                    # Check if this sum already exists among pairs in chosen
                    for i in range(len(chosen)):
                        for j in range(i, len(chosen)):
                            if chosen[i] + chosen[j] == s:
                                coll += 1
                                break
                if coll > best_collisions:
                    best_collisions = coll
                    best_val = v
            if best_val is not None:
                chosen.append(best_val)
                used.add(best_val)
        return sorted(chosen)

    # Generate multiple greedy constructions with different starting offsets
    candidates = []
    for offset in range(0, 20):
        cand = greedy_construct(offset)
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            candidates.append(cand)

    # Also try structured patterns
    # Pairs summing to same constant - very effective
    for c in [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 128]:
        # Find 6 pairs that sum to c, all distinct, in [0,128]
        pairs = []
        for a in range(0, c // 2 + 1):
            b = c - a
            if a != b and 0 <= a <= 128 and 0 <= b <= 128:
                pairs.append((a, b))
        if len(pairs) >= 6:
            # Take first 6 pairs
            cand = []
            for a, b in pairs[:6]:
                cand.append(a)
                cand.append(b)
            if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                candidates.append(sorted(cand))

    # Triple-sum collisions: three pairs all sum to same value
    for c in [30, 40, 50, 60, 70, 80]:
        # Try to find 4 pairs summing to c
        pairs = []
        used_in_pairs = set()
        for a in range(0, c // 2):
            b = c - a
            if a not in used_in_pairs and b not in used_in_pairs and 0 <= b <= 128:
                pairs.append((a, b))
                used_in_pairs.add(a)
                used_in_pairs.add(b)
            if len(pairs) >= 4:
                break
        if len(pairs) >= 4:
            cand = []
            for a, b in pairs[:4]:
                cand.append(a)
                cand.append(b)
            if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
                candidates.append(sorted(cand))

    # Best known so far
    best_candidate = None
    best_score = 12  # worst case

    for cand in candidates:
        if len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        s = max_sidon_size(cand)
        if s < best_score:
            best_score = s
            best_candidate = cand[:]

    # Local search: try swapping elements to reduce max Sidon size
    if best_candidate is None:
        best_candidate = list(range(12))
        best_score = 12

    # Try random perturbations with collision-aware scoring
    random.seed(42)
    for iteration in range(20000):
        if time.time() > deadline:
            break

        # Start from best and make a small change
        cand = best_candidate[:]

        # Randomly swap one element with another value in [0,128]
        idx = random.randint(0, 11)
        old_val = cand[idx]
        new_val = random.randint(0, 128)

        if new_val in cand and new_val != old_val:
            continue

        cand[idx] = new_val

        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue

        # Quick filter: check collision score before expensive exact check
        new_coll = collision_score(cand)
        old_coll = collision_score(best_candidate)

        # Only do expensive check if collision score improved or is close
        if new_coll >= old_coll or random.random() < 0.3:
            s = max_sidon_size(cand)
            if s < best_score:
                best_score = s
                best_candidate = cand[:]

    # Final polish: try all single-element swaps from best
    for idx in range(12):
        if time.time() > deadline:
            break
        old_val = best_candidate[idx]
        for new_val in range(0, 129):
            if new_val == old_val or new_val in best_candidate:
                continue
            cand = best_candidate[:]
            cand[idx] = new_val
            s = max_sidon_size(cand)
            if s < best_score:
                best_score = s
                best_candidate = cand[:]

    return best_candidate
# EVOLVE-BLOCK-END
