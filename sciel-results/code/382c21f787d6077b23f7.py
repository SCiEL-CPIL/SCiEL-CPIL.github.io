import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill

    n = 12
    m = 40
    N = 1 << n  # 4096

    # Precompute comparability: comp[a][b] = True if a and b are comparable (one strict subset of other)
    # Using numpy for speed
    a_exp = np.arange(N)[:, None]
    b_exp = np.arange(N)[None, :]
    subset_mask = (a_exp & b_exp) == a_exp
    # comp[i][j] = True if i is strict subset of j OR j is strict subset of i
    comp = (subset_mask & (a_exp != b_exp)) | (subset_mask.T & (a_exp != b_exp))
    # comp is symmetric, diagonal is False
    # Convert to flat list of bytes for faster access
    comp_flat = comp.flatten().tolist()
    # comp_flat[i * N + j] = 1 if comparable

    def comparable(a, b):
        return comp_flat[a * N + b] == 1

    def calc_score(masks):
        """Calculate number of comparable pairs in masks."""
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                if comparable(masks[i], masks[j]):
                    count += 1
        return count

    def best_subset_of_size(k, fixed_elements=None):
        """
        Generate a good family by taking subsets of a base set.
        Strategy: use all subsets of a 6-element set, pick 40 best ones.
        """
        if fixed_elements is None:
            fixed_elements = list(range(6))  # Use elements 0-5 as base
        base_mask = 0
        for e in fixed_elements:
            base_mask |= (1 << e)

        # Generate all subsets of base_mask
        subsets = []
        sub = 0
        while True:
            subsets.append(sub)
            sub = (sub - base_mask) & base_mask
            if sub == 0:
                break

        # We have 2^6 = 64 subsets. Pick 40 that maximize internal comparability.
        # Greedy: start with empty set and full set (most connected), then add most connected.
        # Actually, all subsets of a 6-set form a Boolean lattice B_6.
        # Total comparable pairs in B_6 = 3^6 - 2^6 = 729 - 64 = 665.
        # We pick 40 of 64. Which 40 maximize internal comparable pairs?
        # The "middle" layers have the most elements but least comparability.
        # Layers 0,1,2 and 4,5,6 have more comparability per element.
        # Let's just take all 64 and do local search to pick best 40.

        # Actually, let's use a different strategy:
        # Build a chain-like structure using multiple "spines"
        # Spine 1: subsets of {0,1,2,3,4,5}
        # Spine 2: subsets of {6,7,8,9,10,11}
        # Cross comparability between spines is limited.

        # Better: use a single 6-set base and pick 40 subsets optimally.
        # Let's do greedy: start with most "central" subsets.
        # In B_6, the subsets with sizes 2,3,4 are the most connected.
        # Let's take all subsets of sizes 2,3,4 from B_6: C(6,2)+C(6,3)+C(6,4) = 15+20+15 = 50.
        # Pick best 40 from these 50.

        # Or simpler: just take all 64 and let local search find best 40.
        # But we need exactly 40 distinct masks from 0..4095.

        # Let's try: take all subsets of {0,1,2,3,4,5} that have size in {2,3,4}
        # That's 50 sets. Remove 5 least useful ones.
        candidates = []
        for sub in subsets:
            sz = bin(sub).count('1')
            if sz in (2, 3, 4):
                candidates.append(sub)

        if len(candidates) < m:
            # Add more from other sizes
            for sub in subsets:
                if sub not in candidates:
                    candidates.append(sub)

        # Take first m candidates as initial family
        initial = candidates[:m]

        # Ensure they are distinct (they should be since subsets are unique)
        # Now do local search to optimize
        return initial

    def local_search(masks, time_limit):
        """
        Local search to maximize comparable pairs.
        masks: list of m distinct masks
        Returns best masks found.
        """
        score = calc_score(masks)
        best_score = score
        best_masks = masks[:]

        # Precompute current comparability counts for fast delta
        # comp_count[i] = number of j != i such that masks[i] comparable with masks[j]
        comp_count = [0] * m
        for i in range(m):
            for j in range(i + 1, m):
                if comparable(masks[i], masks[j]):
                    comp_count[i] += 1
                    comp_count[j] += 1

        T = 5.0
        T_min = 0.1
        alpha = 0.9995

        while time.time() < time_limit:
            # Pick random index to replace
            i = random.randint(0, m - 1)
            old_m = masks[i]

            # Pick new mask: bias towards masks close to existing ones in the lattice
            # Strategy: pick a random mask, or pick a mask that's a neighbor (add/remove one element)
            r = random.random()
            if r < 0.3:
                # Pick random mask
                new_m = random.randint(0, N - 1)
            elif r < 0.6:
                # Add one random element to old_m
                available = [b for b in range(n) if not (old_m >> b) & 1]
                if available:
                    e = random.choice(available)
                    new_m = old_m | (1 << e)
                else:
                    new_m = random.randint(0, N - 1)
            elif r < 0.9:
                # Remove one random element from old_m
                available = [b for b in range(n) if (old_m >> b) & 1]
                if available:
                    e = random.choice(available)
                    new_m = old_m & ~(1 << e)
                else:
                    new_m = random.randint(0, N - 1)
            else:
                # Pick from the same "base" region: masks that share many bits with some existing mask
                j = random.randint(0, m - 1)
                if j == i:
                    j = (j + 1) % m
                ref = masks[j]
                # Flip 1-3 random bits
                bits = list(range(n))
                random.shuffle(bits)
                new_m = ref
                for b in bits[:random.randint(1, 3)]:
                    new_m ^= (1 << b)

            # Check if new_m is already in masks
            if new_m in masks:
                continue

            # Calculate delta: change in score if we replace masks[i] with new_m
            # Remove old_m's contributions, add new_m's contributions
            delta = 0
            for j in range(m):
                if j == i:
                    continue
                old_comp = comparable(old_m, masks[j])
                new_comp = comparable(new_m, masks[j])
                if new_comp and not old_comp:
                    delta += 1
                elif old_comp and not new_comp:
                    delta -= 1

            if delta >= 0 or random.random() < np.exp(delta / T):
                # Accept
                masks[i] = new_m
                score += delta
                if score > best_score:
                    best_score = score
                    best_masks = masks[:]

            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min

        return best_masks, best_score

    # Generate multiple initial families and pick the best
    best_overall = None
    best_overall_score = -1

    # Strategy 1: Subsets of a 6-element set
    init1 = best_subset_of_size(3)
    # Ensure exactly 40 distinct
    seen = set()
    unique_init1 = []
    for x in init1:
        if x not in seen:
            seen.add(x)
            unique_init1.append(x)
    while len(unique_init1) < m:
        x = random.randint(0, N - 1)
        if x not in seen:
            seen.add(x)
            unique_init1.append(x)
    unique_init1 = unique_init1[:m]

    # Strategy 2: Chain-like structure
    # Build a chain using element 0, then branch
    init2 = []
    # Main chain: {0}, {0,1}, {0,1,2}, ..., {0,1,2,3,4,5,6,7,8,9,10,11}
    for k in range(1, 13):
        mask = (1 << k) - 1
        if mask not in init2:
            init2.append(mask)
    # Add branches: for each chain element, add variants
    for k in range(1, 13):
        base = (1 << k) - 1
        # Replace one element with a different one
        for replace_pos in range(k):
            for new_elem in range(k, 12):
                new_mask = base & ~(1 << replace_pos) | (1 << new_elem)
                if new_mask not in init2:
                    init2.append(new_mask)
                    if len(init2) >= m:
                        break
            if len(init2) >= m:
                break
        if len(init2) >= m:
            break

    # Ensure exactly 40 distinct
    seen2 = set()
    unique_init2 = []
    for x in init2:
        if x not in seen2:
            seen2.add(x)
            unique_init2.append(x)
    while len(unique_init2) < m:
        x = random.randint(0, N - 1)
        if x not in seen2:
            seen2.add(x)
            unique_init2.append(x)
    unique_init2 = unique_init2[:m]

    # Strategy 3: Two 6-sets
    init3 = []
    base1 = (1 << 6) - 1  # elements 0-5
    base2 = 0
    for e in range(6, 12):
        base2 |= (1 << e)
    # Subsets of base1 with size 2-4
    sub1 = 0
    while True:
        sz = bin(sub1).count('1')
        if sz in (2, 3, 4) and sub1 not in init3:
            init3.append(sub1)
        sub1 = (sub1 - base1) & base1
        if sub1 == 0:
            break
    # Subsets of base2 with size 2-4
    sub2 = 0
    while True:
        sz = bin(sub2).count('1')
        if sz in (2, 3, 4) and sub2 not in init3:
            init3.append(sub2)
        sub2 = (sub2 - base2) & base2
        if sub2 == 0:
            break

    seen3 = set()
    unique_init3 = []
    for x in init3:
        if x not in seen3:
            seen3.add(x)
            unique_init3.append(x)
    while len(unique_init3) < m:
        x = random.randint(0, N - 1)
        if x not in seen3:
            seen3.add(x)
            unique_init3.append(x)
    unique_init3 = unique_init3[:m]

    # Run local search on each initial family
    candidates = [unique_init1, unique_init2, unique_init3]
    time_per_candidate = (deadline - time.time()) / len(candidates)

    for idx, init in enumerate(candidates):
        remaining = deadline - time.time() - 2  # 2s buffer
        if remaining <= 0:
            break
        # Allocate time proportionally
        alloc = min(time_per_candidate, remaining)
        result, sc = local_search(init, start_time + time.time() - start_time + alloc)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = result

    # If we have time, do one more round of local search on the best
    remaining = deadline - time.time() - 2
    if remaining > 1 and best_overall is not None:
        result, sc = local_search(best_overall, start_time + time.time() - start_time + remaining)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall = result

    # Convert masks to lists of ints
    result = []
    for mask in best_overall:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    # Verify constraints
    assert len(result) == 40
    assert all(isinstance(x, list) for x in result)
    assert all(all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 11 for x in s) for s in result)
    assert all(len(set(s)) == len(s) for s in result)  # distinct within each set
    assert len(set(frozenset(s) for s in result)) == 40  # distinct sets

    return result
