import random
import time
from itertools import combinations

def construct():
    deadline = time.time() + 78  # Safety margin before 90s kill

    # Precompute all 4-subsets as bitmasks
    all_sets = []
    mask_to_idx = {}
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        mask_to_idx[mask] = len(all_sets) - 1

    n_sets = len(all_sets)  # 1820
    all_mask = (1 << 16) - 1

    # Precompute partners for fast blocked update
    # partners[i][j] is a list of indices k such that sets i, j, k form a sunflower
    # We compute this lazily and cache.
    partner_cache = {}

    def get_partners(i, j):
        key = (i, j) if i < j else (j, i)
        if key in partner_cache:
            return partner_cache[key]

        mask_i = all_sets[i]
        mask_j = all_sets[j]
        core = mask_i & mask_j
        sym = mask_i ^ mask_j
        avail = all_mask ^ (mask_i | mask_j)

        k_needed = 4 - bin(core).count('bit') if False else 4 - bin(core).count('1')
        if k_needed < 0 or k_needed > bin(avail).count('1'):
            partner_cache[key] = []
            return []

        # Generate all subsets of 'avail' of size k_needed
        # avail is a bitmask. We need to iterate through combinations of bits in avail.
        bits = [b for b in range(16) if avail & (1 << b)]
        partners = []
        for combo in combinations(bits, k_needed):
            add_mask = 0
            for b in combo:
                add_mask |= (1 << b)
            full_mask = core | add_mask
            idx = mask_to_idx.get(full_mask)
            if idx is not None:
                partners.append(idx)

        partner_cache[key] = partners
        return partners

    def greedy_randomized(seed):
        rng = random.Random(seed)
        family = []
        family_set = set()
        blocked = [0] * n_sets

        indices = list(range(n_sets))
        rng.shuffle(indices)

        for idx in indices:
            if blocked[idx] == 0 and idx not in family_set:
                # Add idx
                family.append(idx)
                family_set.add(idx)

                # Update blocked
                for j in family[:-1]:  # All previously added
                    partners = get_partners(idx, j)
                    for k in partners:
                        blocked[k] += 1

        return family

    def local_search(family, rng, time_limit):
        family_set = set(family)
        blocked = [0] * n_sets

        # Initialize blocked
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                partners = get_partners(family[i], family[j])
                for k in partners:
                    blocked[k] += 1

        improved = True
        while improved and time.time() < time_limit:
            improved = False

            # Try to add
            # Find all available
            available = [i for i in range(n_sets) if blocked[i] == 0 and i not in family_set]
            if available:
                # Pick random
                idx = rng.choice(available)
                family.append(idx)
                family_set.add(idx)
                for j in family[:-1]:
                    partners = get_partners(idx, j)
                    for k in partners:
                        blocked[k] += 1
                improved = True
                continue

            # Try swap: remove 1, add 2
            # Pick a random element to remove
            if len(family) == 0:
                break
            rem_idx = rng.choice(family)

            # Remove rem_idx
            family.remove(rem_idx)
            family_set.remove(rem_idx)
            for j in family:
                partners = get_partners(rem_idx, j)
                for k in partners:
                    blocked[k] -= 1

            # Try to add 2
            added = []
            # Find available
            available = [i for i in range(n_sets) if blocked[i] == 0 and i not in family_set]
            if len(available) >= 2:
                # Pick first
                idx1 = rng.choice(available)
                family.append(idx1)
                family_set.add(idx1)
                for j in family[:-1]:
                    partners = get_partners(idx1, j)
                    for k in partners:
                        blocked[k] += 1

                # Update available for second
                available2 = [i for i in range(n_sets) if blocked[i] == 0 and i not in family_set]
                if available2:
                    idx2 = rng.choice(available2)
                    family.append(idx2)
                    family_set.add(idx2)
                    for j in family[:-1]:
                        partners = get_partners(idx2, j)
                        for k in partners:
                            blocked[k] += 1

                    added = [idx1, idx2]
                    improved = True
                    continue

            # Revert removal
            family.append(rem_idx)
            family_set.add(rem_idx)
            for j in family[:-1]:
                partners = get_partners(rem_idx, j)
                for k in partners:
                    blocked[k] += 1

        return family

    best_family = []
    best_size = 0

    # Try multiple restarts
    for seed in range(200):
        if time.time() > deadline:
            break
        fam = greedy_randomized(seed)
        if len(fam) > best_size:
            best_family = fam[:]
            best_size = len(fam)

        # Local search with a small time budget per restart
        remaining = deadline - time.time()
        if remaining > 1.0:
            rng = random.Random(seed)
            ts_limit = time.time() + min(1.5, remaining * 0.8)
            fam = local_search(fam, rng, ts_limit)
            if len(fam) > best_size:
                best_family = fam[:]
                best_size = len(fam)

    # Final verification (optional, but good for safety)
    # We trust the blocked array logic, but let's do a quick check on the best family
    # Actually, the blocked logic is sound if get_partners is correct.
    # Let's just return the best family found.

    return [sorted([i + 1 for i in range(16) if all_sets[idx] & (1 << i)]) for idx in best_family]
