import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    n = 12
    m = 40
    total_masks = 1 << n  # 4096 possible subsets

    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    # Count comparable pairs in a family (list of masks)
    def count_comparable(family):
        count = 0
        k = len(family)
        for i in range(k):
            a = family[i]
            for j in range(i + 1, k):
                b = family[j]
                if (a & b) == a and a != b:
                    count += 1
                elif (b & a) == b and b != a:
                    count += 1
        return count

    # Generate initial solution: use a greedy chain-based approach
    random.seed(42)

    # Start with a chain of all sizes 0-12 (13 sets)
    chain = []
    for size in range(13):
        mask = (1 << size) - 1  # {0, 1, ..., size-1}
        chain.append(mask)

    # Now add 27 more sets greedily
    family = list(chain)

    # Candidate pools: generate random masks
    candidates = list(range(total_masks))
    random.shuffle(candidates)

    # Remove masks already in family
    family_set = set(family)
    candidates = [c for c in candidates if c not in family_set]

    # Greedily add masks that maximize comparable pairs
    for _ in range(m - len(family)):
        best_mask = None
        best_gain = -1
        # Search more candidates for better initial solution
        for c in candidates[:500]:
            # Compute gain if we add c
            gain = 0
            for f in family:
                if (c & f) == c and c != f:
                    gain += 1
                elif (f & c) == f and f != c:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = c
        if best_mask is not None:
            family.append(best_mask)
            family_set.add(best_mask)
            candidates.remove(best_mask)

    # Local search: try to replace one mask with another to improve
    best_score = count_comparable(family)
    best_family = list(family)

    # Compute gain/loss efficiently using precomputed comparability
    def compute_net_gain(fam, comp, idx, new_mask, fam_set):
        """Compute net gain of replacing fam[idx] with new_mask"""
        if new_mask in fam_set:
            return -1000000  # Invalid: duplicate
        k = len(fam)
        gain = 0
        loss = 0
        old_mask = fam[idx]
        for j in range(k):
            if j == idx:
                continue
            f = fam[j]
            # Gain: new_mask comparable with f?
            if (new_mask & f) == new_mask and new_mask != f:
                gain += 1
            elif (f & new_mask) == f and f != new_mask:
                gain += 1
            # Loss: old_mask comparable with f?
            if (old_mask & f) == old_mask and old_mask != f:
                loss += 1
            elif (f & old_mask) == f and f != old_mask:
                loss += 1
        return gain - loss

    # Generate candidate list for a given old_mask, excluding masks in exclude_set
    def generate_candidates(old_mask, exclude_set, n_random=30):
        candidate_list = []
        # Add some random candidates
        for _ in range(n_random):
            c = random.randint(0, total_masks - 1)
            if c not in exclude_set:
                candidate_list.append(c)
        # Add structured candidates: ALL subsets of old_mask
        sub = old_mask
        while sub:
            if sub not in exclude_set and sub != old_mask:
                candidate_list.append(sub)
            sub = (sub - 1) & old_mask
        # Include empty set if old_mask is not empty
        if old_mask != 0 and 0 not in exclude_set:
            candidate_list.append(0)
        # Supersets of old_mask: enumerate ALL supersets
        # A superset of old_mask is old_mask | x where x is a subset of (~old_mask) & (total_masks - 1)
        complement = (~old_mask) & (total_masks - 1)
        sup = complement
        while sup:
            new_mask = old_mask | sup
            if new_mask not in exclude_set and new_mask != old_mask:
                candidate_list.append(new_mask)
            sup = (sup - 1) & complement
        # Also add old_mask | complement (the full superset) if not already added
        full_sup = old_mask | complement
        if full_sup not in exclude_set and full_sup != old_mask:
            candidate_list.append(full_sup)
        # Add masks that differ by exactly one bit (Hamming distance 1)
        for bit in range(n):
            new_mask = old_mask ^ (1 << bit)
            if new_mask not in exclude_set:
                candidate_list.append(new_mask)
        return candidate_list

    # Try multiple random restarts if time allows
    restarts = 0
    while time.time() < deadline - 8.0:
        restarts += 1
        # Create a perturbed version of best_family
        family = list(best_family)

        # Randomly remove 1-2 sets and replace with new ones (less aggressive)
        num_replace = random.randint(1, 2)
        to_replace = random.sample(range(m), num_replace)
        family_set = set(family)

        for idx in to_replace:
            # Find a good replacement using systematic search over candidates
            best_mask = None
            best_gain = -1
            # Use a mix of random and structured candidates
            # First try masks close in size to the one being replaced
            old_mask = family[idx]
            old_size = popcount[old_mask]

            # Generate candidates: all masks not in family
            # Prioritize masks with similar popcount or that are subsets/supersets
            candidate_list = generate_candidates(old_mask, family_set)

            # Try each candidate
            for c in candidate_list:
                if c in family_set:
                    continue
                net_gain = compute_net_gain(family, None, idx, c, family_set)
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_mask = c

            if best_mask is not None and best_gain > 0:
                family_set.remove(old_mask)
                family[idx] = best_mask
                family_set.add(best_mask)

        score = count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = list(family)

    # Final polish: try single replacements with exhaustive search over ALL possible masks
    while time.time() < deadline - 1.0:
        improved = False
        family_set = set(best_family)
        for i in range(m):
            # Try to replace best_family[i] with a better mask - check ALL masks
            best_mask = None
            best_gain = -1
            old_mask = best_family[i]

            # Exhaustive: try every mask not in family
            for c in range(total_masks):
                if c in family_set:
                    continue
                net_gain = compute_net_gain(best_family, None, i, c, family_set)
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_mask = c

            if best_mask is not None and best_gain > 0:
                family_set.remove(old_mask)
                best_family[i] = best_mask
                family_set.add(best_mask)
                improved = True
                break
        if not improved:
            break

    # 2-swap local search: try replacing two sets simultaneously
    # This can escape local optima that single-replacement moves can't
    while time.time() < deadline - 0.5:
        improved = False
        family_set = set(best_family)

        # Try ALL pairs of indices systematically (C(40,2) = 780 pairs)
        # For each pair, use structured candidates (subsets/supersets)
        for i in range(m):
            if time.time() > deadline - 0.5:
                break
            for j in range(i + 1, m):
                if time.time() > deadline - 0.5:
                    break

                old_i = best_family[i]
                old_j = best_family[j]

                # Generate candidates: subsets and supersets of each
                # For position i: candidates are subsets/supersets of old_i not in family
                cand_i = []
                # Subsets of old_i
                sub = old_i
                while sub:
                    if sub not in family_set and sub != old_j:
                        cand_i.append(sub)
                    sub = (sub - 1) & old_i
                if old_i != 0 and 0 not in family_set and 0 != old_j:
                    cand_i.append(0)
                # Supersets of old_i (limited to avoid explosion)
                complement_i = (~old_i) & (total_masks - 1)
                sup = complement_i
                while sup:
                    new_mask = old_i | sup
                    if new_mask not in family_set and new_mask != old_j:
                        cand_i.append(new_mask)
                    sup = (sup - 1) & complement_i
                full_sup_i = old_i | complement_i
                if full_sup_i not in family_set and full_sup_i != old_j:
                    cand_i.append(full_sup_i)

                # For position j: candidates are subsets/supersets of old_j not in family
                cand_j = []
                sub = old_j
                while sub:
                    if sub not in family_set and sub != old_i:
                        cand_j.append(sub)
                    sub = (sub - 1) & old_j
                if old_j != 0 and 0 not in family_set and 0 != old_i:
                    cand_j.append(0)
                complement_j = (~old_j) & (total_masks - 1)
                sup = complement_j
                while sup:
                    new_mask = old_j | sup
                    if new_mask not in family_set and new_mask != old_i:
                        cand_j.append(new_mask)
                    sup = (sup - 1) & complement_j
                full_sup_j = old_j | complement_j
                if full_sup_j not in family_set and full_sup_j != old_i:
                    cand_j.append(full_sup_j)

                # Limit candidate lists to avoid O(n^2) explosion
                cand_i = cand_i[:50]
                cand_j = cand_j[:50]

                best_pair = None
                best_pair_gain = -1

                # Try combinations of candidates for i and j
                for ci in cand_i:
                    if ci == old_j or ci in family_set:
                        continue
                    for cj in cand_j:
                        if cj == old_i or cj in family_set or cj == ci:
                            continue

                        # Compute net gain for 2-swap
                        gain = 0
                        loss = 0
                        for k in range(m):
                            if k == i or k == j:
                                continue
                            f = best_family[k]
                            if (ci & f) == ci and ci != f:
                                gain += 1
                            elif (f & ci) == f and f != ci:
                                gain += 1
                            if (cj & f) == cj and cj != f:
                                gain += 1
                            elif (f & cj) == f and f != cj:
                                gain += 1
                            if (old_i & f) == old_i and old_i != f:
                                loss += 1
                            elif (f & old_i) == f and f != old_i:
                                loss += 1
                            if (old_j & f) == old_j and old_j != f:
                                loss += 1
                            elif (f & old_j) == f and f != old_j:
                                loss += 1
                        if (old_i & old_j) == old_i and old_i != old_j:
                            loss += 1
                        elif (old_j & old_i) == old_j and old_j != old_i:
                            loss += 1
                        if (ci & cj) == ci and ci != cj:
                            gain += 1
                        elif (cj & ci) == cj and cj != ci:
                            gain += 1

                        net_gain = gain - loss
                        if net_gain > best_pair_gain:
                            best_pair_gain = net_gain
                            best_pair = (ci, cj)

                if best_pair is not None and best_pair_gain > 0:
                    ci, cj = best_pair
                    family_set.discard(old_i)
                    family_set.discard(old_j)
                    best_family[i] = ci
                    best_family[j] = cj
                    family_set.add(ci)
                    family_set.add(cj)
                    improved = True

        if not improved:
            break

    # Convert masks to lists of integers
    result = []
    for mask in best_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)

    # Ensure types are correct: lists of ints
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]

    return result
