"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    n = 16
    k = 4

    # Precompute all 4-subsets as bitmasks
    all_subsets = []
    for row in combinations(range(n), k):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_subsets.append(mask)

    num_subsets = len(all_subsets)

    # Precompute conflict table:
    # For each pair (i, j) of 4-sets, determine the set of "cores" (2-subsets of [n])
    # that would make a sunflower if a third set shares that core.
    # A sunflower with sets A, B, C exists iff there is a core I (2-subset) such that
    # I ⊆ A∩B, I ⊆ A∩C, I ⊆ B∩C.
    # So for pair (A, B), the dangerous cores are all 2-subsets of A∩B.
    # If we maintain a count of how many chosen sets contain each core,
    # then adding a new set M conflicts with pair (A,B) iff
    #   core_count[core] >= 2 for some core in M ∩ (A∩B)
    # i.e., some core in M is already in at least 2 chosen sets from A∩B.
    #
    # More precisely: for new set M, it forms a sunflower with existing A, B iff
    #   ∃ 2-subset I ⊆ (M ∩ A ∩ B)
    # which is equivalent to |M ∩ A ∩ B| >= 2.
    #
    # Fast check: for each existing pair (A,B) that both intersect M in >= 2 elements
    # with M... that's still O(n^2).
    #
    # Better: precompute for each pair of 4-sets (i,j) the bitmask of their intersection.
    # When adding M, for each existing set A, compute inter = M & A.
    # If popcount(inter) >= 2, then for each existing B with (B & inter) having popcount >= 2,
    # we have a sunflower. But this is still complex.
    #
    # Simplest fast approach: For each core (2-subset), maintain a list of chosen sets
    # containing that core. When adding M, for each core in M (there are C(4,2)=6 cores),
    # check if any pair in the list for that core has intersection >= 2 with each other
    # AND both contain that core. Actually, if two sets both contain core I, and M also
    # contains core I, then we need A∩B ⊇ I, which is already true. We need A∩M ⊇ I and B∩M ⊇ I,
    # which is true since M contains I. So the sunflower condition reduces to:
    #   ∃ 2-subset I ⊆ M such that at least 2 chosen sets contain I.
    #
    # Wait, that's not quite right. The sunflower condition is A∩B = A∩M = B∩M = I for some I.
    # If I is a 2-subset of M, and two chosen sets A, B both contain I, we still need
    # A∩B = I (not just ⊇ I), A∩M = I, B∩M = I.
    #
    # Let me re-derive: A, B, M form a sunflower iff A∩B = A∩M = B∩M.
    # Let I = A∩M. Then we need B∩M = I and A∩B = I.
    # So I ⊆ M, I ⊆ A, I ⊆ B, and A∩B = I, A∩M = I, B∩M = I.
    #
    # For a given M and a chosen set A: I = A∩M.
    # We need another chosen set B such that B∩M = I and A∩B = I.
    # B∩M = I means B contains I and B∩(M\I) = ∅.
    # A∩B = I means B contains I and B∩(A\I) = ∅.
    # So B must contain I and be disjoint from (M\I) ∪ (A\I).
    #
    # This is getting complex. Let's use a different fast approach:
    # Precompute for each pair of 4-sets whether they can be part of a sunflower
    # with a third set. Actually, let's just use the original O(n) check but with
    # numpy for speed, or accept the O(n^2) in the inner loop since n <= 250.

    # Actually, the original can_add is O(|chosen|^2) worst case per addition,
    # but in practice the groups are small. Let me optimize the inner loop.

    def popcount(x):
        return bin(x).count('1')

    def can_add_fast(mask_new, chosen_masks):
        """Check if mask_new can be added without forming a sunflower."""
        # For each chosen set A, compute I = A ∩ mask_new
        # Group by I. For each group with >= 2 sets, check if any pair has intersection = I
        # But we can do better: for each chosen A, if popcount(A & mask_new) < 2, skip
        # (since I must be at least... actually I can be 0,1,2,3,4)
        # Actually I can be any size 0-4. But for a sunflower we need A∩B = I too.
        # If I has size 0, then A∩B must be empty, A∩M empty, B∩M empty.
        # If I has size 1, then A∩B must be exactly that 1 element, etc.
        
        # Group chosen sets by their intersection with mask_new
        # Use a dict: intersection_mask -> list of chosen set indices
        groups = {}
        for idx, m in enumerate(chosen_masks):
            inter = m & mask_new
            if inter not in groups:
                groups[inter] = []
            groups[inter].append(idx)
        
        for inter, indices in groups.items():
            if len(indices) < 2:
                continue
            L = len(indices)
            # Check all pairs in this group
            for i in range(L):
                gi = chosen_masks[indices[i]]
                for j in range(i + 1, L):
                    gj = chosen_masks[indices[j]]
                    if (gi & gj) == inter:
                        return False
        return True

    best_family = []
    best_size = 0

    # Strategy 1: Deterministic Greedy (Lexicographic)
    chosen = []
    for mask in all_subsets:
        if time.time() > deadline:
            break
        if can_add_fast(mask, chosen):
            chosen.append(mask)

    if len(chosen) > best_size:
        best_size = len(chosen)
        best_family = chosen[:]

    # Local search polish: try to add more sets or swap
    def try_local_search(family, time_limit):
        """Try to improve family by adding or swapping sets."""
        family = family[:]
        chosen_set = set(family)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any set not in family
            for mask in all_subsets:
                if mask in chosen_set:
                    continue
                if can_add_fast(mask, family):
                    family.append(mask)
                    chosen_set.add(mask)
                    improved = True
            # Try swaps: remove one, add one
            if time.time() < time_limit:
                for i in range(len(family)):
                    if time.time() > time_limit:
                        break
                    saved = family.pop(i)
                    chosen_set.discard(saved)
                    # Try to add two sets to compensate
                    added = []
                    for mask in all_subsets:
                        if mask in chosen_set or mask == saved:
                            continue
                        if can_add_fast(mask, family + added):
                            added.append(mask)
                            chosen_set.add(mask)
                            if len(added) >= 2:
                                break
                    if len(added) >= 1:
                        family.extend(added)
                        improved = True
                    else:
                        family.insert(i, saved)
                        chosen_set.add(saved)
        return family

    # Apply local search to best family
    if time.time() < deadline - 5.0:
        polished = try_local_search(best_family, deadline - 5.0)
        if len(polished) > best_size:
            best_size = len(polished)
            best_family = polished

    # Strategy 2: Randomized Greedy Restarts with local search
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break

        random.seed(12345 + restart)
        shuffled = all_subsets[:]
        random.shuffle(shuffled)

        chosen = []
        for mask in shuffled:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(mask, chosen):
                chosen.append(mask)

        if len(chosen) > best_size:
            # Apply local search
            if time.time() < deadline - 5.0:
                polished = try_local_search(chosen, deadline - 5.0)
                if len(polished) > best_size:
                    best_size = len(polished)
                    best_family = polished
            else:
                best_size = len(chosen)
                best_family = chosen[:]

            if best_size >= 250:
                break

    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))

    return result
# EVOLVE-BLOCK-END
