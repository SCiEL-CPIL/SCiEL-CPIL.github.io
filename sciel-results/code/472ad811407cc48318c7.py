"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

P = {'n': 12, 'm': 40}

def construct():
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Precompute bit counts for all masks
    bit_counts = [0] * total_masks
    for mask in range(1, total_masks):
        bit_counts[mask] = bit_counts[mask & (mask - 1)] + 1

    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def calc_score(family):
        """Calculate number of comparable unordered pairs in family."""
        score = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if is_comparable(family[i], family[j]):
                    score += 1
        return score

    # Build a candidate pool: select a manageable subset of all 4096 masks
    # We'll work with a pool of at most ~200 masks to keep B&B tractable
    # Strategy: pick masks from each size, biased toward having good comparability structure

    random.seed(42)

    # Generate candidate pools from each size
    by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        by_size[bit_counts[mask]].append(mask)

    # Build a greedy initial chain-like solution
    # Start with empty set, then greedily add sets that are comparable with most existing
    def greedy_initial():
        family = []
        used = set()
        # Start with empty set (mask 0)
        family.append(0)
        used.add(0)

        while len(family) < m:
            best_mask = None
            best_comparables = -1
            # Try masks from sizes 1 to n, biased toward small sizes for chain structure
            for size in range(1, n + 1):
                if not by_size[size]:
                    continue
                # Sample a few candidates from this size
                candidates = by_size[size]
                # Prefer masks that contain or are contained by existing family members
                sample_size = min(len(candidates), 20)
                indices = random.sample(range(len(candidates)), sample_size)
                for idx in indices:
                    mask = candidates[idx]
                    if mask in used:
                        continue
                    # Count comparables with existing family
                    comparables = 0
                    for f in family:
                        if is_comparable(mask, f):
                            comparables += 1
                    # Also prefer masks that are "central" (mid-size) for future growth
                    if comparables > best_comparables:
                        best_comparables = comparables
                        best_mask = mask
            if best_mask is None:
                # Fallback: pick any unused mask
                for size in range(n + 1):
                    for mask in by_size[size]:
                        if mask not in used:
                            best_mask = mask
                            break
                    if best_mask is not None:
                        break
            if best_mask is None:
                break
            family.append(best_mask)
            used.add(best_mask)
        return family

    family = greedy_initial()
    current_score = calc_score(family)
    best_family = list(family)
    best_score = current_score

    # Build a candidate pool for B&B: include current family + promising additions
    # A mask is promising if it's comparable with many current family members
    def build_candidate_pool(family, pool_size=300):
        used = set(family)
        pool = list(family)
        # Score candidate masks by number of comparables with family
        candidates = []
        for mask in range(total_masks):
            if mask in used:
                continue
            comparables = 0
            for f in family:
                if is_comparable(mask, f):
                    comparables += 1
            if comparables > 0:
                candidates.append((comparables, mask))
        # Sort by comparables descending, take top pool_size
        candidates.sort(reverse=True)
        for _, mask in candidates[:pool_size]:
            pool.append(mask)
        return pool

    # Tomita-style branch-and-bound for maximum clique
    # We want to find a clique of size >= m in the comparability graph restricted to pool
    # Actually, we want to maximize comparable pairs, not necessarily find a full clique
    # But a clique of size m gives m*(m-1)/2 = 780 pairs, which is the theoretical max

    # Let's use a different approach: local search with smart moves
    # 1. Try to replace a mask with one that's comparable with more family members
    # 2. Try 2-swap moves
    # 3. Simulated annealing acceptance

    def local_search(family, deadline):
        current_score = calc_score(family)
        best_family = list(family)
        best_score = current_score

        # Precompute comparability matrix for the family
        m_local = len(family)

        no_improve = 0
        max_no_improve = 3000
        temp = 5.0  # Initial temperature for simulated annealing
        cooling = 0.999

        while time.time() < deadline and no_improve < max_no_improve:
            # Try a random swap
            idx = random.randint(0, m_local - 1)
            old_mask = family[idx]

            # Generate candidates: masks comparable with many family members
            # Sample from sizes that are adjacent to old_mask's size
            old_size = bit_counts[old_mask]
            candidate_sizes = set()
            for ds in range(-3, 4):
                s = old_size + ds
                if 0 <= s <= n:
                    candidate_sizes.add(s)

            new_mask = None
            for _ in range(10):  # Try up to 10 random candidates
                size = random.choice(list(candidate_sizes))
                if not by_size[size]:
                    continue
                candidate = random.choice(by_size[size])
                if candidate == old_mask or candidate in family:
                    continue
                # Check if comparable with most family members
                comparables = 0
                for j in range(m_local):
                    if j != idx and is_comparable(candidate, family[j]):
                        comparables += 1
                if comparables >= m_local - 5:  # At least comparable with most
                    new_mask = candidate
                    break

            if new_mask is None:
                no_improve += 1
                continue

            # Calculate delta
            delta = 0
            for j in range(m_local):
                if j == idx:
                    continue
                other = family[j]
                if is_comparable(old_mask, other):
                    delta -= 1
                if is_comparable(new_mask, other):
                    delta += 1

            # Simulated annealing acceptance
            if delta > 0:
                family[idx] = new_mask
                current_score += delta
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_family = list(family)
            else:
                # Accept with probability exp(delta/temp)
                if random.random() < math.exp(delta / temp):
                    family[idx] = new_mask
                    current_score += delta
                no_improve += 1

            temp *= cooling

        return best_family, best_score

    # Run local search with multiple restarts
    restart_deadline = deadline - 5.0
    num_restarts = 0
    while time.time() < restart_deadline:
        family, score = local_search(family, restart_deadline)
        if score > best_score:
            best_score = score
            best_family = list(family)
        num_restarts += 1
        # Occasional restart with fresh greedy init
        if num_restarts % 5 == 0:
            family = greedy_initial()

    # Final polish: exhaustive single-swap improvement
    final_family = list(best_family)
    final_score = calc_score(final_family)
    for _ in range(100):  # Up to 100 polish passes
        improved = False
        for idx in range(len(final_family)):
            old_mask = final_family[idx]
            best_delta = 0
            best_new = None
            # Try all masks from nearby sizes
            old_size = bit_counts[old_mask]
            for size in range(max(0, old_size - 2), min(n, old_size + 3) + 1):
                if not by_size[size]:
                    continue
                for candidate in by_size[size]:
                    if candidate == old_mask or candidate in final_family:
                        continue
                    delta = 0
                    for j in range(len(final_family)):
                        if j == idx:
                            continue
                        if is_comparable(old_mask, final_family[j]):
                            delta -= 1
                        if is_comparable(candidate, final_family[j]):
                            delta += 1
                    if delta > best_delta:
                        best_delta = delta
                        best_new = candidate
            if best_new is not None and best_delta > 0:
                final_family[idx] = best_new
                final_score += best_delta
                improved = True
        if not improved:
            break

    if final_score > best_score:
        best_score = final_score
        best_family = final_family

    # Convert best_family to lists of integers
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)

    return result
# EVOLVE-BLOCK-END
