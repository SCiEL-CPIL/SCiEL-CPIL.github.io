import time
import random

_start_time = time.time()

def construct():
    deadline = _start_time + 85
    n = 12
    m = 40
    total_masks = 1 << n  # 4096

    # Precompute comparability as bitsets for fast intersection
    # comp_bits[a] is a Python int with bit b set iff a and b are comparable
    comp_bits = [0] * total_masks
    for a in range(total_masks):
        for b in range(a + 1, total_masks):
            if (a | b) == a or (a | b) == b:
                comp_bits[a] |= (1 << b)
                comp_bits[b] |= (1 << a)

    # Precompute degree of each vertex in the full comparability graph
    full_degree = [bin(comp_bits[a]).count('1') for a in range(total_masks)]

    def count_comparables(family):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                if (comp_bits[family[i]] >> family[j]) & 1:
                    cnt += 1
        return cnt

    def greedy_initialization(seed=None):
        """Build an initial family of size m using greedy strategy."""
        rng = random.Random(seed)
        family = []
        used = set()

        # Start with a maximal chain
        for sz in range(n + 1):
            mask = (1 << sz) - 1
            family.append(mask)
            used.add(mask)

        # Greedily add remaining sets by maximizing comparability with existing
        remaining = [mask for mask in range(total_masks) if mask not in used]
        
        # Shuffle for diversity across restarts
        if seed is not None:
            rng.shuffle(remaining)
        
        for _ in range(m - len(family)):
            best_mask = -1
            best_deg = -1
            for c in remaining:
                deg = 0
                for s in family:
                    if (comp_bits[c] >> s) & 1:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_mask = c
            if best_mask == -1:
                break
            family.append(best_mask)
            used.add(best_mask)
            # Remove efficiently using a set-based approach
            remaining = [x for x in remaining if x != best_mask]

        # Ensure exactly m sets
        while len(family) < m:
            for mask in range(total_masks):
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                    break

        return family

    def hill_climb(family, time_budget):
        """Local search with efficient neighborhood evaluation."""
        current_score = count_comparables(family)
        best_family = list(family)
        best_score = current_score
        used = set(family)
        fam_set = set(family)
        
        # Precompute neighbor counts for faster evaluation
        def get_degree_in_family(mask, exclude_idx=None):
            """Count how many family members are comparable to mask."""
            deg = 0
            for j, s in enumerate(family):
                if exclude_idx is not None and j == exclude_idx:
                    continue
                if (comp_bits[mask] >> s) & 1:
                    deg += 1
            return deg

        improved = True
        max_iterations = 500
        iteration = 0
        while improved and iteration < max_iterations:
            if time.time() > time_budget:
                break
            improved = False
            iteration += 1

            for i in range(len(family)):
                if time.time() > time_budget:
                    break
                old_set = family[i]
                old_deg = get_degree_in_family(old_set, i)

                best_new = None
                best_gain = 0

                # Try all possible replacements
                for new_set in range(total_masks):
                    if new_set == old_set or new_set in fam_set:
                        continue
                    new_deg = get_degree_in_family(new_set, i)
                    gain = new_deg - old_deg
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_set

                if best_new is not None and best_gain > 0:
                    family[i] = best_new
                    fam_set.remove(old_set)
                    fam_set.add(best_new)
                    current_score += best_gain
                    improved = True

                    if current_score > best_score:
                        best_score = current_score
                        best_family = list(family)

                    break  # Restart scan from beginning after a change

        return best_family, best_score

    # Phase 1: Multi-restart greedy initialization + hill climbing
    best_family = None
    best_score = -1

    # Try several different initializations
    num_restarts = 5
    time_per_restart = (deadline - 5) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        seed = restart * 12345 + 67890
        family = greedy_initialization(seed)
        score = count_comparables(family)
        
        # Hill climb this candidate
        hc_deadline = min(deadline - 5, time.time() + time_per_restart * 0.7)
        hc_family, hc_score = hill_climb(family, hc_deadline)
        
        if hc_score > best_score:
            best_score = hc_score
            best_family = hc_family
    
    # Phase 2: Final polish with remaining time
    if time.time() < deadline - 3 and best_family is not None:
        best_family, best_score = hill_climb(best_family, deadline - 3)

    # Convert to lists of ints
    result = []
    for mask in best_family:
        lst = [int(i) for i in range(n) if (mask >> i) & 1]
        result.append(lst)

    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    assert len(res) == 40
    seen = set()
    for r in res:
        fr = frozenset(r)
        assert fr not in seen
        seen.add(fr)
        for v in r:
            assert isinstance(v, int)
            assert 0 <= v <= 11
    masks = []
    for r in res:
        m = 0
        for v in r:
            m |= (1 << v)
        masks.append(m)
    cnt = 0
    for i in range(40):
        for j in range(i+1, 40):
            a, b = masks[i], masks[j]
            if a | b == a or a | b == b:
                cnt += 1
    print(f"Time: {t1-t0:.2f}s, Comparables: {cnt}")
