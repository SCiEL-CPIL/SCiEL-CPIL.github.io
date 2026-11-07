import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 82  # safety margin before 90s kill

    # Bit-mask representation: mask i corresponds to subset {b : (i>>b)&1}
    # Comparability: mask a is subset of mask b iff (a & ~b) == 0
    # Comparable (strict) iff (a|b) != (a&b) and (a&~b)==0 or (b&~a)==0

    total_masks = 1 << n  # 4096

    # Precompute popcount (number of 1-bits) for all masks
    popcount = [0] * total_masks
    for mask in range(1, total_masks):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    # Precompute comparability matrix as a list of bitsets for fast counting
    # comp_bits[i] is a Python int whose j-th bit is 1 iff i and j are comparable
    comp_bits = [0] * total_masks
    for i in range(total_masks):
        bits = 0
        for j in range(i + 1, total_masks):
            if i != j and ((i & ~j) == 0 or (j & ~i) == 0):
                bits |= (1 << j)
        comp_bits[i] = bits

    # Count edges involving a given mask within a family (as a set of masks)
    def edge_count(mask, family_set):
        """Count how many masks in family_set are comparable to mask."""
        bits = comp_bits[mask]
        # Count bits set in bits that are also in family_set
        # We need to count intersection of bits with the family's mask-set
        # Since family is a set of ints, we iterate
        cnt = 0
        for m2 in family_set:
            if m2 != mask and (bits >> m2) & 1:
                cnt += 1
        return cnt

    # Faster: precompute for each mask a sorted list of comparable masks
    # Actually, let's use a different approach: maintain a bitset of the family
    # and use bit operations to count.
    # family_bits: a Python int where bit i is set iff mask i is in the family
    # Then edge_count(mask, family_bits) = popcount(comp_bits[mask] & family_bits) - (1 if mask in family else 0)

    def popcount_int(x):
        """Count set bits in a Python int."""
        cnt = 0
        while x:
            x &= x - 1
            cnt += 1
        return cnt

    def edge_count_bits(mask, family_bits):
        """Count edges from mask to other members of family represented as bitset."""
        inter = comp_bits[mask] & family_bits
        # Subtract self if mask is in family
        if (family_bits >> mask) & 1:
            inter &= ~(1 << mask)
        return popcount_int(inter)

    # Initial configuration: greedy construction
    # Start with all 4096 masks, greedily remove to get to 40
    # Better: start with a structured initial set

    # Greedy: pick masks one by one, each time picking the one that maximizes
    # comparable pairs with already-picked masks
    # This is O(40 * 4096) which is fine

    selected = []
    selected_set = set()
    selected_bits = 0  # bitset representation

    # Precompute for each mask, how many it's comparable to in a growing set
    # We'll do greedy: at each step, for each candidate not yet selected,
    # compute how many selected masks it's comparable to, pick the max

    # To speed up, maintain a running count of comparability to selected set
    # comp_to_selected[j] = number of selected masks comparable to j
    comp_to_selected = [0] * total_masks

    for step in range(m):
        best_mask = -1
        best_score = -1
        for j in range(total_masks):
            if j in selected_set:
                continue
            score = comp_to_selected[j]
            # Tie-break: prefer smaller sets (more likely to be subsets of others)
            if score > best_score or (score == best_score and (best_mask == -1 or popcount[j] < popcount[best_mask])):
                best_score = score
                best_mask = j
        selected.append(best_mask)
        selected_set.add(best_mask)
        selected_bits |= (1 << best_mask)
        # Update comp_to_selected for all other masks
        for j in range(total_masks):
            if j != best_mask and (comp_bits[best_mask] >> j) & 1:
                comp_to_selected[j] += 1

    # Compute initial score
    def total_score(family_bits):
        """Compute total comparable pairs in family represented as bitset."""
        # Sum over all masks in family of edge_count, divide by 2
        # But we can do: for each mask in family, count edges to masks with higher index
        # Simpler: use the comp_bits approach
        # Actually, let's just compute it directly for the initial config
        cnt = 0
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i + 1, len(sel_list)):
                b = sel_list[j]
                if (comp_bits[a] >> b) & 1:
                    cnt += 1
        return cnt

    current_score = total_score(selected_bits)

    # Simulated annealing with incremental delta evaluation
    # When we swap mask i (in family) with mask j (not in family):
    #   delta = edges(j, family \ {i}) - edges(i, family \ {i})
    #   where edges(x, S) = number of masks in S comparable to x

    # Build a list of selected masks for iteration
    selected_list = list(selected_set)

    # Temperature schedule
    t0 = 50.0
    t_min = 0.1
    # We'll do a fixed number of iterations or run until deadline
    rng = random.Random(42)

    best_score = current_score
    best_family_bits = selected_bits
    best_selected_set = set(selected_set)

    iteration = 0
    while time.time() < deadline:
        # Temperature: linear cooling
        # Estimate total iterations we can do
        # Each iteration: pick random i, j, compute delta in O(m) time
        # O(m) = O(40) per iteration, so we can do ~1M iterations in 80s
        # Let's use a cooling schedule over the remaining time
        remaining = deadline - time.time()
        if remaining <= 0:
            break

        # Simple cooling: temperature based on iteration count
        # We'll do up to ~500000 iterations
        max_iter = 500000
        if iteration >= max_iter:
            break

        temp = t0 * (t_min / t0) ** (iteration / max_iter)

        # Pick random element to swap out
        i_idx = rng.randrange(m)
        mask_out = selected_list[i_idx]

        # Pick random element to swap in
        mask_in = rng.randrange(total_masks)
        if mask_in in selected_set:
            iteration += 1
            continue

        # Compute delta
        # family without mask_out
        # edges(mask_in, family \ {mask_out}) - edges(mask_out, family \ {mask_out})

        # edges(mask, family \ {mask_out}):
        # = popcount(comp_bits[mask] & (selected_bits & ~(1 << mask_out)))
        #   minus self if mask is in family (mask_out is, mask_in is not)

        fam_without_out = selected_bits & ~(1 << mask_out)

        edges_in = popcount_int(comp_bits[mask_in] & fam_without_out)
        # mask_in is not in fam_without_out, so no self-subtraction needed

        edges_out = popcount_int(comp_bits[mask_out] & fam_without_out)
        # mask_out is not in fam_without_out (we removed it), so no self-subtraction

        delta = edges_in - edges_out

        if delta > 0 or (delta < 0 and rng.random() < pow(2.718281828, delta / temp)):
            # Accept the move
            current_score += delta
            selected_set.discard(mask_out)
            selected_set.add(mask_in)
            selected_bits = (selected_bits & ~(1 << mask_out)) | (1 << mask_in)
            selected_list[i_idx] = mask_in

            if current_score > best_score:
                best_score = current_score
                best_family_bits = selected_bits
                best_selected_set = set(selected_set)

        iteration += 1

        # Periodic check to avoid long pauses
        if iteration % 10000 == 0:
            pass  # no action needed, just a checkpoint

    # Use the best configuration found
    result = []
    for mask in best_selected_set:
        verts = [b for b in range(n) if (mask >> b) & 1]
        result.append(verts)

    # Ensure exactly m distinct lists of Python ints
    assert len(result) == m
    assert len(set(tuple(x) for x in result)) == m
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]

    return result
