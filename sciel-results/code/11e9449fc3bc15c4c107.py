import time

_START = time.time()
_DEADLINE = 85.0  # Safety margin before 90s limit

MOD = 101
N = 10
_FULL = 1 << N  # 1024


def construct():
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Emergency fallback
        inv = [0] * (N + 1)
        for x in range(1, N + 1):
            inv[x] = pow(x, -1, MOD)
        first = {}
        for mask in range(1, _FULL):
            s = 0
            for i in range(N):
                if mask & (1 << i):
                    s = (s + inv[i + 1]) % MOD
            if s not in first:
                first[s] = mask
        rows = []
        for r in range(MOD):
            m = first.get(r, 1)
            denoms = [i + 1 for i in range(N) if m & (1 << i)]
            rows.append(denoms)
        return rows

    # 1. Compute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # 2. Precompute sum and popcount for all 1024 masks
    sums = [0] * _FULL
    pop = [0] * _FULL
    for mask in range(1, _FULL):
        lsb = mask & (-mask)
        idx = lsb.bit_length()  # 1-based index
        prev = mask ^ lsb
        sums[mask] = (sums[prev] + inv[idx]) % MOD
        pop[mask] = pop[prev] + 1

    # 3. Group masks by residue
    masks_by_residue = [[] for _ in range(MOD)]
    for mask in range(1, _FULL):
        r = sums[mask]
        masks_by_residue[r].append(mask)

    # 4. For each residue, find the minimum popcount mask that achieves it
    min_pop_for_residue = [100] * MOD
    best_mask_for_residue = [0] * MOD

    for mask in range(1, _FULL):
        r = sums[mask]
        p = pop[mask]
        if p < min_pop_for_residue[r]:
            min_pop_for_residue[r] = p
            best_mask_for_residue[r] = mask

    # 5. Determine the minimum k such that all residues have a mask of size <= k
    current_max = max(min_pop_for_residue)

    # 6. Try to reduce the maximum row length by finding alternative assignments
    # Use a greedy approach: for residues with the maximum popcount, try to find
    # an alternative mask with the same popcount but different denominators
    # This may help if we can reassign some residues to free up denominators

    # First, let's check if we can do better by trying all combinations of masks
    # for residues that have multiple options

    # Build a list of residues sorted by their minimum popcount (descending)
    residues_by_pop = sorted(range(MOD), key=lambda r: -min_pop_for_residue[r])

    # Try to assign masks to residues in a way that minimizes the maximum popcount
    # Start with the greedy assignment and try to improve it

    assigned_mask = best_mask_for_residue[:]
    assigned_pop = min_pop_for_residue[:]

    # Try to improve: for each residue with the maximum popcount, see if there's
    # another mask with the same or smaller popcount that doesn't conflict
    # (though there's no conflict constraint here, so this is just checking if
    # we can find a smaller mask)

    # Actually, the minimum popcount is already optimal per residue. The issue is
    # that we want to minimize the MAXIMUM across all residues.
    # So we need to check if there's a way to assign masks such that the max is
    # less than current_max.

    # Let's try a different approach: for each possible max value k from current_max
    # down to 1, check if we can assign masks to all residues such that each
    # residue gets a mask of size <= k.

    # For this, we can use a matching-like approach or just check feasibility.

    # First, let's see what the minimum possible max is
    # We know current_max is an upper bound. Let's try k = current_max - 1, etc.

    # For each k, check if every residue has at least one mask of size <= k
    for k in range(current_max - 1, 0, -1):
        feasible = True
        for r in range(MOD):
            if not any(pop[m] <= k for m in masks_by_residue[r]):
                feasible = False
                break
        if feasible:
            current_max = k
            break

    # Now, construct the final assignment using masks of size <= current_max
    rows = []
    for r in range(MOD):
        # Find a mask of size <= current_max for this residue
        # Prefer smaller masks
        best_mask = None
        best_size = current_max + 1
        for m in masks_by_residue[r]:
            if pop[m] < best_size:
                best_size = pop[m]
                best_mask = m
        mask = best_mask if best_mask is not None else best_mask_for_residue[r]
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                denoms.append(i + 1)
        rows.append(denoms)

    # Verify all rows are nonempty
    for r in range(MOD):
        if not rows[r]:
            for mask in range(1, _FULL):
                if sums[mask] == r:
                    denoms = [i + 1 for i in range(N) if mask & (1 << i)]
                    rows[r] = denoms
                    break

    return rows
