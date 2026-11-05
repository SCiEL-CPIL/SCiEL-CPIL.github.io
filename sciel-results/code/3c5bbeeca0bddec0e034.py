import time

_START = time.time()
_DEADLINE = 55.0  # Safety margin before 60s limit

MOD = 101
N = 10
_FULL = 1 << N  # 1024


def construct():
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Emergency fallback: use first valid mask per residue
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
            m = first.get(r, 1)  # mask=1 means [1], sum=1
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

    # 3. For each residue, find the minimum popcount mask that achieves it
    # This is the greedy optimum since denominators can recur across rows
    min_pop_for_residue = [100] * MOD
    best_mask_for_residue = [0] * MOD

    for mask in range(1, _FULL):
        r = sums[mask]
        p = pop[mask]
        if p < min_pop_for_residue[r]:
            min_pop_for_residue[r] = p
            best_mask_for_residue[r] = mask

    # 4. Local search refinement: try to reduce max row length
    # Current max is max(min_pop_for_residue)
    current_max = max(min_pop_for_residue)

    # Try to see if we can do better by reassigning some residues
    # to different masks (not just the minimum popcount one)
    # For each residue, collect all masks that achieve it
    masks_by_residue = [[] for _ in range(MOD)]
    for mask in range(1, _FULL):
        r = sums[mask]
        masks_by_residue[r].append(mask)

    # Sort masks for each residue by popcount
    for r in range(MOD):
        masks_by_residue[r].sort(key=lambda m: pop[m])

    # Local search: try to find an assignment where max popcount < current_max
    # Use a simple iterative improvement
    # Start with greedy assignment
    assignment = [best_mask_for_residue[r] for r in range(MOD)]
    assigned_masks = set(assignment)

    # Try to improve: for residues with popcount == current_max,
    # see if there's another mask with smaller popcount that's available
    # (But since denominators can recur, all masks are always available)
    # So the greedy is already optimal.

    # The only way to improve would be if we misidentified the minimum.
    # Let's verify and construct rows.

    # 5. Construct rows from the optimal assignment
    rows = []
    for r in range(MOD):
        mask = best_mask_for_residue[r]
        denoms = []
        for i in range(N):
            if mask & (1 << i):
                denoms.append(i + 1)
        rows.append(denoms)

    # Verify all rows are nonempty
    for r in range(MOD):
        if not rows[r]:
            # Fallback: use mask with smallest popcount that sums to r
            # This shouldn't happen, but handle it
            for mask in range(1, _FULL):
                if sums[mask] == r:
                    denoms = [i + 1 for i in range(N) if mask & (1 << i)]
                    rows[r] = denoms
                    break

    return rows
