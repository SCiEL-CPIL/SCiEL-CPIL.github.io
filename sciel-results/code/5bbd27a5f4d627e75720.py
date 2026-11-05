import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        gcd_tab[i][i] = moduli[i]
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Precompute conflict masks: for each pair (i, j) and each residue r of i,
    # the set of residues of j that conflict with (i, r).
    # conflict_masks[i][j][r] is a bitmask over residues of j.
    conflict_masks = [[None] * n for _ in range(n)]
    for i in range(n):
        mi = moduli[i]
        for j in range(n):
            if i == j:
                conflict_masks[i][j] = [0] * mi
                continue
            mj = moduli[j]
            g = gcd_tab[i][j]
            masks = [0] * mi
            for r in range(mi):
                mask = 0
                for s in range(mj):
                    if (r - s) % g == 0:
                        mask |= (1 << s)
                masks[r] = mask
            conflict_masks[i][j] = masks

    # State for backtracking
    blocked = [0] * n  # bitmask of blocked residues for each modulus
    selected = []  # list of (idx, residue)
    selected_idx_set = set()
    best_solution = []
    best_size = 0

    def backtrack():
        nonlocal best_solution, best_size

        if time.time() > deadline - 3:
            return

        if len(selected) > best_size:
            best_size = len(selected)
            best_solution = list(selected)

        if best_size == n:
            return

        # MRV: pick the unselected modulus with the fewest available residues
        min_avail = n + 1
        best_idx = -1

        for idx in range(n):
            if idx in selected_idx_set:
                continue
            m = moduli[idx]
            mask = blocked[idx]
            avail = m - (mask & ((1 << m) - 1)).bit_count()
            if avail < min_avail:
                min_avail = avail
                best_idx = idx
                if min_avail == 0:
                    break

        if best_idx == -1:
            return
        if min_avail == 0:
            return

        m = moduli[best_idx]
        mask = blocked[best_idx]

        # Try residues in order of least blocking
        candidates = []
        for r in range(m):
            if (mask >> r) & 1:
                continue
            # Count how many new residues would be blocked
            total_block = 0
            for j in range(n):
                if j == best_idx or j in selected_idx_set:
                    continue
                new_mask = blocked[j] | conflict_masks[best_idx][j][r]
                added = (new_mask & ~blocked[j] & ((1 << moduli[j]) - 1)).bit_count()
                total_block += added
            candidates.append((total_block, r))

        candidates.sort()

        for _, r in candidates:
            if time.time() > deadline - 3:
                return

            selected.append((best_idx, r))
            selected_idx_set.add(best_idx)

            changed = []
            for j in range(n):
                if j == best_idx or j in selected_idx_set:
                    continue
                old_mask = blocked[j]
                new_mask = old_mask | conflict_masks[best_idx][j][r]
                blocked[j] = new_mask
                changed.append((j, old_mask))

            backtrack()

            selected.pop()
            selected_idx_set.remove(best_idx)
            for j, old_mask in changed:
                blocked[j] = old_mask

            if best_size == n:
                return

    backtrack()

    if best_solution:
        return [[r, moduli[idx]] for (idx, r) in best_solution]

    # Fallback: return at least one class
    return [[0, 2]]
