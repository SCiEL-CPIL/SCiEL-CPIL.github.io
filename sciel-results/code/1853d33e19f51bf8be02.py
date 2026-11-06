import time
import numpy as np

N = 512
DEADLINE = 55.0  # safety margin before 60s limit

def _precompute_ap_pairs():
    """Precompute all (a, d) such that a, a+d, a+2d, a+3d are all in [1, N].
    Return arrays A (start) and D (diff), and for each element x, the list of (idx, role)
    where role indicates which position x occupies in the AP."""
    pairs = []
    for d in range(1, (N - 1) // 3 + 1):
        a_max = N - 3 * d
        for a in range(1, a_max + 1):
            pairs.append((a, d))
    A = np.array([p[0] - 1 for p in pairs], dtype=np.int32)  # 0-indexed
    D = np.array([p[1] for p in pairs], dtype=np.int32)
    return A, D

def _ap_conflict_count(mask, A, D):
    """Count number of 4-APs fully contained in the set represented by mask (0-indexed bool array).
    mask[i] is True if (i+1) is in the set."""
    if len(A) == 0:
        return 0
    # For each AP (a, d), check if all four positions are in mask
    pos1 = A
    pos2 = A + D
    pos3 = A + 2 * D
    pos4 = A + 3 * D
    count = np.sum(mask[pos1] & mask[pos2] & mask[pos3] & mask[pos4])
    return int(count)

def _ap_conflict_for_add(mask, x0, A, D):
    """x0 is 0-indexed. Count how many 4-APs would be completed if we add x0.
    An AP is completed if x0 is one of the four positions and the other three are already in mask."""
    if len(A) == 0:
        return 0
    # x0 appears as position 1: a = x0
    # x0 appears as position 2: a = x0 - d
    # x0 appears as position 3: a = x0 - 2d
    # x0 appears as position 4: a = x0 - 3d
    total = 0
    # As position 1: a = x0, need mask[a+d], mask[a+2d], mask[a+3d]
    d_vals = D
    a_vals = A
    # Find indices where A == x0
    idx1 = np.where(a_vals == x0)[0]
    if len(idx1) > 0:
        p2 = a_vals[idx1] + d_vals[idx1]
        p3 = a_vals[idx1] + 2 * d_vals[idx1]
        p4 = a_vals[idx1] + 3 * d_vals[idx1]
        valid = (p2 < N) & (p3 < N) & (p4 < N)
        if np.any(valid):
            total += np.sum(mask[p2[valid]] & mask[p3[valid]] & mask[p4[valid]])
    
    # As position 2: a = x0 - d, so d = x0 - a, need a >= 1 (a0 >= 0)
    # a0 = x0 - d, so d = x0 - a0. Need a0 >= 0 and d >= 1.
    # For each (a, d) in pairs, check if a + d == x0, i.e., d == x0 - a
    # a0 = x0 - d, so we need d = x0 - a0. Let's iterate differently.
    # For each d, a0 = x0 - d. Need a0 >= 0 and a0 + d = x0 (trivially true).
    # Need mask[a0], mask[a0+2d], mask[a0+3d]
    # a0 = x0 - d, so mask[x0-d], mask[x0+d], mask[x0+2d]
    # d ranges from 1 to min(x0, (N-1-x0)//1) but also a0+3d < N => x0-d+3d = x0+2d < N => d < (N-x0)/2
    max_d2 = min(x0, (N - 1 - x0) // 2)
    if max_d2 > 0:
        ds = np.arange(1, max_d2 + 1, dtype=np.int32)
        p1 = x0 - ds
        p3 = x0 + ds
        p4 = x0 + 2 * ds
        valid = (p1 >= 0) & (p3 < N) & (p4 < N)
        if np.any(valid):
            total += np.sum(mask[p1[valid]] & mask[p3[valid]] & mask[p4[valid]])
    
    # As position 3: a + 2d = x0, so a0 = x0 - 2d. Need mask[a0], mask[a0+d], mask[a0+3d]
    # mask[x0-2d], mask[x0-d], mask[x0+d]
    max_d3 = min(x0 // 2, (N - 1 - x0) // 1)
    # Also need a0+3d = x0+d < N => d < (N-x0)
    max_d3 = min(max_d3, N - 1 - x0)
    if max_d3 > 0:
        ds = np.arange(1, max_d3 + 1, dtype=np.int32)
        p1 = x0 - 2 * ds
        p2 = x0 - ds
        p4 = x0 + ds
        valid = (p1 >= 0) & (p2 >= 0) & (p4 < N)
        if np.any(valid):
            total += np.sum(mask[p1[valid]] & mask[p2[valid]] & mask[p4[valid]])
    
    # As position 4: a + 3d = x0, so a0 = x0 - 3d. Need mask[a0], mask[a0+d], mask[a0+2d]
    # mask[x0-3d], mask[x0-2d], mask[x0-d]
    max_d4 = min(x0 // 3, (N - 1 - x0) // 1)
    if max_d4 > 0:
        ds = np.arange(1, max_d4 + 1, dtype=np.int32)
        p1 = x0 - 3 * ds
        p2 = x0 - 2 * ds
        p3 = x0 - ds
        valid = (p1 >= 0) & (p2 >= 0) & (p3 >= 0)
        if np.any(valid):
            total += np.sum(mask[p1[valid]] & mask[p2[valid]] & mask[p3[valid]])
    
    return int(total)

def _ap_conflict_for_remove(mask, x0, A, D):
    """x0 is 0-indexed. Count how many 4-APs are broken if we remove x0.
    An AP is broken if x0 is one of the four positions and the other three are in mask."""
    # This is the same as _ap_conflict_for_add but with the current mask
    # (since removing x0 breaks APs that currently contain x0 and the other three)
    return _ap_conflict_for_add(mask, x0, A, D)

def _greedy_add(mask, A, D, start_time):
    """Try to add elements greedily. Return new mask."""
    added = True
    while added:
        if time.time() - start_time > DEADLINE:
            break
        added = False
        # Try adding elements in random order for diversity
        candidates = np.where(~mask)[0]
        if len(candidates) == 0:
            break
        np.random.shuffle(candidates)
        for x0 in candidates:
            if time.time() - start_time > DEADLINE:
                break
            conflict = _ap_conflict_for_add(mask, x0, A, D)
            if conflict == 0:
                mask[x0] = True
                added = True
    return mask

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # Precompute AP pairs
    A, D = _precompute_ap_pairs()
    
    # Initial set: base-4 digits 0-3
    mask = np.zeros(N, dtype=bool)
    for x in range(1, N + 1):
        # Check if all base-4 digits are in {0,1,2,3}
        # Since base 4 digits are always 0-3, this is always true!
        # Wait, that's not right. The classic construction is numbers whose base-4
        # digits are only 0 or 1. Let me use that.
        pass
    
    # Use the classic: numbers with only digits 0 and 1 in base 4
    mask = np.zeros(N, dtype=bool)
    for x in range(1, N + 1):
        ok = True
        v = x
        while v > 0:
            if v % 4 > 1:
                ok = False
                break
            v //= 4
        if ok:
            mask[x - 1] = True
    
    # Verify initial set is 4-AP free
    assert _ap_conflict_count(mask, A, D) == 0, "Initial set should be 4-AP free"
    
    # Greedy add to fill gaps
    mask = _greedy_add(mask, A, D, start_time)
    
    # Local search: add/remove/swap
    best_size = np.sum(mask)
    best_mask = mask.copy()
    
    while time.time() - start_time < DEADLINE - 2:
        improved = False
        in_mask = np.where(mask)[0]
        out_mask = np.where(~mask)[0]
        
        # Try single adds
        if len(out_mask) > 0:
            np.random.shuffle(out_mask)
            for x0 in out_mask:
                if time.time() - start_time > DEADLINE - 2:
                    break
                conflict = _ap_conflict_for_add(mask, x0, A, D)
                if conflict == 0:
                    mask[x0] = True
                    improved = True
                    break
            if improved:
                if np.sum(mask) > best_size:
                    best_size = np.sum(mask)
                    best_mask = mask.copy()
        
        # Try single removes (only if we can add more later, but for now just try)
        # Actually, removing doesn't directly help unless we can add more.
        # Try swaps: remove one, add another
        if not improved and len(in_mask) > 0 and len(out_mask) > 0:
            # Try random swaps
            for _ in range(min(200, len(in_mask) * len(out_mask) // 10 + 10)):
                if time.time() - start_time > DEADLINE - 2:
                    break
                ri = rng.randint(len(in_mask))
                ro = rng.randint(len(out_mask))
                x_remove = in_mask[ri]
                x_add = out_mask[ro]
                
                # Check if removing x_remove and adding x_add results in no APs
                # New mask: remove x_remove, add x_add
                # Count APs in new mask
                old_conflict = _ap_conflict_count(mask, A, D)  # should be 0
                # After swap, count APs
                mask[x_remove] = False
                mask[x_add] = True
                new_conflict = _ap_conflict_count(mask, A, D)
                if new_conflict == 0:
                    improved = True
                    if np.sum(mask) > best_size:
                        best_size = np.sum(mask)
                        best_mask = mask.copy()
                else:
                    mask[x_remove] = True
                    mask[x_add] = False
        
        if not improved:
            # Try more aggressive: random restarts or perturbations
            # Perturb: remove a random element, try to add others
            if len(in_mask) > 0:
                ri = rng.randint(len(in_mask))
                x_remove = in_mask[ri]
                mask[x_remove] = False
                # Try greedy add
                mask = _greedy_add(mask, A, D, start_time)
                if np.sum(mask) > best_size:
                    best_size = np.sum(mask)
                    best_mask = mask.copy()
                else:
                    # Restore
                    mask[x_remove] = True
    
    # Return best found
    result = best_mask.copy()
    # Final verification
    assert _ap_conflict_count(result, A, D) == 0, "Result must be 4-AP free"
    return sorted(np.where(result)[0] + 1)

# Quick test
if __name__ == "__main__":
    import time
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Size: {len(result)}, Time: {t1-t0:.2f}s")
    # Verify
    A, D = _precompute_ap_pairs()
    mask = np.zeros(N, dtype=bool)
    for x in result:
        mask[x-1] = True
    print(f"AP conflicts: {_ap_conflict_count(mask, A, D)}")
