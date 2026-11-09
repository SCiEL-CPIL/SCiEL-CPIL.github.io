"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    MOD = 101
    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, MOD) for x in range(1, 11)]

    # Enumerate all nonempty subsets of {1..10}, compute (size, residue)
    # subsets indexed by mask 1..1023
    subset_size = [0] * (1 << 10)
    subset_res = [0] * (1 << 10)
    for mask in range(1, 1 << 10):
        s = 0
        sz = 0
        for x in range(1, 11):
            if mask & (1 << (x - 1)):
                s = (s + inv[x]) % MOD
                sz += 1
        subset_size[mask] = sz
        subset_res[mask] = s

    # For each k, the set of residues achievable by subsets of size <= k
    # We want smallest k such that all 101 residues are covered.
    # Precompute: for each residue, the minimum subset size that achieves it.
    min_size_for_res = [11] * MOD  # 11 = sentinel (unachievable)
    for mask in range(1, 1 << 10):
        r = subset_res[mask]
        sz = subset_size[mask]
        if sz < min_size_for_res[r]:
            min_size_for_res[r] = sz

    # The optimal max row length is max over residues of min_size_for_res[r]
    # (since we can reuse subsets, no conflict)
    best_k = max(min_size_for_res)

    # Build the covering: for each residue, pick a subset of size <= best_k
    # Prefer smallest size; if tie, any.
    # Build a list of subsets grouped by (residue, size) to pick quickly.
    # For each residue, store the best (smallest size) subset mask.
    best_mask_for_res = [0] * MOD
    for mask in range(1, 1 << 10):
        r = subset_res[mask]
        sz = subset_size[mask]
        if best_mask_for_res[r] == 0 or sz < subset_size[best_mask_for_res[r]]:
            best_mask_for_res[r] = mask

    # Convert mask to list of denominators
    def mask_to_list(mask):
        return [x for x in range(1, 11) if mask & (1 << (x - 1))]

    result = []
    for r in range(MOD):
        m = best_mask_for_res[r]
        assert m != 0, f"Residue {r} not covered"
        result.append(mask_to_list(m))

    # Safety: verify all residues covered and distinct within row
    for r in range(MOD):
        row = result[r]
        assert len(row) >= 1
        assert len(set(row)) == len(row)
        s = sum(inv[x] for x in row) % MOD
        assert s == r, f"Row {r}: sum {s} != {r}"

    return result
# EVOLVE-BLOCK-END
