"""Program: vectorized Hall feasibility + exact bipartite matching verification."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from functools import lru_cache

N = 12
MAX_M = 1000000
MAX_L = 144
LCM_12 = 27720
FULL_MASK = (1 << N) - 1  # 0xFFF


def _exact_has_matching(m, L):
    """Exact bipartite matching: is there a perfect matching in (m, m+L]?
    Returns True iff there exist distinct b_1..b_12 in (m, m+L] with i | b_i."""
    # For each i in 1..12, the set of candidate positions j (0-indexed in (m, m+L])
    # such that (i) divides (m+1+j).
    # Build adjacency: adj[i] = set of j values
    # Then check if a perfect matching exists via augmenting paths.
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        start = m + 1
        q = (start + i - 1) // i
        v = q * i
        while v <= m + L:
            j = v - m - 1
            adj[i - 1].append(j)
            v += i

    # Hungarian / augmenting path matching
    # match[j] = which i is matched to position j (or -1)
    match = [-1] * L

    def try_augment(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match[j] == -1 or try_augment(match[j], visited):
                match[j] = i
                return True
        return False

    for i in range(N):
        visited = [False] * L
        if not try_augment(i, visited):
            return False
    return True


def _compute_f_exact(m):
    """Compute f(N, m) exactly via binary search with bipartite matching."""
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _exact_has_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def _vectorized_hall_feasibility(residues, L):
    """For a given L, check Hall's condition for all residues simultaneously.
    Returns a boolean array of length len(residues)."""
    R = len(residues)
    # For each residue r and position j in 0..L-1, compute divisor mask
    # div_mask[r, j] = bitmask of i in 1..12 such that i | (r+1+j)
    # This is equivalent to: i | (r + 1 + j)
    # Since we work mod LCM_12, and L <= 144, we can compute directly.

    # Build a (R, L) array of divisor masks
    div_masks = np.zeros((R, L), dtype=np.uint32)
    for i in range(1, N + 1):
        bit = 1 << (i - 1)
        # For residue r and position j: i | (r + 1 + j)
        # r + 1 + j ≡ 0 (mod i)  =>  j ≡ -r-1 (mod i)
        # For each r, the valid j values are j = ((-r-1) mod i) + k*i for k>=0, j < L
        # Vectorize over r:
        # j_start[r] = ((-r - 1) mod i)
        j_starts = np.mod(-residues - 1, i)  # shape (R,)
        # For each r, valid j: j_starts[r], j_starts[r]+i, j_starts[r]+2i, ... < L
        # We can do: for each k such that j_starts + k*i < L
        max_k = (L - 1 - j_starts) // i  # max k for each r
        # Create indices
        ks = np.arange(L // i + 1)  # possible k values
        # j_vals = j_starts[:, None] + ks[None, :] * i
        # Valid where j_vals < L
        j_vals = j_starts[:, None] + ks[None, :] * i
        valid = j_vals < L
        # Scatter: for each (r, k) where valid, set div_masks[r, j_vals[r,k]] |= bit
        # Use np.where to get indices
        r_idx, j_idx = np.where(valid)
        div_masks[r_idx, j_idx] |= bit

    # Now for each residue r, we have L positions with divisor masks.
    # Apply zeta transform per row to get H[r, t] = count of positions with mask subset of t
    # H shape: (R, 4096)
    H = np.zeros((R, 1 << N), dtype=np.int32)
    # First, frequency count per row
    # For each r, freq[r, mask] = count of j with div_masks[r,j] == mask
    # Then zeta transform
    # Vectorized: use np.add.at for frequency
    for r in range(R):
        freq = np.bincount(div_masks[r], minlength=1 << N).astype(np.int32)
        H[r] = freq

    # Zeta transform across all rows simultaneously
    for i in range(N):
        bit = 1 << i
        H[:, bit::2 * bit] += H[:, 0::2 * bit]

    # Hall condition: for each subset S (mask s), |N(S)| >= |S|
    # |N(S)| = L - H[complement of s]
    # complement of s = FULL_MASK ^ s
    _SUBSETS = np.arange(1, 1 << N, dtype=np.int32)
    _COMPLEMENTS = FULL_MASK ^ _SUBSETS
    _POPCOUNTS = np.array([bin(s).count('1') for s in _SUBSETS], dtype=np.int32)

    # nS[r, s] = L - H[r, complement(s)]
    nS = L - H[:, _COMPLEMENTS]  # shape (R, 4095)
    required = _POPCOUNTS  # shape (4095,)

    # Check all(r, s): nS[r,s] >= required[s]
    feasible = np.all(nS >= required[None, :], axis=1)
    return feasible


def construct():
    t0 = time.time()
    deadline = 82.0  # safety margin

    # Step 1: For each L from MAX_L down to 1, find which residues are feasible.
    # We want the maximum L such that there exists some residue r with Hall feasible.
    # But Hall is necessary and sufficient for bipartite matching here? 
    # Actually, Hall's condition IS the exact criterion for bipartite matching existence.
    # So Hall feasibility == exact matching feasibility.
    # Therefore we can just use Hall and it's exact.

    # Let's verify: Hall's theorem says a perfect matching exists iff for every subset
    # of left nodes, |N(S)| >= |S|. This is exactly what we check. So Hall IS exact.

    # So we just need to find max L and a residue r that is Hall-feasible.
    # Use binary search on L for each residue? Or scan L from high to low?

    # Better: for each L from 144 down to 1, check if any residue is feasible.
    # But that's 144 * (27720 checks) which might be slow.
    
    # Alternative: binary search L for each residue, but do it vectorized.
    # For a given L, compute feasibility for all 27720 residues at once.
    # Then binary search per residue.

    residues = np.arange(LCM_12, dtype=np.int32)

    # Binary search for each residue's f value
    # lo[r] = 1, hi[r] = MAX_L
    lo = np.ones(LCM_12, dtype=np.int32)
    hi = np.full(LCM_12, MAX_L, dtype=np.int32)

    while True:
        # Check if any lo != hi
        if not np.any(lo < hi):
            break
        mid = (lo + hi) // 2
        # For each residue, check feasibility at mid
        # Group residues by mid value to batch
        unique_mids = np.unique(mid)
        for um in unique_mids:
            mask = mid == um
            if not np.any(mask):
                continue
            feasible = _vectorized_hall_feasibility(residues[mask], int(um))
            # Update lo/hi
            idx = np.where(mask)[0]
            for k, idx_val in enumerate(idx):
                if feasible[k]:
                    hi[idx_val] = um
                else:
                    lo[idx_val] = um + 1
        if time.time() - t0 > deadline - 10:
            break

    # f values are in lo (or hi, they should be equal)
    f_values = lo  # should equal hi
    best_idx = np.argmax(f_values)
    best_f = int(f_values[best_idx])
    best_r = int(residues[best_idx])

    # Verify with exact matching (should agree since Hall is exact)
    if time.time() - t0 < deadline - 5:
        verify_f = _compute_f_exact(best_r)
        if verify_f != best_f:
            # Something wrong, fall back
            best_f = verify_f

    return int(best_r)
# EVOLVE-BLOCK-END
