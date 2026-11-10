import numpy as np
import itertools
import time
import sys

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 575  # safety margin before the 630s kill

    # ---- Candidate pool: all integer vectors of norm^2 = 2 or 4 ----
    # This is a mixed-norm pool (norm^2 2 and 4) which allows denser packing
    # than a pure norm^2=2 set, because the minimum squared distance constraint
    # is min_dist^2 >= max_norm^2 (here max_norm^2 = 4).
    candidates = []
    # Type A: two non-zero coordinates, each +-1 (norm^2 = 2)
    for i, j in itertools.combinations(range(d), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(d, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
    # Type B: one coordinate +-2 (norm^2 = 4)
    for i in range(d):
        for s in (2, -2):
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            candidates.append(v)

    n = len(candidates)
    cand_arr = np.array(candidates, dtype=np.int64)

    # ---- Conflict matrix: pair (i,j) conflicts if dist^2 < max_norm^2 = 4 ----
    # We want min over all pairs of (p_i - p_j)^2 >= max over all points of p_i^2.
    # With mixed norms, max_norm^2 can be 4 (if any type-B point is selected).
    # To be safe, enforce dist^2 >= 4 for all pairs (this is the binding constraint
    # when any type-B is present, and type-A pairs with dist^2 >= 4 are fine).
    # Actually the constraint is: min_pair_dist^2 >= max_point_norm^2.
    # If we only use type-A (norm^2=2), we need dist^2 >= 2.
    # If we use any type-B (norm^2=4), we need dist^2 >= 4.
    # Since we want a valid configuration, and we'll include type-B points,
    # we enforce dist^2 >= 4 for all pairs. This is stricter but yields valid sets.

    # Precompute pairwise squared distances
    # For memory, n can be up to C(11,2)*4 + 22 = 220 + 22 = 242
    # That's fine: 242x242 matrix
    dist2 = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            diff = cand_arr[i] - cand_arr[j]
            dist2[i, j] = int(np.dot(diff, diff))
            dist2[j, i] = dist2[i, j]

    # Conflict: dist^2 < 4
    conflict = dist2 < 4

    # Remove self-conflicts (shouldn't exist, but just in case)
    np.fill_diagonal(conflict, False)

    # Degrees for greedy ordering
    degrees = np.sum(conflict, axis=1).astype(np.int32)

    # ---- Greedy with multiple orderings ----
    best_set = []
    best_count = 0

    # Ascending degree (pick low-conflict points first)
    order_asc = np.argsort(degrees)
    # Descending degree
    order_desc = np.argsort(-degrees)

    orders = [order_asc, order_desc]

    # Add some random orders
    for seed in range(30):
        if time.time() > deadline - 120:
            break
        rng = np.random.default_rng(seed)
        orders.append(rng.permutation(n))

    for order in orders:
        if time.time() > deadline - 60:
            break
        selected = []
        forbidden = np.zeros(n, dtype=bool)
        for ci in order:
            if not forbidden[ci]:
                selected.append(ci)
                forbidden[conflict[ci]] = True
        if len(selected) > best_count:
            best_count = len(selected)
            best_set = selected[:]

    # ---- Local search: 1-for-1 swaps ----
    # Represent selected set as a bitmask for fast incompatibility checks
    # n <= 242, use Python int bitmask
    def set_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    def mask_incompatibility(mask):
        # Returns a bitmask of all candidates incompatible with any in mask
        # i.e., for each selected s, all j such that conflict[j,s] is True
        # We can precompute for each s the conflict row as a bitmask
        # But that's expensive to precompute for all s. Instead, iterate.
        incompat = 0
        m = mask
        while m:
            lb = m & (-m)
            s = lb.bit_length() - 1
            m ^= lb
            # Add all j conflicting with s
            incompat |= conflict_bitmasks[s]
        return incompat

    # Precompute conflict bitmask for each candidate
    conflict_bitmasks = []
    for s in range(n):
        bm = 0
        row = conflict[s]
        for j in range(n):
            if row[j]:
                bm |= (1 << j)
        conflict_bitmasks.append(bm)

    # Local search
    if best_count > 0 and time.time() < deadline - 10:
        cur_set = best_set[:]
        cur_mask = set_to_mask(cur_set)
        cur_incompat = mask_incompatibility(cur_mask)
        improved = True
        max_rounds = 200
        for _round in range(max_rounds):
            if time.time() > deadline - 10:
                break
            improved = False
            # Try removing each element and finding a better replacement
            for si in range(len(cur_set)):
                if time.time() > deadline - 10:
                    break
                removed = cur_set[si]
                # Remove removed from cur_mask and cur_incompat
                # Recompute incompat without removed
                new_mask = cur_mask & ~(1 << removed)
                new_incompat = mask_incompatibility(new_mask)

                # Find all candidates not in new_mask and not in new_incompat
                # that are not the removed one itself
                avail_mask = 0
                for j in range(n):
                    if not (new_incompat >> j) & 1 and not (new_mask >> j) & 1 and j != removed:
                        avail_mask |= (1 << j)

                if avail_mask == 0:
                    continue

                # Pick a candidate from avail_mask that minimizes degree
                # (i.e., conflicts with fewest available candidates)
                best_j = -1
                best_deg = n + 1
                m = avail_mask
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    m ^= lb
                    # Degree against avail_mask
                    deg = (conflict_bitmasks[j] & avail_mask).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_j = j

                if best_j >= 0:
                    # Perform the swap
                    cur_set[si] = best_j
                    cur_mask = new_mask | (1 << best_j)
                    cur_incompat = new_incompat | conflict_bitmasks[best_j]
                    improved = True
                    break
            if not improved:
                break

        if len(cur_set) > best_count:
            best_count = len(cur_set)
            best_set = cur_set[:]

    # ---- Build the point array ----
    if not best_set:
        # Fallback: return just the type-B points (22 points)
        pts = []
        for i in range(d):
            p1 = np.zeros(d, dtype=np.int64)
            p1[i] = 2
            pts.append(p1)
            p2 = np.zeros(d, dtype=np.int64)
            p2[i] = -2
            pts.append(p2)
        return np.array(pts, dtype=np.int64)

    points = [cand_arr[i] for i in best_set]
    points_array = np.array(points, dtype=np.int64)

    # Round and cast (already int, but per contract)
    points_array = np.around(points_array).astype(np.int64)

    return points_array
