import time
import random
import numpy as np

N = 64
NUM_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 300s limit

    xs = np.arange(N, dtype=np.int32)
    ys = np.arange(N, dtype=np.int32)
    all_x = np.repeat(xs, N)
    all_y = np.tile(ys, N)

    # Compute squared distance matrix
    dx = all_x[:, None] - all_x[None, :]
    dy = all_y[:, None] - all_y[None, :]
    dist2 = (dx * dx + dy * dy).astype(np.int32)

    # For each point P, precompute the number of points at each squared distance
    # dist_counts[P] is an array of length (MAX_DIST+1) where dist_counts[P][d] = number of Q != P with dist2[P,Q] = d
    MAX_DIST2 = 2 * N * N
    dist_counts = np.zeros((NUM_POINTS, MAX_DIST2 + 1), dtype=np.int32)

    for P in range(NUM_POINTS):
        d_vals = dist2[P]
        d_vals[P] = MAX_DIST2 + 1  # Mark self as invalid
        # Use np.bincount but we need to handle the range
        # Actually, let's just use a loop with vectorized counting
        unique_d, counts = np.unique(d_vals, return_counts=True)
        for d, c in zip(unique_d, counts):
            if d <= MAX_DIST2:
                dist_counts[P, d] = c

    # For each point P and each distance d, store a bitmask of points at that distance
    # But this would use too much memory. Instead, we'll compute on the fly.
    # Alternative: for each P, create a dict: dist -> list of points
    # This is O(N^3) memory which is 64^3 * 8 bytes = 2M bytes, acceptable
    dist_map = [{} for _ in range(NUM_POINTS)]
    for P in range(NUM_POINTS):
        for Q in range(NUM_POINTS):
            if P != Q:
                d = int(dist2[P, Q])
                if d not in dist_map[P]:
                    dist_map[P][d] = []
                dist_map[P][d].append(Q)

    def can_add(P, S, dist_sets):
        """Check if P can be added to set S without forming an isosceles triangle."""
        n = len(S)
        if n == 0:
            return True
        if n == 1:
            return True

        # Get distances from P to all points in S
        d_vals = dist2[P, S]

        # Check if all distances are distinct (P is not apex)
        if n > 1:
            # Use sorting for O(n log n)
            sorted_d = np.sort(d_vals)
            if np.any(sorted_d[:-1] == sorted_d[1:]):
                return False

        # Check if any point in S has P at a distance it already has
        for i in range(n):
            d = int(d_vals[i])
            if d in dist_sets[i]:
                return False

        return True

    def add_point(P, S, dist_sets):
        """Add point P to set S, updating dist_sets."""
        S.append(P)
        new_dist_set = set()
        if len(S) > 1:
            d_vals = dist2[P, S[:-1]]
            for i, d in enumerate(d_vals):
                d_int = int(d)
                dist_sets[i].add(d_int)
                new_dist_set.add(d_int)
        dist_sets.append(new_dist_set)

    def remove_point(P, S, dist_sets):
        """Remove point P from set S, updating dist_sets."""
        idx = S.index(P)
        for i in range(len(S)):
            if i == idx:
                continue
            d = int(dist2[P, S[i]])
            dist_sets[i].discard(d)
        S.pop(idx)
        dist_sets.pop(idx)

    def greedy_construction(order, time_limit):
        """Greedy construction with time limit."""
        S = []
        dist_sets = []
        in_S = np.zeros(NUM_POINTS, dtype=bool)

        for P in order:
            if time.time() > time_limit:
                break
            if in_S[P]:
                continue

            if can_add(P, S, dist_sets):
                in_S[P] = True
                add_point(P, S, dist_sets)

        return S, dist_sets, in_S

    def local_search(S, dist_sets, in_S, time_limit):
        """Local search with add/remove moves."""
        best_S = S[:]
        best_dist_sets = [s.copy() for s in dist_sets]
        best_size = len(S)

        no_improve = 0
        max_no_improve = 50

        while time.time() < time_limit and no_improve < max_no_improve:
            added = False

            # Try to add a random point
            complement_size = NUM_POINTS - len(S)
            if complement_size > 0:
                batch_size = min(200, complement_size)
                candidates = random.sample(range(NUM_POINTS), batch_size)
                for P in candidates:
                    if in_S[P]:
                        continue

                    if can_add(P, S, dist_sets):
                        in_S[P] = True
                        add_point(P, S, dist_sets)

                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S[:]
                            best_dist_sets = [s.copy() for s in dist_sets]
                            no_improve = 0
                        else:
                            no_improve += 1

                        added = True
                        break

            if not added:
                no_improve += 1

                # Perturbation: remove 1-2 random points, try to add 2-3
                if len(S) > 3:
                    num_remove = random.choice([1, 1, 2])
                    removed = []
                    for _ in range(num_remove):
                        if not S:
                            break
                        rem_idx = random.randrange(len(S))
                        rem_P = S[rem_idx]
                        in_S[rem_P] = False
                        remove_point(rem_P, S, dist_sets)
                        removed.append(rem_P)

                    # Try to add points
                    added_count = 0
                    target_add = num_remove + 1
                    complement_size = NUM_POINTS - len(S)
                    if complement_size > 0:
                        batch_size = min(100, complement_size)
                        candidates = random.sample(range(NUM_POINTS), batch_size)
                        for P in candidates:
                            if in_S[P]:
                                continue

                            if can_add(P, S, dist_sets):
                                in_S[P] = True
                                add_point(P, S, dist_sets)
                                added_count += 1

                                if len(S) > best_size:
                                    best_size = len(S)
                                    best_S = S[:]
                                    best_dist_sets = [s.copy() for s in dist_sets]
                                    no_improve = 0

                                if added_count >= target_add:
                                    break

                    no_improve = 0

        return best_S, best_dist_sets

    # Main loop
    best_set = []
    best_size = 0
    rng = np.random.default_rng(42)

    # Multiple restarts with different random seeds
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break

        # Create random order
        order = rng.permutation(NUM_POINTS).tolist()

        # Greedy construction
        greedy_time = time.time() + min(3, (deadline - time.time()) / 4)
        S, dist_sets, in_S = greedy_construction(order, greedy_time)

        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

        # Local search
        remaining = deadline - time.time()
        if remaining > 5:
            search_time = min(10, remaining / 4)
            S, dist_sets = local_search(S, dist_sets, in_S, time.time() + search_time)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]

        # Early exit if we've found a good solution
        if best_size > 60:
            break

    # Convert indices to coordinates
    result = [(idx % N, idx // N) for idx in best_set]

    return result

if __name__ == "__main__":
    points = construct()
    print(f"Found set of size {len(points)}")
    print(points[:10])
