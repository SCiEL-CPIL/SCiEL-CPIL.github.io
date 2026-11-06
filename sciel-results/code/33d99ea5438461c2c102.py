import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        all_words = list(itertools.product(range(q), repeat=n))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        neighbors = []
        for i, w in enumerate(all_words):
            neigh_set = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    new_w_list = list(w)
                    new_w_list[pos] = val
                    new_w = tuple(new_w_list)
                    neigh_set.add(word_to_idx[new_w])
            neighbors.append(list(neigh_set))
        N = q ** n
        uncovered = np.ones(N, dtype=bool)
        C = []
        start_time = time.time()
        time_limit = 100.0
        while np.any(uncovered):
            best_idx = -1
            best_count = -1
            for i in range(N):
                cnt = np.sum(uncovered[neighbors[i]])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = i
            if best_count == 0:
                break
            C.append(best_idx)
            uncovered[neighbors[best_idx]] = False
            if time.time() - start_time > time_limit:
                break
        coverage_counts = np.zeros(N, dtype=int)
        for c in C:
            coverage_counts[neighbors[c]] += 1
        changed = True
        while changed:
            changed = False
            new_C = []
            for c in C:
                if np.min(coverage_counts[neighbors[c]]) >= 2:
                    coverage_counts[neighbors[c]] -= 1
                    changed = True
                else:
                    new_C.append(c)
            C = new_C
        return [list(all_words[i]) for i in C]

    # Specific optimization for q=3, n=8, radius=1
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors and convert to numpy arrays
    neighbors = []
    for i, w in enumerate(all_words):
        neigh_set = {i}
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh_set.add(word_to_idx[new_w])
        neighbors.append(list(neigh_set))

    neighbors_np = [np.array(n_list, dtype=int) for n_list in neighbors]

    # Fast Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 115.0

    # Greedy: pick word covering most uncovered
    while np.any(uncovered):
        best_idx = -1
        best_count = -1
        for i in range(N):
            cnt = np.sum(uncovered[neighbors_np[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_count == 0:
            break
        C.append(best_idx)
        uncovered[neighbors_np[best_idx]] = False
        if time.time() - start_time > time_limit:
            break

    # Local Search with Voronoi Volume Maximization
    def get_counts(code):
        counts = np.zeros(N, dtype=int)
        for c in code:
            counts[neighbors_np[c]] += 1
        return counts

    def reduce_code(code, counts):
        if not code:
            return code, counts
        current_code = list(code)
        current_counts = counts.copy()

        changed = True
        while changed:
            changed = False
            next_code = []
            for c in current_code:
                if np.min(current_counts[neighbors_np[c]]) >= 2:
                    current_counts[neighbors_np[c]] -= 1
                    changed = True
                else:
                    next_code.append(c)
            current_code = next_code
        return current_code, current_counts

    C, counts = reduce_code(C, get_counts(C))
    best_size = len(C)
    best_C = C.copy()
    best_counts = counts.copy()

    # Voronoi Volume Maximization
    def compute_voronoi_volume(code, counts):
        # Assign each point to the nearest codeword (by index order as tie-breaker)
        # Volume = number of points assigned to each codeword
        volumes = np.zeros(len(code), dtype=int)
        if not code:
            return volumes

        # Create assignment array
        assignment = np.full(N, -1, dtype=int)
        for idx, c in enumerate(code):
            mask = neighbors_np[c]
            # Points covered only by this codeword
            unique = mask[(counts[mask] == 1)]
            assignment[unique] = idx
            volumes[idx] += len(unique)

        # For points covered by multiple codewords, assign to the first one
        for idx, c in enumerate(code):
            mask = neighbors_np[c]
            # Points covered by multiple codewords but not yet assigned
            multi = mask[(counts[mask] > 1) & (assignment == -1)]
            assignment[multi] = idx
            volumes[idx] += len(multi)

        return volumes

    # Local Search Loop with Voronoi-guided moves
    while time.time() - start_time < time_limit:
        old_size = len(C)
        C, counts = reduce_code(C, counts)

        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = counts.copy()

        if time.time() - start_time > time_limit:
            break

        # Compute Voronoi volumes for current code
        if len(C) > 0:
            volumes = compute_voronoi_volume(C, counts)

        # Swap Pass with Voronoi guidance
        if len(C) == old_size:
            improved = False
            indices = list(range(len(C)))
            random.shuffle(indices)

            for idx in indices:
                if time.time() - start_time > time_limit:
                    break
                c = C[idx]
                u_mask = counts[neighbors_np[c]] == 1
                unique_points = neighbors_np[c][u_mask]

                if len(unique_points) == 0:
                    C.remove(c)
                    counts[neighbors_np[c]] -= 1
                    improved = True
                    break

                if len(unique_points) > 2:
                    continue

                candidates_set = set(neighbors[unique_points.tolist()[0]]) if len(unique_points) > 0 else set()
                for p in unique_points.tolist()[1:]:
                    candidates_set &= set(neighbors[p])
                    if not candidates_set:
                        break

                if not candidates_set:
                    continue

                c_set = set(C)
                candidates_set -= c_set
                if not candidates_set:
                    continue

                # Select candidate that would maximize Voronoi volume
                best_candidate = None
                best_volume_gain = -1

                for c_prime in list(candidates_set)[:5]:  # Limit to 5 candidates for speed
                    # Estimate volume gain
                    old_vol = volumes[idx] if idx < len(volumes) else 0

                    # Points that would be lost
                    lost_points = set(neighbors[c]) - set(neighbors[c_prime])
                    # Points that would be gained
                    gained_points = set(neighbors[c_prime]) - set(neighbors[c])

                    # Rough estimate: gain if gained points are not covered by others
                    gain_estimate = 0
                    for p in gained_points:
                        if counts[p] == 1:  # Only covered by c
                            gain_estimate += 1

                    if gain_estimate > best_volume_gain:
                        best_volume_gain = gain_estimate
                        best_candidate = c_prime

                if best_candidate is None:
                    best_candidate = random.choice(list(candidates_set))
                c_prime = best_candidate

                old_C = C.copy()
                old_counts = counts.copy()

                C.remove(c)
                C.append(c_prime)

                counts[neighbors_np[c]] -= 1
                counts[neighbors_np[c_prime]] += 1

                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue

                new_C, new_counts = reduce_code(C, counts)

                if len(new_C) < len(old_C):
                    C = new_C
                    counts = new_counts
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_counts = new_counts.copy()
                    improved = True
                    break
                else:
                    improved = True
                    break

            if not improved:
                for _ in range(10):
                    if time.time() - start_time > time_limit:
                        break
                    if not C:
                        break
                    c_idx = random.randint(0, len(C) - 1)
                    c = C[c_idx]
                    cand = neighbors[c]
                    c_prime = cand[random.randint(0, len(cand) - 1)]
                    if c_prime in C:
                        continue

                    old_C = C.copy()
                    old_counts = counts.copy()

                    C.remove(c)
                    C.append(c_prime)
                    counts[neighbors_np[c]] -= 1
                    counts[neighbors_np[c_prime]] += 1

                    if np.any(counts < 1):
                        C = old_C
                        counts = old_counts
                        continue

                    new_C, new_counts = reduce_code(C, counts)
                    if len(new_C) < len(old_C):
                        C = new_C
                        counts = new_counts
                        if len(C) < best_size:
                            best_size = len(C)
                            best_C = C.copy()
                            best_counts = new_counts.copy()
                        break
                    else:
                        C = new_C
                        counts = new_counts

    # Final Reduction
    best_C, _ = reduce_code(best_C, get_counts(best_C))

    return [list(all_words[i]) for i in best_C]
