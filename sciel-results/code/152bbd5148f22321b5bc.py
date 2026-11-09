import time
import random
import sys
import numpy as np

N = 9
MAX_VERTS = 2 ** N
DEADLINE = 118.0  # Safety margin for 150s limit


def construct():
    """
    Construct the longest induced path (snake) in Q_9.
    
    Strategy: Multi-start with hybrid approach:
    1. Greedy bidirectional growth with Warnsdorff's rule (smooth surrogate scoring)
    2. Local search improvement: try to extend stuck snakes by backtracking
    3. Final polish pass: try to insert vertices into the path
    4. Diversity through different random seeds and parameters
    """
    start_time = time.time()
    best_path = None
    best_len = 0

    # Precompute neighbors as numpy array for fast indexing
    neighbor_array = np.zeros((MAX_VERTS, N), dtype=np.int32)
    for v in range(MAX_VERTS):
        for b in range(N):
            neighbor_array[v, b] = v ^ (1 << b)

    # Precompute bit masks
    bit_masks = np.array([1 << b for b in range(N)], dtype=np.int32)

    def _check_time():
        return (time.time() - start_time) < DEADLINE

    def grow_snake(start_vertex, max_steps=500, randomness=0.1, use_backtrack=False):
        """
        Grow a snake from start_vertex using Warnsdorff's rule with smooth surrogate scoring.
        Returns the path as a list.
        """
        path = [start_vertex]
        used = np.zeros(MAX_VERTS, dtype=bool)
        used[start_vertex] = True

        # deg_in_path[v] = number of neighbors of v that are in the current path
        deg_in_path = np.zeros(MAX_VERTS, dtype=np.int32)

        for w in neighbor_array[start_vertex]:
            deg_in_path[w] += 1

        steps = 0
        while steps < max_steps:
            if not _check_time():
                break

            left_end = path[0]
            right_end = path[-1]

            # Get valid moves for left end using numpy
            left_neighbors = neighbor_array[left_end]
            left_valid_mask = (~used[left_neighbors]) & (deg_in_path[left_neighbors] == 1)
            valid_left = left_neighbors[left_valid_mask]

            # Get valid moves for right end using numpy
            right_neighbors = neighbor_array[right_end]
            right_valid_mask = (~used[right_neighbors]) & (deg_in_path[right_neighbors] == 1)
            valid_right = right_neighbors[right_valid_mask]

            if len(valid_left) == 0 and len(valid_right) == 0:
                break

            # Smooth surrogate scoring: soft-min via log-sum-exp of negative counts
            def count_onward_smooth(candidates):
                """
                Compute onward counts for all candidates at once using numpy.
                Returns array of onward counts.
                """
                if len(candidates) == 0:
                    return np.array([], dtype=np.int32)
                cand_neighbors = neighbor_array[candidates]  # shape (len(candidates), N)
                # For each candidate, count neighbors w where used[w] == False and deg_in_path[w] == 0
                cand_used = used[cand_neighbors]  # shape (len(candidates), N)
                cand_deg = deg_in_path[cand_neighbors]  # shape (len(candidates), N)
                valid_mask = (~cand_used) & (cand_deg == 0)
                onward_counts = np.sum(valid_mask, axis=1)
                return onward_counts

            # Score left candidates
            best_left = None
            best_left_score = -1
            if len(valid_left) > 0:
                onward_left = count_onward_smooth(valid_left)
                # Smooth min: use softmax-weighted selection with temperature
                # Lower onward count = more constrained = better (Warnsdorff)
                scores = -onward_left.astype(np.float64) + random.random() * randomness * len(valid_left)
                # Use argmax of scores (most negative onward + noise)
                best_idx = np.argmax(scores)
                best_left = int(valid_left[best_idx])
                best_left_score = scores[best_idx]

            # Score right candidates
            best_right = None
            best_right_score = -1
            if len(valid_right) > 0:
                onward_right = count_onward_smooth(valid_right)
                scores = -onward_right.astype(np.float64) + random.random() * randomness * len(valid_right)
                best_idx = np.argmax(scores)
                best_right = int(valid_right[best_idx])
                best_right_score = scores[best_idx]

            # Choose which side to extend
            if best_left is None:
                path.append(best_right)
                used[best_right] = True
                for w in neighbor_array[best_right]:
                    deg_in_path[w] += 1
            elif best_right is None:
                path.insert(0, best_left)
                used[best_left] = True
                for w in neighbor_array[best_left]:
                    deg_in_path[w] += 1
            else:
                # Choose side with more constrained (lower onward) option
                # Use the smooth score comparison
                if best_left_score > best_right_score:
                    path.insert(0, best_left)
                    used[best_left] = True
                    for w in neighbor_array[best_left]:
                        deg_in_path[w] += 1
                elif best_right_score > best_left_score:
                    path.append(best_right)
                    used[best_right] = True
                    for w in neighbor_array[best_right]:
                        deg_in_path[w] += 1
                else:
                    if random.random() < 0.5:
                        path.insert(0, best_left)
                        used[best_left] = True
                        for w in neighbor_array[best_left]:
                            deg_in_path[w] += 1
                    else:
                        path.append(best_right)
                        used[best_right] = True
                        for w in neighbor_array[best_right]:
                            deg_in_path[w] += 1

            steps += 1

        return path

    def try_extend_with_backtrack(path, max_backtracks=20):
        """
        Try to extend a stuck snake by backtracking a few steps and trying different branches.
        Returns the extended path if successful, otherwise returns the original path.
        """
        if not _check_time():
            return path

        original_len = len(path)
        best = list(path)

        # Try backtracking from different positions
        for backtrack_depth in range(1, min(max_backtracks, len(path) - 2) + 1):
            if not _check_time():
                break

            # Try removing the last 'backtrack_depth' vertices and regrowing
            truncated = path[:-backtrack_depth] if backtrack_depth > 0 else list(path)

            if len(truncated) < 2:
                continue

            # Regrow from the truncated path
            new_path = list(truncated)
            used = np.zeros(MAX_VERTS, dtype=bool)
            for v in new_path:
                used[v] = True

            deg_in_path = np.zeros(MAX_VERTS, dtype=np.int32)
            for v in new_path:
                for w in neighbor_array[v]:
                    deg_in_path[w] += 1

            # Try to extend from both ends
            for _ in range(10):  # Try up to 10 extension steps
                if not _check_time():
                    break

                left_end = new_path[0]
                right_end = new_path[-1]

                left_neighbors = neighbor_array[left_end]
                left_valid_mask = (~used[left_neighbors]) & (deg_in_path[left_neighbors] == 1)
                valid_left = left_neighbors[left_valid_mask]

                right_neighbors = neighbor_array[right_end]
                right_valid_mask = (~used[right_neighbors]) & (deg_in_path[right_neighbors] == 1)
                valid_right = right_neighbors[right_valid_mask]

                if len(valid_left) == 0 and len(valid_right) == 0:
                    break

                # Choose a random valid move
                if len(valid_left) > 0 and (len(valid_right) == 0 or random.random() < 0.5):
                    v = int(random.choice(valid_left))
                    new_path.insert(0, v)
                    used[v] = True
                    for w in neighbor_array[v]:
                        deg_in_path[w] += 1
                elif len(valid_right) > 0:
                    v = int(random.choice(valid_right))
                    new_path.append(v)
                    used[v] = True
                    for w in neighbor_array[v]:
                        deg_in_path[w] += 1
                else:
                    break

            if len(new_path) > len(best):
                best = new_path

        return best

    def try_insert_polish(path):
        """
        Final polish pass: try to insert additional vertices into the path.
        For each pair of consecutive vertices in the path, check if there's an unused
        vertex that can be inserted between them (differs by one bit from both).
        Also try to extend ends.
        Returns the polished path.
        """
        if not _check_time():
            return path

        new_path = list(path)
        used = np.zeros(MAX_VERTS, dtype=bool)
        for v in new_path:
            used[v] = True

        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            if not _check_time():
                break

            # Try to extend the ends first
            for end_idx in [0, -1]:
                if end_idx == 0:
                    end_vertex = new_path[0]
                else:
                    end_vertex = new_path[-1]

                end_neighbors = neighbor_array[end_vertex]
                valid_mask = ~used[end_neighbors]
                # Also need deg_in_path check: the candidate must not be adjacent to any other path vertex
                # Compute deg_in_path for end_neighbors
                end_deg = np.zeros(len(end_neighbors), dtype=np.int32)
                for i, w in enumerate(end_neighbors):
                    for u in new_path:
                        if u == end_vertex:
                            continue
                        if w in neighbor_array[u]:
                            end_deg[i] += 1
                valid_mask = valid_mask & (end_deg <= 1)

                if np.any(valid_mask):
                    candidates = end_neighbors[valid_mask]
                    v = int(random.choice(candidates))
                    if end_idx == 0:
                        new_path.insert(0, v)
                    else:
                        new_path.append(v)
                    used[v] = True
                    improved = True

            # Try to insert between consecutive vertices
            if not _check_time():
                break

            for i in range(len(new_path) - 1):
                if not _check_time():
                    break

                v1 = new_path[i]
                v2 = new_path[i + 1]

                # Find common neighbors of v1 and v2
                # A vertex w can be inserted if w is neighbor of v1, neighbor of v2, 
                # w is not used, and w is not adjacent to any other path vertex (except v1, v2)
                v1_neighbors = set(neighbor_array[v1].tolist())
                v2_neighbors = set(neighbor_array[v2].tolist())
                common = v1_neighbors & v2_neighbors

                for w in common:
                    if used[w]:
                        continue

                    # Check if w is adjacent to any path vertex other than v1 and v2
                    w_neighbors = set(neighbor_array[w].tolist())
                    conflict = False
                    for j, u in enumerate(new_path):
                        if u == v1 or u == v2:
                            continue
                        if u in w_neighbors:
                            conflict = True
                            break
                    if conflict:
                        continue

                    # Insert w between v1 and v2
                    new_path.insert(i + 1, w)
                    used[w] = True
                    improved = True
                    break

        return new_path

    # Multi-start search
    attempts = 0
    while _check_time():
        attempts += 1

        # Vary randomness for diversity
        if attempts % 5 == 0:
            randomness = random.choice([0.0, 0.05, 0.1, 0.2, 0.5])
        else:
            randomness = random.uniform(0.0, 0.15)

        # Random start vertex
        v0 = random.randint(0, MAX_VERTS - 1)

        # Grow snake
        p = grow_snake(v0, max_steps=600, randomness=randomness)

        # Try to extend with backtracking if we have time
        if time.time() - start_time < DEADLINE - 5.0:
            p = try_extend_with_backtrack(p, max_backtracks=15)

        # Try insert polish if we have time
        if time.time() - start_time < DEADLINE - 3.0:
            p = try_insert_polish(p)

        if len(p) > best_len:
            best_len = len(p)
            best_path = p

        # Early termination if we've found a very long snake
        if best_len >= 200:
            break

    if best_path is None:
        return [0]

    return best_path
