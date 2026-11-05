import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    # Precompute neighbors (closed balls of radius 1)
    neighbors = []
    for w in words:
        neigh = [word_to_idx[w]]
        w_list = list(w)
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list[pos] = val
                neigh.append(word_to_idx[tuple(w_list)])
                w_list[pos] = orig
        neighbors.append(neigh)

    # Ball sizes
    ball_sizes = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)

    # Build CSR incidence matrix: rows = codewords, cols = points covered
    # Actually, we need: for each point, which codewords cover it (for gain computation)
    # And for each codeword, which points it covers (for updating counts)

    # For gain: gains[c] = sum over uncovered points p of [c covers p]
    # This equals: for each uncovered point p, add 1 to all c in neighbors[p]
    # We can do this with np.bincount if we flatten

    # Precompute neighbor arrays
    neighbor_arrays = [np.array(neigh, dtype=np.int32) for neigh in neighbors]

    # For fast "does codeword c cover point p": use neighbor_sets
    # But building 2187 sets is fine
    neighbor_sets = [set(neigh) for neigh in neighbors]

    # --- Greedy Set Cover ---
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()

    # Precompute: for each point p, the list of codewords that cover it (= neighbors[p])
    # This is already neighbors[p]

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break

        # Compute gains: for each codeword c, how many uncovered points it covers
        # gains[c] = number of uncovered points p such that c in neighbors[p]
        # = for each uncovered p, add 1 to all c in neighbors[p]
        # Use bincount: flatten all neighbors of uncovered points
        uncovered_idx = np.where(uncovered)[0]
        if len(uncovered_idx) == 0:
            break

        # Build flat array of all codeword indices that cover any uncovered point
        # For each uncovered point p, its neighbors are the codewords covering it
        flat = []
        for p in uncovered_idx:
            flat.extend(neighbors[p])
        flat = np.array(flat, dtype=np.int32)
        gains = np.bincount(flat, minlength=N).astype(np.int32)

        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break

        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbor_arrays[best_idx]] = False

    # --- Local Search ---
    # Build counts: counts[p] = number of codewords covering point p
    counts = np.zeros(N, dtype=np.int32)
    for c in code:
        counts[neighbor_arrays[c]] += 1

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False

        # Deletion pass: remove all codewords whose removal doesn't uncover any point
        # A codeword c is removable if for all p in neighbors[c], counts[p] > 1
        to_remove = []
        for c in code:
            if np.min(counts[neighbor_arrays[c]]) > 1:
                to_remove.append(c)

        if to_remove:
            improved = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1

        # Swap pass
        random.shuffle(code)
        for c in code:
            if time.time() > deadline - 5:
                break

            # Check if c is removable
            if np.min(counts[neighbor_arrays[c]]) > 1:
                continue

            # Critical points: points covered only by c
            crit = []
            for p in neighbor_arrays[c]:
                if counts[p] == 1:
                    crit.append(int(p))

            if not crit:
                continue

            # Try to find a replacement w not in code_set such that:
            # After removing c and adding w, all points remain covered.
            # Points that would become uncovered: those in neighbors[c] not in neighbors[w]
            # and with counts[p] == 1 (only covered by c)
            # So we need: for all p in neighbors[c] with counts[p]==1, p must be in neighbors[w]
            # i.e., w must cover all critical points

            # Candidates: codewords that cover all critical points
            # Start from candidates covering the first critical point
            first_crit = crit[0]
            cands = [x for x in neighbors[first_crit] if x not in code_set]

            best_w = -1
            best_score = -1

            for w in cands:
                # Check if w covers all critical points
                valid = True
                for k in crit:
                    if k not in neighbor_sets[w]:
                        valid = False
                        break
                if not valid:
                    continue

                # Score: maximize the number of points where adding w creates redundancy
                # (points with counts[p] == 1 that are also covered by w, excluding c's neighbors)
                # Actually, simpler: maximize total coverage of w on points with count >= 1
                # Or: minimize the "damage" - but since we require all crit covered,
                # just pick the one that maximizes additional useful coverage
                score = 0
                for p in neighbor_arrays[w]:
                    if counts[p] >= 1:
                        score += 1
                # Prefer codewords that cover more already-covered points (redundancy)
                # This makes the code more robust for future swaps

                if score > best_score:
                    best_score = score
                    best_w = w

            if best_w != -1:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1

                code.append(best_w)
                code_set.add(best_w)
                counts[neighbor_arrays[best_w]] += 1

                improved = True

    # Final cleanup
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbor_arrays[idx]] += 1

    changed = True
    while changed:
        changed = False
        to_remove = []
        for c in code:
            if np.min(counts[neighbor_arrays[c]]) > 1:
                to_remove.append(c)
        if to_remove:
            changed = True
            for c in to_remove:
                code.remove(c)
                code_set.remove(c)
                counts[neighbor_arrays[c]] -= 1

    return [list(words[i]) for i in code]
