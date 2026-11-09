"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # wall-clock limit with margin

    base = 51975  # 3^3 * 5^2 * 7 * 11

    # Precompute all divisors > 1 of base
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    divisors.sort()
    n_divs = len(divisors)

    # Precompute position arrays for each divisor
    # pos_arrays[m_idx] = numpy array of positions covered when residue=0
    # For residue r, positions are pos_arrays[m_idx] + r (mod base)
    pos_arrays = []
    for d in divisors:
        pos_arrays.append(np.arange(0, base, d, dtype=np.int32))

    # Greedy initialization with coverage-count approach
    def greedy_init(order):
        """Greedy: process moduli in given order, each picks best residue."""
        cover_count = np.zeros(base, dtype=np.int16)
        classes = []
        for m_idx in order:
            m = divisors[m_idx]
            pos0 = pos_arrays[m_idx]
            best_r = 0
            best_count = -1
            # Try each residue; count how many new positions we cover
            for r in range(m):
                positions = (pos0 + r) % base
                new_cover = np.sum(cover_count[positions] == 0)
                if new_cover > best_count:
                    best_count = new_cover
                    best_r = r
            classes.append([m_idx, best_r])
            positions = (pos0 + best_r) % base
            cover_count[positions] += 1
        uncovered = int(np.sum(cover_count == 0))
        return classes, uncovered

    # Fast evaluation using coverage count
    def evaluate_classes(classes):
        """Count uncovered residues. classes is list of [m_idx, r]."""
        cover_count = np.zeros(base, dtype=np.int16)
        for m_idx, r in classes:
            positions = (pos_arrays[m_idx] + r) % base
            cover_count[positions] += 1
        return int(np.sum(cover_count == 0))

    # Generate multiple initial solutions
    best_classes = None
    best_score = base

    # Try several deterministic orderings
    orderings = []
    orderings.append(list(range(n_divs)))  # increasing
    orderings.append(list(range(n_divs - 1, -1, -1)))  # decreasing
    orderings.append(sorted(range(n_divs), key=lambda i: -divisors[i]))  # largest m first
    orderings.append(sorted(range(n_divs), key=lambda i: (base // divisors[i], divisors[i])))
    orderings.append(sorted(range(n_divs), key=lambda i: divisors[i] % 10))

    # Add random orderings
    for _ in range(15):
        perm = list(range(n_divs))
        random.shuffle(perm)
        orderings.append(perm)

    for i, order in enumerate(orderings):
        if time.time() - start_time > deadline - 25:
            break
        try:
            classes, sc = greedy_init(order)
            if sc < best_score:
                best_score = sc
                best_classes = [c[:] for c in classes]
        except Exception:
            pass

    # Simulated annealing with fast delta computation
    if best_classes is not None and time.time() - start_time < deadline - 3:
        curr_classes = [c[:] for c in best_classes]
        curr_score = best_score

        # Build coverage count array
        cover_count = np.zeros(base, dtype=np.int16)
        for m_idx, r in curr_classes:
            positions = (pos_arrays[m_idx] + r) % base
            cover_count[positions] += 1

        def compute_delta(m_idx, r_old, r_new):
            """Compute score change when changing residue of class m_idx from r_old to r_new."""
            m = divisors[m_idx]
            pos0 = pos_arrays[m_idx]
            old_positions = (pos0 + r_old) % base
            new_positions = (pos0 + r_new) % base

            # Positions that will become uncovered (were covered only by this class)
            old_only = np.sum((cover_count[old_positions] == 1))

            # Positions that will become covered (currently uncovered)
            new_uncovered = np.sum((cover_count[new_positions] == 0))

            # But we need to be careful: positions in both old and new sets
            # For positions in old set: cover_count decreases by 1
            # For positions in new set: cover_count increases by 1
            # Net change in uncovered = old_only - new_uncovered
            # But if a position is in both old and new, it's unchanged
            # Actually, let's just compute it properly:
            # After change: for old_positions, cover_count -= 1; for new_positions, cover_count += 1
            # Uncovered increases by: count of old_positions where cover_count was 1 (becomes 0)
            # Uncovered decreases by: count of new_positions where cover_count was 0 (becomes 1)
            # But positions in intersection: cover_count net change is 0, so no effect
            # However, if a position is in both, and cover_count was 1, it stays at 1
            # So the simple formula works IF we handle the intersection correctly
            # Let's just use the simple formula; it's correct because:
            # - old_only counts positions where removing this class makes them uncovered
            # - new_uncovered counts positions where adding this class makes them covered
            # - positions in both: removing then adding = net 0, so they don't affect uncovered count
            # Wait, that's not quite right. If a position is in both old and new:
            #   Before: cover_count = c (includes this class)
            #   After removing old: c-1
            #   After adding new: c
            #   So it's back to original, no change in uncovered status
            # If a position is only in old:
            #   Before: c (includes this class)
            #   After: c-1
            #   Uncovered if c-1 == 0, i.e., c == 1
            # If a position is only in new:
            #   Before: c (does not include this class)
            #   After: c+1
            #   Becomes covered if c == 0
            # So the formula is correct!
            return int(old_only - new_uncovered)

        # SA parameters
        T = float(curr_score + 1) if curr_score > 0 else 5.0
        T_min = 0.01
        cooling = 0.9995

        best_sa_classes = [c[:] for c in curr_classes]
        best_sa_score = curr_score

        n_iter = 0
        while time.time() - start_time < deadline - 3 and T > T_min:
            n_iter += 1

            # Pick a random class to modify
            idx = random.randint(0, len(curr_classes) - 1)
            m_idx_old, r_old = curr_classes[idx]
            m_old = divisors[m_idx_old]

            # Generate new residue
            r_new = random.randint(0, m_old - 1)
            if r_new == r_old:
                continue

            # Compute delta
            delta = compute_delta(m_idx_old, r_old, r_new)

            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept: update cover_count
                pos0 = pos_arrays[m_idx_old]
                old_positions = (pos0 + r_old) % base
                new_positions = (pos0 + r_new) % base
                cover_count[old_positions] -= 1
                cover_count[new_positions] += 1

                curr_classes[idx] = [m_idx_old, r_new]
                curr_score += delta

                if curr_score < best_sa_score:
                    best_sa_score = curr_score
                    best_sa_classes = [c[:] for c in curr_classes]

            T *= cooling

        if best_sa_score < best_score:
            best_score = best_sa_score
            best_classes = [c[:] for c in best_sa_classes]

    # Final local search: try changing each class's residue to the best one
    if best_classes is not None and time.time() - start_time < deadline - 1:
        # Rebuild cover_count
        cover_count = np.zeros(base, dtype=np.int16)
        for m_idx, r in best_classes:
            positions = (pos_arrays[m_idx] + r) % base
            cover_count[positions] += 1
        curr_score = int(np.sum(cover_count == 0))

        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 1:
                break
            for i in range(len(best_classes)):
                if time.time() - start_time > deadline - 1:
                    break
                m_idx, r = best_classes[i]
                m = divisors[m_idx]
                # Try all residues for this class
                best_r = r
                best_sc = curr_score
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Compute delta
                    pos0 = pos_arrays[m_idx]
                    old_positions = (pos0 + r) % base
                    new_positions = (pos0 + new_r) % base
                    old_only = np.sum(cover_count[old_positions] == 1)
                    new_uncovered = np.sum(cover_count[new_positions] == 0)
                    delta = int(old_only - new_uncovered)
                    sc = curr_score + delta
                    if sc < best_sc:
                        best_sc = sc
                        best_r = new_r

                if best_r != r:
                    # Apply the best change
                    pos0 = pos_arrays[m_idx]
                    old_positions = (pos0 + r) % base
                    new_positions = (pos0 + best_r) % base
                    cover_count[old_positions] -= 1
                    cover_count[new_positions] += 1
                    best_classes[i] = [m_idx, best_r]
                    curr_score = best_sc
                    improved = True

        best_score = curr_score

    # Convert indices back to actual moduli
    result = [[divisors[m_idx], r] for m_idx, r in best_classes]
    return result
# EVOLVE-BLOCK-END
