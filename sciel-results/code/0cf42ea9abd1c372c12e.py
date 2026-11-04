import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    # Build coverage bitsets for fast operations
    # coverage[c] = bitmask of points covered by codeword c
    # For space_size up to 2187, we can use Python integers as bitsets

    def build_coverage_matrix():
        """Build a matrix where cov_matrix[i,j] = 1 if point j is covered by codeword i"""
        cov = np.zeros((space_size, space_size), dtype=np.uint8)
        for i in range(space_size):
            for nb in neighbors[i]:
                cov[i, nb] = 1
        return cov

    cov_matrix = build_coverage_matrix()

    def compute_uncovered(code_set):
        """Return boolean array of uncovered points."""
        if len(code_set) == 0:
            return np.ones(space_size, dtype=bool)
        covered = np.zeros(space_size, dtype=bool)
        for c in code_set:
            covered |= cov_matrix[c]
        return ~covered

    def greedy_cover(seed=0):
        """Greedy set cover with randomized tie-breaking."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            # Compute gain for each potential codeword
            gains = np.sum(uncovered[cov_matrix], axis=1)
            # Add small random perturbation to break ties
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[cov_matrix[best]] = False
        return code

    def greedy_repair(code_set, max_time=10.0):
        """Add codewords to cover all uncovered points."""
        code_list = list(code_set)
        uncovered = compute_uncovered(code_list)
        while np.any(uncovered) and (time.time() - t_start) < max_time:
            gains = np.sum(uncovered[cov_matrix], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            uncovered[cov_matrix[best]] = False
        return code_list

    # Phase 1: Get initial solutions via multiple greedy starts
    best_code = None
    best_size = space_size + 1

    for seed in range(15):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        if not compute_uncovered(code).any():
            if len(code) < best_size:
                best_code = code[:]
                best_size = len(code)

    if best_code is None:
        best_code = greedy_cover(0)
        best_size = len(best_code)

    # Phase 2: Randomized local search with kick-and-repair
    rng = np.random.default_rng(42)

    cur_code = best_code[:]
    cur_size = len(cur_code)

    # Local search: try removing random codewords and repairing
    for restart in range(8):
        if (time.time() - t_start) > deadline - 10.0:
            break

        # Start from current best or a new greedy
        if restart == 0:
            cur_code = best_code[:]
        else:
            cur_code = greedy_cover(restart + 100)
            if compute_uncovered(cur_code).any():
                cur_code = greedy_repair(cur_code, deadline - 20.0)

        cur_size = len(cur_code)

        # Kick-and-repair: remove k random codewords, then repair
        for k in range(1, 5):
            if (time.time() - t_start) > deadline - 5.0:
                break
            for trial in range(40):
                if (time.time() - t_start) > deadline - 5.0:
                    break
                if len(cur_code) <= k:
                    break

                # Remove k random codewords
                remove_indices = rng.choice(len(cur_code), size=k, replace=False)
                remove_set = set(int(ri) for ri in remove_indices)

                new_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
                new_code = greedy_repair(new_code, deadline - 10.0)

                if not compute_uncovered(new_code).any() and len(new_code) < len(cur_code):
                    cur_code = new_code
                    cur_size = len(cur_code)
                    break  # Restart with improved state

        # Deterministic removal pass: remove any codeword whose removal doesn't uncover points
        if (time.time() - t_start) < deadline - 5.0:
            changed = True
            while changed and (time.time() - t_start) < deadline - 5.0:
                changed = False
                for i in range(len(cur_code)):
                    c = cur_code[i]
                    # Check if removing c leaves any point uncovered
                    remaining = [x for j, x in enumerate(cur_code) if j != i]
                    if not compute_uncovered(remaining).any():
                        cur_code.pop(i)
                        cur_size -= 1
                        changed = True
                        break

        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Phase 3: Final aggressive removal pass
    if (time.time() - t_start) < deadline - 2.0:
        cur_code = best_code[:]
        changed = True
        while changed and (time.time() - t_start) < deadline - 2.0:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                remaining = [x for j, x in enumerate(cur_code) if j != i]
                if not compute_uncovered(remaining).any():
                    cur_code.pop(i)
                    changed = True
                    break
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    if compute_uncovered(best_code).any():
        # Fallback: add more codewords
        best_code = greedy_repair(best_code, deadline - 1.0)
        if compute_uncovered(best_code).any():
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
