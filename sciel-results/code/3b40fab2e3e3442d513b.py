import time
import numpy as np

def construct(q, n, radius):
    t0 = time.time()
    deadline = 118.0  # safety margin before 120s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

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

    # For each space point, which codewords (positions in the permutation) cover it?
    # Actually we need: for a given set of chosen codeword positions, count coverage per point.
    # We'll precompute: for each space point p, the list of codeword indices c such that p in neighbors[c].
    # That's the transpose of neighbors.
    # Since neighbors is (space_size, nb_count), the transpose is (nb_count, space_size) but we need
    # for each space point, which codeword indices cover it.
    # Let's build cover_of_point: for each point p, list of codeword indices c where p is in neighbors[c].
    # Equivalent: for each c, for each p in neighbors[c], add c to cover_of_point[p].
    # We'll store as a list of arrays, but for SA speed we need fast coverage counting.
    # Alternative: represent the code as a boolean array of length space_size (1 if chosen).
    # Then coverage[p] = sum over c of chosen[c] * (p in neighbors[c]).
    # We can precompute a matrix M where M[c, p] = 1 if p in neighbors[c], else 0.
    # That's just neighbors as a 0/1 matrix. M is (space_size, space_size) which is 2187x2187 ~ 4.8M entries.
    # That's fine.

    M = np.zeros((space_size, space_size), dtype=np.int32)
    for c in range(space_size):
        M[c, neighbors[c]] = 1

    # Now coverage = M.T @ chosen_vec, where chosen_vec is boolean/int of length space_size.
    # For SA on permutations: we maintain a permutation perm of length space_size, and we choose
    # the first K positions as codewords. K is variable? No, fixed-size approach: we fix K and search.
    # But we want to minimize K. Strategy: start with K large enough to be valid, then SA minimizes
    # uncovered count, and we also try to reduce K.
    # Actually, the inspiration says "fixed-size permutation-based ILS". Let's do SA on permutations
    # where the objective is uncovered_count, and we fix the number of codewords K.
    # We'll try multiple K values, starting from a greedy upper bound, and decrease K.

    # Greedy upper bound to get initial K
    def greedy_cover():
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t0) < deadline - 80.0:
            # gain = number of uncovered points covered by each codeword
            # uncovered[neighbors[c]] gives the uncovered points covered by c
            gains = np.sum(uncovered[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    init_code = greedy_cover()
    K_init = len(init_code)
    # Ensure valid
    if len(init_code) < space_size:
        # verify
        cov = np.zeros(space_size, dtype=np.int32)
        for c in init_code:
            cov[neighbors[c]] += 1
        if not np.all(cov > 0):
            # repair
            uncovered = cov == 0
            while np.any(uncovered) and (time.time() - t0) < deadline - 75.0:
                gains = np.sum(uncovered[neighbors], axis=1)
                best = np.argmax(gains)
                if gains[best] == 0:
                    break
                init_code.append(int(best))
                cov[neighbors[best]] += 1
                uncovered = cov == 0
    K_init = len(init_code)

    best_code = init_code[:]
    best_K = K_init

    # SA on permutations for a fixed K
    def sa_for_K(K, max_time):
        """Run SA on permutations to find a valid covering of size K.
        Returns the list of K codeword indices, or None if not found."""
        if (time.time() - t0) > deadline - 5.0:
            return None
        rng = np.random.default_rng()
        perm = rng.permutation(space_size)
        # chosen = first K of perm
        chosen_vec = np.zeros(space_size, dtype=np.int32)
        chosen_vec[perm[:K]] = 1
        cov = M.T @ chosen_vec
        uncovered = int(np.sum(cov == 0))

        if uncovered == 0:
            return perm[:K].tolist()

        T = 50.0
        T_min = 0.01
        cooling = 0.995
        best_perm = perm.copy()
        best_uncovered = uncovered

        t_start = time.time()
        while (time.time() - t_start) < max_time and T > T_min:
            # Move: swap two positions in perm
            i = rng.integers(0, K)
            j = rng.integers(0, space_size)
            if i == j:
                continue
            # Swap perm[i] and perm[j]
            old_i = perm[i]
            old_j = perm[j]
            perm[i] = old_j
            perm[j] = old_i

            # Update chosen_vec and cov
            # If i < K and j < K: both in/out? No, both were in, both still in, just swapped. No change in chosen set.
            # If i < K and j >= K: old_i removed, old_j added.
            # If i >= K and j < K: old_j removed, old_i added.
            # If i >= K and j >= K: no change in chosen set.
            if i < K and j >= K:
                # remove old_i, add old_j
                cov -= M[old_i]
                cov += M[old_j]
                new_uncovered = int(np.sum(cov == 0))
                delta = new_uncovered - uncovered
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    uncovered = new_uncovered
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_perm = perm.copy()
                        if uncovered == 0:
                            return perm[:K].tolist()
                else:
                    # revert
                    perm[i] = old_i
                    perm[j] = old_j
                    cov += M[old_i]
                    cov -= M[old_j]
            elif i >= K and j < K:
                # remove old_j, add old_i
                cov -= M[old_j]
                cov += M[old_i]
                new_uncovered = int(np.sum(cov == 0))
                delta = new_uncovered - uncovered
                if delta <= 0 or rng.random() < np.exp(-delta / T):
                    uncovered = new_uncovered
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_perm = perm.copy()
                        if uncovered == 0:
                            return perm[:K].tolist()
                else:
                    perm[i] = old_i
                    perm[j] = old_j
                    cov += M[old_j]
                    cov -= M[old_i]
            else:
                # no change in chosen set, but perm changed; no need to revert for coverage
                pass

            T *= cooling

        if best_uncovered == 0:
            return best_perm[:K].tolist()
        return None

    # Try decreasing K from K_init
    # We have limited time; allocate time per K value.
    # Start from K_init and try K_init-1, K_init-2, etc.
    # Each SA run gets a fraction of the remaining time.
    time_per_sa = 8.0  # seconds per SA run
    K = K_init
    while K > 150 and (time.time() - t0) < deadline - 10.0:
        result = sa_for_K(K, time_per_sa)
        if result is not None:
            best_code = result
            best_K = K
            # Try to go lower
            K -= 1
        else:
            # Try a few more times at this K with fresh restarts
            found = False
            for _ in range(3):
                if (time.time() - t0) > deadline - 10.0:
                    break
                result = sa_for_K(K, time_per_sa)
                if result is not None:
                    best_code = result
                    best_K = K
                    found = True
                    K -= 1
                    break
            if not found:
                # Can't achieve this K, stop
                break

    # Final deterministic removal pass on best
    if (time.time() - t0) < deadline - 3.0:
        cur_code = best_code[:]
        cov = np.zeros(space_size, dtype=np.int32)
        for c in cur_code:
            cov[neighbors[c]] += 1
        changed = True
        while changed and (time.time() - t0) < deadline - 3.0:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cov[neighbors[c]] -= 1
                    changed = True
                    break
        if len(cur_code) < len(best_code):
            best_code = cur_code

    # Final verification
    if len(best_code) == 0:
        return [idx_to_word(i) for i in range(space_size)]

    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
    if not np.all(final_cov > 0):
        # Fallback: rebuild from greedy
        fc = greedy_cover()
        fcov = np.zeros(space_size, dtype=np.int32)
        for c in fc:
            fcov[neighbors[c]] += 1
        if np.all(fcov > 0):
            return [idx_to_word(c) for c in fc]
        return [idx_to_word(i) for i in range(space_size)]

    return [idx_to_word(c) for c in best_code]
