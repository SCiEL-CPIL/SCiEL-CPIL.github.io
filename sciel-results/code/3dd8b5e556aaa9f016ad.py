"""Simulated annealing for the ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock


def _tetracode_words():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _initial_codewords(q, n):
    # Direct sum of ternary tetracode with all words on remaining coords: 9 * 3**(n-4).
    tetracode = _tetracode_words()
    rest = n - 4
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=rest):
            words.append(h + list(tail))
    return words


def _cover_array(codewords, q, n, radius):
    # cov[w] = number of codewords within Hamming distance <= radius of word w.
    m = len(codewords)
    total = q ** n
    if m == 0:
        return np.zeros(total, dtype=np.int32)

    cd = np.array(codewords, dtype=np.int32)  # shape (m, n)
    cov = np.zeros(total, dtype=np.int32)

    for r in range(0, radius + 1):
        # enumerate all subsets of positions of size r (positions where a word may differ)
        if r == 0:
            # exact matches
            idx = cd.reshape(-1)  # flatten base-q digits
            # each codeword is a single index
            for i in range(m):
                v = cd[i, 0]
                for j in range(1, n):
                    v = v * q + cd[i, j]
                cov[v] += 1
            continue
        for pos in itertools.combinations(range(n), r):
            # for each codeword and each assignment of other symbols to these positions
            others = [p for p in range(n) if p not in pos]
            for i in range(m):
                # build prefix over 'others' positions to get base index offset
                # index = sum over positions of digit * q**(n-1-pos)
                # We'll compute base value from 'others' positions, then for each
                # combination of the r positions (each 0..q-1) add contribution.
                base = 0
                for p in others:
                    base = base * q + cd[i, p]
                # But base must account for positions in order; easier: compute full
                # index by iterating all positions, fixing 'pos' to placeholder 0,
                # then add q**(n-1-p) * delta for each changed position.
                # Precompute powers.
                # We'll do it generically:
                pass
            # Simpler approach: iterate all q**r assignments and compute index directly.
            # For speed, precompute the contribution multiplier for each position.
            powq = [q ** (n - 1 - p) for p in range(n)]
            # base index with 'pos' positions set to 0
            for i in range(m):
                base = 0
                for p in range(n):
                    if p in pos:
                        d = 0
                    else:
                        d = cd[i, p]
                    base = base * q + d
                # now for each assignment of the r positions
                for assign in itertools.product(range(q), repeat=r):
                    v = base
                    for k, p in enumerate(pos):
                        v += (assign[k] - 0) * powq[p]
                    cov[v] += 1
    return cov


def _cover_array_fast(codewords, q, n, radius):
    """Compute coverage counts via vectorized numpy where possible."""
    m = len(codewords)
    total = q ** n
    if m == 0:
        return np.zeros(total, dtype=np.int32)

    cd = np.array(codewords, dtype=np.int64)  # (m, n)
    cov = np.zeros(total, dtype=np.int64)

    powq = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    # base index of each codeword
    idx = cd.dot(powq)  # (m,)

    for r in range(0, radius + 1):
        if r == 0:
            np.add.at(cov, idx, 1)
            continue
        for pos in itertools.combinations(range(n), r):
            pos_arr = np.array(pos, dtype=np.int64)
            others = np.array([p for p in range(n) if p not in pos], dtype=np.int64)
            # base index with pos positions zeroed
            base_idx = cd[:, others].dot(powq[others])  # (m,)
            contrib = powq[pos_arr]  # (r,)
            # iterate all q**r assignments
            for assign in itertools.product(range(q), repeat=r):
                assign_arr = np.array(assign, dtype=np.int64)
                offsets = assign_arr.dot(contrib)
                targets = base_idx + offsets
                np.add.at(cov, targets, 1)
    return cov.astype(np.int32)


def _greedy_reduce(codewords, q, n, radius, time_limit=5.0):
    """Remove redundant codewords while coverage is maintained."""
    if not codewords:
        return codewords
    t0 = time.time()
    cov = _cover_array_fast(codewords, q, n, radius)
    changed = True
    while changed and (time.time() - t0) < time_limit:
        changed = False
        for i in range(len(codewords)):
            if (time.time() - t0) >= time_limit:
                break
            # tentatively remove codeword i
            cov_i = _cover_array_fast([codewords[i]], q, n, radius)
            new_cov = cov - cov_i
            if np.all(new_cov >= 1):
                cov = new_cov
                codewords.pop(i)
                changed = True
                break
    return codewords


def construct(q, n, radius):
    global _DEADLINE
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)

    # Start from the tetracode construction (known valid covering).
    codewords = _initial_codewords(q, n)
    best_codewords = list(codewords)
    best_len = len(codewords)

    # Initial coverage check to ensure validity
    cov = _cover_array_fast(codewords, q, n, radius)
    if not np.all(cov >= 1):
        # Should not happen for the tetracode construction, but be safe.
        return best_codewords

    # Greedy reduction first to get a smaller starting point.
    codewords = _greedy_reduce(codewords, q, n, radius, time_limit=3.0)
    if len(codewords) < best_len:
        best_len = len(codewords)
        best_codewords = list(codewords)

    # Simulated annealing
    m = len(codewords)
    # Precompute per-codeword coverage contributions for faster delta computation.
    # cov_total = sum of individual cov arrays.
    individual_covs = [_cover_array_fast([c], q, n, radius) for c in codewords]
    cov_total = np.sum(np.stack(individual_covs), axis=0)

    def _delta_remove(i):
        # If we remove codeword i, how many words become uncovered?
        # A word becomes uncovered if cov_total[w] == 1 and individual_covs[i][w] == 1.
        # i.e., words covered only by codeword i.
        only_by_i = (cov_total == 1) & (individual_covs[i] == 1)
        return int(np.sum(only_by_i))

    def _delta_add(new_word):
        # If we add new_word, how many newly covered words (were uncovered)?
        cov_new = _cover_array_fast([new_word], q, n, radius)
        newly = (cov_total == 0) & (cov_new == 1)
        return int(np.sum(newly))

    def _delta_move(old_word, new_word):
        # Remove old, add new.
        cov_old = individual_covs[_idx_of_old] if False else None
        # We'll compute directly.
        cov_new = _cover_array_fast([new_word], q, n, radius)
        # Words that were covered only by old_word and not by new_word become uncovered.
        # We need individual cov of old_word.
        # This is expensive; approximate by computing full re-eval occasionally.
        return None

    # Simpler SA: operate on a fixed-size set, allow moves that change size.
    # Energy = size + BIG * uncovered_count. We track uncovered_count = sum(cov_total == 0).
    # Use a temperature schedule.

    uncovered = int(np.sum(cov_total == 0))
    energy = m + 10000 * uncovered

    T = 50.0
    T_min = 0.01
    cooling = 0.9995
    rng = np.random.default_rng(42)

    # Precompute individual covs list as numpy arrays.
    ind = individual_covs  # list of arrays

    def _energy():
        return len(codewords) + 10000 * int(np.sum(cov_total == 0))

    steps = 0
    while time.time() < _DEADLINE - 3.0:
        steps += 1
        # Choose a move type
        move_type = rng.choice(3, p=[0.5, 0.3, 0.2])  # remove, add, mutate
        if move_type == 0:  # remove
            if m <= 1:
                continue
            i = int(rng.integers(0, m))
            delta_uncovered = _delta_remove(i)
            delta_size = -1
            delta_energy = delta_size + 10000 * delta_uncovered
            if rng.random() < min(1.0, np.exp(-delta_energy / T)):
                # accept
                cov_total = cov_total - ind[i]
                codewords.pop(i)
                ind.pop(i)
                m -= 1
                energy += delta_energy
        elif move_type == 1:  # add
            # pick a random word
            new_word = [int(rng.integers(0, q)) for _ in range(n)]
            # avoid duplicates
            if any(c == new_word for c in codewords):
                continue
            cov_new = _cover_array_fast([new_word], q, n, radius)
            delta_uncovered = -int(np.sum((cov_total == 0) & (cov_new == 1)))  # newly covered
            delta_size = 1
            delta_energy = delta_size + 10000 * delta_uncovered
            if rng.random() < min(1.0, np.exp(-delta_energy / T)):
                cov_total = cov_total + cov_new
                codewords.append(new_word)
                ind.append(cov_new)
                m += 1
                energy += delta_energy
        else:  # mutate: pick a codeword, change one or two positions
            if m == 0:
                continue
            i = int(rng.integers(0, m))
            old_word = codewords[i]
            new_word = list(old_word)
            # change 1 or 2 random positions
            num_changes = int(rng.integers(1, 3))
            for _ in range(num_changes):
                p = int(rng.integers(0, n))
                new_val = int(rng.integers(0, q))
                if new_val != new_word[p]:
                    new_word[p] = new_val
            # avoid duplicates
            if any(c == new_word for c in codewords):
                continue
            # compute delta: remove old, add new
            cov_new = _cover_array_fast([new_word], q, n, radius)
            # words that become uncovered: covered only by old and not by new
            # words that become covered: uncovered and covered by new
            # delta_uncovered = (newly uncovered) - (newly covered)
            # newly uncovered: cov_total[w]==1 and ind[i][w]==1 and cov_new[w]==0
            newly_uncovered = int(np.sum((cov_total == 1) & (ind[i] == 1) & (cov_new == 0)))
            # newly covered: cov_total[w]==0 and cov_new[w]==1
            newly_covered = int(np.sum((cov_total == 0) & (cov_new == 1)))
            delta_uncovered = newly_uncovered - newly_covered
            delta_size = 0
            delta_energy = delta_size + 10000 * delta_uncovered
            if rng.random() < min(1.0, np.exp(-delta_energy / T)):
                cov_total = cov_total - ind[i] + cov_new
                codewords[i] = new_word
                ind[i] = cov_new
                energy += delta_energy

        # Update best
        if energy < 0 or (cov_total is not None and int(np.sum(cov_total == 0)) == 0 and m < best_len):
            if int(np.sum(cov_total == 0)) == 0 and m < best_len:
                best_len = m
                best_codewords = list(codewords)

        # Cool down
        T = max(T_min, T * cooling)

    # Final greedy reduction on the best valid solution.
    if int(np.sum(cov_total == 0)) == 0:
        # We have a valid current solution; try to reduce it further.
        reduced = _greedy_reduce(list(codewords), q, n, radius, time_limit=3.0)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_codewords = reduced

    # Ensure the best is valid.
    final_cov = _cover_array_fast(best_codewords, q, n, radius)
    if not np.all(final_cov >= 1):
        # Fallback to the initial tetracode construction.
        return _initial_codewords(q, n)

    return best_codewords
# EVOLVE-BLOCK-END
