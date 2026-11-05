"""Structural rewrite: vectorized greedy covering-code construction with
count-based local search, precomputed ball arrays, and a kick mechanism.

The entry point is construct(q, n, radius) -> list of codewords.
"""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock


def _ball_indices(codeword, q, n, radius):
    """Return the set of indices (as a numpy int64 array) within Hamming
    distance <= radius of codeword."""
    m = len(codeword)
    # Precompute powers of q for position weights
    powq = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    base = 0
    for p in range(n):
        base = base * q + codeword[p]
    indices = [base]
    for r in range(1, radius + 1):
        for pos in itertools.combinations(range(n), r):
            others = [p for p in range(n) if p not in pos]
            # base with pos positions zeroed
            base_val = 0
            for p in others:
                base_val = base_val * q + codeword[p]
            # For each assignment of the r positions
            for assign in itertools.product(range(q), repeat=r):
                v = base_val
                for k, p in enumerate(pos):
                    v += assign[k] * powq[p]
                indices.append(v)
    return np.array(indices, dtype=np.int64)


def _ball_indices_fast(codeword, q, n, radius):
    """Faster ball computation using vectorization over assignments."""
    powq = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    base = 0
    for p in range(n):
        base = base * q + codeword[p]
    result = [base]
    for r in range(1, radius + 1):
        for pos in itertools.combinations(range(n), r):
            pos_arr = np.array(pos, dtype=np.int64)
            others = [p for p in range(n) if p not in pos]
            # base with pos positions zeroed
            base_val = 0
            for p in others:
                base_val = base_val * q + codeword[p]
            # Generate all q**r assignments
            assignments = np.array(list(itertools.product(range(q), repeat=r)), dtype=np.int64)
            contrib = powq[pos_arr]  # (r,)
            offsets = assignments.dot(contrib)  # (q**r,)
            targets = base_val + offsets
            result.extend(targets.tolist())
    return np.array(result, dtype=np.int64)


def _initial_codewords(q, n):
    """Tetracode direct-sum construction."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    rest = n - 4
    words = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=rest):
            words.append(h + list(tail))
    return words


def _cover_array_fast(codewords, q, n, radius):
    """Compute coverage counts for all words in the space."""
    m = len(codewords)
    total = q ** n
    if m == 0:
        return np.zeros(total, dtype=np.int32)
    cd = np.array(codewords, dtype=np.int64)
    cov = np.zeros(total, dtype=np.int64)
    powq = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
    idx = cd.dot(powq)
    for r in range(0, radius + 1):
        if r == 0:
            np.add.at(cov, idx, 1)
            continue
        for pos in itertools.combinations(range(n), r):
            pos_arr = np.array(pos, dtype=np.int64)
            others = np.array([p for p in range(n) if p not in pos], dtype=np.int64)
            base_idx = cd[:, others].dot(powq[others])
            contrib = powq[pos_arr]
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
    # Precompute individual balls for faster delta
    balls = [_ball_indices_fast(c, q, n, radius) for c in codewords]
    changed = True
    while changed and (time.time() - t0) < time_limit:
        changed = False
        for i in range(len(codewords)):
            if (time.time() - t0) >= time_limit:
                break
            # Check if removing codeword i leaves all words covered
            # A word becomes uncovered iff cov[w] == 1 and balls[i] contains w
            ball_mask = np.zeros(q ** n, dtype=np.int32)
            ball_mask[balls[i]] = 1
            newly_uncovered = np.sum((cov == 1) & (ball_mask == 1))
            if newly_uncovered == 0:
                cov[balls[i]] -= 1
                codewords.pop(i)
                balls.pop(i)
                changed = True
                break
    return codewords


def _greedy_add(codewords, q, n, radius, time_limit=5.0):
    """Add codewords to cover uncovered words, choosing the one that covers
    the most currently-uncovered words."""
    if not codewords:
        return codewords
    t0 = time.time()
    total = q ** n
    cov = _cover_array_fast(codewords, q, n, radius)
    uncovered = np.sum(cov == 0)
    while uncovered > 0 and (time.time() - t0) < time_limit:
        # Find an uncovered word
        uncov_indices = np.where(cov == 0)[0]
        if len(uncov_indices) == 0:
            break
        # Pick a random uncovered word
        target_idx = int(uncov_indices[int(random.random() * len(uncov_indices))])
        # Convert to codeword
        target_word = []
        tmp = target_idx
        for p in range(n - 1, -1, -1):
            target_word.append(tmp % q)
            tmp //= q
        target_word.reverse()
        # Try adding target_word or a nearby word that covers the most uncovered
        # For simplicity, just add the target word itself
        ball = _ball_indices_fast(target_word, q, n, radius)
        cov[ball] += 1
        codewords.append(target_word)
        uncovered = np.sum(cov == 0)
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

    # Initial coverage check
    cov = _cover_array_fast(codewords, q, n, radius)
    if not np.all(cov >= 1):
        return best_codewords

    # Phase 1: Greedy reduction
    codewords = _greedy_reduce(codewords, q, n, radius, time_limit=5.0)
    if len(codewords) < best_len:
        best_len = len(codewords)
        best_codewords = list(codewords)

    # Phase 2: Local search with kicks
    # Use count-based approach: track coverage counts, try to remove/add/mutate
    total = q ** n
    cov = _cover_array_fast(codewords, q, n, radius)
    m = len(codewords)

    # Precompute balls for current codewords
    balls = [_ball_indices_fast(c, q, n, radius) for c in codewords]

    T = 10.0
    T_min = 0.01
    cooling = 0.999
    rng = np.random.default_rng(42)

    # Energy: size + BIG * uncovered
    BIG = 10000
    uncovered_count = int(np.sum(cov == 0))
    energy = m + BIG * uncovered_count

    steps = 0
    stall_count = 0
    stall_threshold = 500

    while time.time() < _DEADLINE - 5.0:
        steps += 1

        # Choose move type
        move_type = rng.choice(3, p=[0.4, 0.3, 0.3])

        if move_type == 0:  # remove
            if m <= 1:
                continue
            i = int(rng.integers(0, m))
            # Compute delta: how many words become uncovered
            # A word becomes uncovered iff cov[w] == 1 and balls[i] contains w
            ball_mask = np.zeros(total, dtype=np.int32)
            ball_mask[balls[i]] = 1
            newly_uncovered = int(np.sum((cov == 1) & (ball_mask == 1)))
            delta_energy = -1 + BIG * newly_uncovered
            if rng.random() < min(1.0, np.exp(-delta_energy / T)):
                cov[balls[i]] -= 1
                codewords.pop(i)
                balls.pop(i)
                m -= 1
                energy += delta_energy
                stall_count = 0
        elif move_type == 1:  # add
            # Pick a random word, prefer uncovered areas
            if rng.random() < 0.7:
                # Target an uncovered word
                uncov = np.where(cov == 0)[0]
                if len(uncov) > 0:
                    target_idx = int(uncov[int(rng.integers(0, len(uncov)))])
                else:
                    target_idx = int(rng.integers(0, total))
            else:
                target_idx = int(rng.integers(0, total))
            # Convert to codeword
            target_word = []
            tmp = target_idx
            for p in range(n - 1, -1, -1):
                target_word.append(tmp % q)
                tmp //= q
            target_word.reverse()
            # Check duplicate
            if any(c == target_word for c in codewords):
                continue
            ball = _ball_indices_fast(target_word, q, n, radius)
            # Compute delta: how many newly covered
            newly_covered = int(np.sum((cov == 0) & (np.isin(np.arange(total), ball))))
            # More efficient:
            ball_mask = np.zeros(total, dtype=np.int32)
            ball_mask[ball] = 1
            newly_covered = int(np.sum((cov == 0) & (ball_mask == 1)))
            delta_energy = 1 - BIG * newly_covered
            if rng.random() < min(1.0, np.exp(-delta_energy / T)):
                cov[ball] += 1
                codewords.append(target_word)
                balls.append(ball)
                m += 1
                energy += delta_energy
                stall_count = 0
        else:  # mutate
            if m == 0:
                continue
            i = int(rng.integers(0, m))
            old_word = codewords[i]
            new_word = list(old_word)
            num_changes = int(rng.integers(1, 3))
            for _ in range(num_changes):
                p = int(rng.integers(0, n))
                new_val = int(rng.integers(0, q))
                if new_val != new_word[p]:
                    new_word[p] = new_val
            if any(c == new_word for c in codewords):
                continue
            new_ball = _ball_indices_fast(new_word, q, n, radius)
            # Delta: remove old ball, add new ball
            old_ball = balls[i]
            # Words that become uncovered: cov == 1 and in old_ball and not in new_ball
            new_ball_mask = np.zeros(total, dtype=np.int32)
            new_ball_mask[new_ball] = 1
            newly_uncovered = int(np.sum((cov == 1) & (np.isin(np.arange(total), old_ball)) & (new_ball_mask == 0)))
            # More efficient:
            old_ball_mask = np.zeros(total, dtype=np.int32)
            old_ball_mask[old_ball] = 1
            newly_uncovered = int(np.sum((cov == 1) & (old_ball_mask == 1) & (new_ball_mask == 0)))
            newly_covered = int(np.sum((cov == 0) & (new_ball_mask == 1)))
            delta_energy = BIG * (newly_uncovered - newly_covered)
            if rng.random() < min(1.0, np.exp(-delta_energy / T)):
                cov[old_ball] -= 1
                cov[new_ball] += 1
                codewords[i] = new_word
                balls[i] = new_ball
                energy += delta_energy
                stall_count = 0

        # Update best
        if uncovered_count != int(np.sum(cov == 0)):
            uncovered_count = int(np.sum(cov == 0))
        if uncovered_count == 0 and m < best_len:
            best_len = m
            best_codewords = list(codewords)

        # Kick mechanism: if stalled, do a large random perturbation
        stall_count += 1
        if stall_count > stall_threshold:
            # Randomly mutate several codewords
            for _ in range(min(5, m // 4)):
                j = int(rng.integers(0, m))
                old_word = codewords[j]
                new_word = [int(rng.integers(0, q)) for _ in range(n)]
                if any(c == new_word for c in codewords):
                    continue
                new_ball = _ball_indices_fast(new_word, q, n, radius)
                old_ball = balls[j]
                cov[old_ball] -= 1
                cov[new_ball] += 1
                codewords[j] = new_word
                balls[j] = new_ball
            stall_count = 0
            T = max(T_min, T * 1.5)  # reheat

        # Cool down
        T = max(T_min, T * cooling)

    # Final greedy reduction on the best valid solution
    if int(np.sum(cov == 0)) == 0:
        reduced = _greedy_reduce(list(codewords), q, n, radius, time_limit=3.0)
        if len(reduced) < best_len:
            best_len = len(reduced)
            best_codewords = reduced

    # Ensure the best is valid
    final_cov = _cover_array_fast(best_codewords, q, n, radius)
    if not np.all(final_cov >= 1):
        return _initial_codewords(q, n)

    return best_codewords
# EVOLVE-BLOCK-END
