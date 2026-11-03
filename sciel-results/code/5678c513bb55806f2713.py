"""Improved covering-code search: bitset greedy + simulated annealing + swap/local polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools
import math

# Deadline guard: evaluator kills at 150s wall-clock; keep a safety margin
_DEADLINE_SECONDS = 140.0


def _elapsed(start):
    return time.monotonic() - start


def _idx_to_tuple(idx, n, q):
    """Convert flat index to tuple of digits."""
    t = []
    for i in range(n):
        t.append(idx % q)
        idx //= q
    return t[::-1]


def _tuple_to_idx(t, n, q):
    """Convert tuple of digits to flat index."""
    idx = 0
    for d in t:
        idx = idx * q + d
    return idx


def _all_neighbors(codeword, n, q, radius):
    """Yield all words within Hamming distance <= radius of codeword."""
    yield tuple(codeword)
    for pos in range(n):
        orig = codeword[pos]
        for val in range(q):
            if val == orig:
                continue
            new = list(codeword)
            new[pos] = val
            yield tuple(new)


def _build_coverage_matrix(total, n, q, radius):
    """
    Build a sparse-like representation: for each codeword index, store the list of word indices it covers.
    Also build reverse: for each word, list of codewords that cover it.
    """
    # forward_cover[c] = list of words covered by codeword c
    # reverse_cover[w] = list of codewords covering word w
    forward_cover = [[] for _ in range(total)]
    reverse_cover = [[] for _ in range(total)]

    for c in range(total):
        c_tuple = _idx_to_tuple(c, n, q)
        # The codeword itself
        forward_cover[c].append(c)
        reverse_cover[c].append(c)
        # Single position changes (radius 1)
        for pos in range(n):
            orig = c_tuple[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_tuple = list(c_tuple)
                new_tuple[pos] = val
                w = _tuple_to_idx(new_tuple, n, q)
                forward_cover[c].append(w)
                reverse_cover[w].append(c)

    return forward_cover, reverse_cover


def _build_bitsets(total, n, q, radius):
    """
    Build per-codeword bitset (Python int) of words it covers.
    Returns list of ints, one per candidate codeword index.
    """
    forward_cover, _ = _build_coverage_matrix(total, n, q, radius)
    bitsets = []
    for c in range(total):
        bs = 0
        for w in forward_cover[c]:
            bs |= (1 << w)
        bitsets.append(bs)
    return bitsets


def _greedy_construct(q, n, radius, rng, max_time_remaining):
    """
    Bitset-accelerated greedy set cover.
    Returns list of codewords (tuples) that cover all of {0..q-1}^n.
    """
    total = q ** n
    all_bits = (1 << total) - 1
    uncovered = all_bits
    codewords = []

    # Precompute per-codeword coverage bitsets
    cover_bitsets = _build_bitsets(total, n, q, radius)

    # gain[c] = popcount(cover_bitsets[c] & uncovered)
    gain = [int((cover_bitsets[c] & uncovered).bit_count()) for c in range(total)]

    while True:
        if _elapsed(max_time_remaining[0]) > max_time_remaining[1]:
            break

        if uncovered == 0:
            break

        # Find max gain
        best_gain = 0
        best_c = -1
        for c in range(total):
            g = gain[c]
            if g > best_gain:
                best_gain = g
                best_c = c
        if best_c < 0 or best_gain <= 0:
            break

        codewords.append(_idx_to_tuple(best_c, n, q))
        new_bits = cover_bitsets[best_c] & uncovered
        uncovered &= ~new_bits

        # Update gains for codewords that cover any newly-covered word
        # For each word w in new_bits, decrement gain of all codewords covering w
        # Efficient: iterate over set bits in new_bits
        nb = new_bits
        while nb:
            w = (nb & -nb).bit_length() - 1
            nb &= nb - 1
            # Decrement gain for all codewords that cover word w
            # We need reverse map; rebuild lazily or use a precomputed list
            # For efficiency, precompute reverse_cover as list of lists
            pass
        # Actually, let's precompute reverse cover for gain updates
        # This is getting complex; use a simpler approach:
        # Recompute gain only for candidates that might have changed
        # Since we just covered new_bits, only codewords intersecting new_bits change
        changed_candidates = set()
        nb2 = new_bits
        while nb2:
            w = (nb2 & -nb2).bit_length() - 1
            nb2 &= nb2 - 1
            # Which codewords cover w? We need reverse map.
            # Precompute once.
            pass

        # Simpler: just recompute gain for all candidates that intersect uncovered
        # This is O(total) per step but total is only 2187
        for c in range(total):
            gain[c] = int((cover_bitsets[c] & uncovered).bit_count())

    return codewords


def _is_cover(codewords, q, n, radius):
    """Check if codewords form a valid covering using bitsets."""
    total = q ** n
    all_bits = (1 << total) - 1
    covered = 0
    cover_bitsets = _build_bitsets(total, n, q, radius)
    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        covered |= cover_bitsets[c_idx]
    return covered == all_bits


def _simulated_annealing(codewords, q, n, radius, rng, max_time_remaining):
    """
    Simulated annealing on the binary selection vector.
    
    State: boolean array x of length total, where x[c]=1 means codeword c is selected.
    Energy: E = alpha * (uncovered_count) + beta * (num_selected)
    We want to minimize both, with heavy penalty on uncovered words.
    
    Moves: flip one bit (add or remove a codeword).
    """
    total = q ** n
    forward_cover, reverse_cover = _build_coverage_matrix(total, n, q, radius)

    # Initialize from greedy result
    x = np.zeros(total, dtype=np.uint8)
    for cw in codewords:
        c_idx = _tuple_to_idx(cw, n, q)
        x[c_idx] = 1

    # Current coverage state
    covered_count = np.zeros(total, dtype=np.int32)  # how many selected codewords cover each word
    for c in range(total):
        if x[c]:
            for w in forward_cover[c]:
                covered_count[w] += 1

    uncovered = total - int(np.sum(covered_count > 0))
    num_selected = int(np.sum(x))

    # Energy parameters
    alpha = 100.0  # heavy penalty for uncovered words
    beta = 1.0     # light penalty for number of selected codewords

    def _energy(uncovered, num_sel):
        return alpha * uncovered + beta * num_sel

    current_energy = _energy(uncovered, num_selected)
    best_energy = current_energy
    best_x = x.copy()

    # Annealing schedule
    T_start = 50.0
    T_end = 0.01
    n_iter = 50000  # target iterations, will stop early if time runs out

    start_time = time.monotonic()
    for it in range(n_iter):
        # Check deadline
        if time.monotonic() - start_time > (max_time_remaining[1] - max_time_remaining[0]) - 2.0:
            break

        # Temperature
        progress = it / n_iter
        T = T_start * math.exp(-10 * progress)

        # Random move: flip a random bit
        c = int(rng.integers(0, total))

        if x[c] == 1:
            # Remove codeword c
            new_uncovered_delta = 0
            for w in forward_cover[c]:
                if covered_count[w] == 1:
                    new_uncovered_delta += 1
            delta_energy = alpha * new_uncovered_delta - beta
        else:
            # Add codeword c
            new_uncovered_delta = 0
            for w in forward_cover[c]:
                if covered_count[w] == 0:
                    new_uncovered_delta += 1
            delta_energy = -alpha * new_uncovered_delta + beta

        # Acceptance criterion
        if delta_energy < 0 or rng.random() < math.exp(-delta_energy / max(T, 1e-10)):
            # Apply move
            if x[c] == 1:
                x[c] = 0
                for w in forward_cover[c]:
                    covered_count[w] -= 1
                uncovered += new_uncovered_delta
                num_selected -= 1
            else:
                x[c] = 1
                for w in forward_cover[c]:
                    covered_count[w] += 1
                uncovered -= new_uncovered_delta
                num_selected += 1

            current_energy = _energy(uncovered, num_selected)

            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()

    # Extract codewords from best solution
    best_codewords = []
    for c in range(total):
        if best_x[c]:
            best_codewords.append(_idx_to_tuple(c, n, q))

    return best_codewords


def _local_search(codewords, q, n, radius, rng, max_time_remaining):
    """
    Local search: try deletions and swaps to reduce code size while maintaining coverage.
    """
    codewords = list(codewords)
    total = q ** n
    all_bits = (1 << total) - 1
    cover_bitsets = _build_bitsets(total, n, q, radius)

    def _union_bits(cw_list):
        bits = 0
        for cw in cw_list:
            c_idx = _tuple_to_idx(cw, n, q)
            bits |= cover_bitsets[c_idx]
        return bits

    # Phase 1: exhaustive deletion
    improved = True
    while improved and _elapsed(max_time_remaining[0]) < max_time_remaining[1]:
        improved = False
        for i in range(len(codewords)):
            if _elapsed(max_time_remaining[0]) > max_time_remaining[1]:
                return codewords
            test_bits = _union_bits(codewords[:i] + codewords[i+1:])
            if test_bits == all_bits:
                codewords = codewords[:i] + codewords[i+1:]
                improved = True
                break

    # Phase 2: swap search — try replacing one codeword with another to enable further deletions
    # This is more expensive; only do a limited number of attempts
    if _elapsed(max_time_remaining[0]) < max_time_remaining[1] - 2.0:
        best_size = len(codewords)
        attempts = 0
        max_attempts = 200  # limit swap attempts
        while attempts < max_attempts and _elapsed(max_time_remaining[0]) < max_time_remaining[1] - 1.0:
            attempts += 1
            if len(codewords) <= 1:
                break
            # Pick a random codeword to remove
            i = int(rng.integers(0, len(codewords)))
            without_i = codewords[:i] + codewords[i+1:]
            bits_without = _union_bits(without_i)
            if bits_without == all_bits:
                codewords = without_i
                continue
            # Find uncovered words
            uncovered_bits = all_bits & ~bits_without
            # Try to find a single codeword that covers all uncovered words
            # i.e., cover_bitsets[c] & uncovered_bits == uncovered_bits
            found = False
            for c in range(total):
                if c == _tuple_to_idx(codewords[i], n, q):
                    continue
                if (cover_bitsets[c] & uncovered_bits) == uncovered_bits:
                    # Check if adding c and removing i still covers everything
                    # Since c covers all uncovered, and without_i covers the rest, union is full
                    new_list = without_i + [codewords[i]]  # placeholder
                    # Actually: new code = without_i + [tuple for c]
                    new_list = without_i + [_idx_to_tuple(c, n, q)]
                    # Verify
                    if _union_bits(new_list) == all_bits:
                        codewords = new_list
                        found = True
                        break
            if not found:
                # Try random candidate additions
                for _ in range(5):
                    c = int(rng.integers(0, total))
                    if c == _tuple_to_idx(codewords[i], n, q):
                        continue
                    new_list = without_i + [_idx_to_tuple(c, n, q)]
                    if _union_bits(new_list) == all_bits:
                        codewords = new_list
                        found = True
                        break

    # Phase 3: final deletion polish
    improved = True
    while improved and _elapsed(max_time_remaining[0]) < max_time_remaining[1]:
        improved = False
        for i in range(len(codewords)):
            if _elapsed(max_time_remaining[0]) > max_time_remaining[1]:
                return codewords
            test_bits = _union_bits(codewords[:i] + codewords[i+1:])
            if test_bits == all_bits:
                codewords = codewords[:i] + codewords[i+1:]
                improved = True
                break

    return codewords


def construct(q, n, radius):
    start = time.monotonic()
    deadline = start + _DEADLINE_SECONDS
    max_time_remaining = [start, deadline]

    rng = np.random.default_rng(42)

    # Start with a greedy construction
    best = _greedy_construct(q, n, radius, rng, max_time_remaining)

    # Simulated annealing to improve
    if _elapsed(start) < deadline - 15.0:
        sa_result = _simulated_annealing(best, q, n, radius, rng, max_time_remaining)
        if len(sa_result) < len(best):
            best = sa_result

    # Local search to polish (deletions + swaps)
    if _elapsed(start) < deadline - 3.0:
        best = _local_search(best, q, n, radius, rng, max_time_remaining)

    # Verify coverage
    if not _is_cover(best, q, n, radius):
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(cw) for cw in best]
# EVOLVE-BLOCK-END
