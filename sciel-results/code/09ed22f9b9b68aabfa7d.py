"""Snake-in-the-box Q_9: combined beam search with bitmask forbidden sets and local refinement."""
import time
import random

N = 9
TOTAL_VERTICES = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# EVOLVE-BLOCK-START

# Precompute neighbor masks: NEIGHBOR_MASKS[v] has bit i set iff v^(1<<b) = i for some b
NEIGHBOR_MASKS = [0] * TOTAL_VERTICES
for v in range(TOTAL_VERTICES):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m


def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True


def _greedy_extend(start, bit_order=None):
    """Greedy extension from a single start vertex using forbidden_mask."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0

    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break

        # Pick the candidate that minimizes future options (Warnsdorff heuristic)
        best_v = None
        best_score = TOTAL_VERTICES + 1
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            new_forbidden = forbidden_mask | NEIGHBOR_MASKS[cur]
            new_used = used_mask | (1 << v)
            next_opts = NEIGHBOR_MASKS[v] & ~(new_used | new_forbidden)
            score = next_opts.bit_count()
            if score < best_score:
                best_score = score
                best_v = v

        if best_v is None:
            break
        path.append(best_v)
        used_mask |= (1 << best_v)
        forbidden_mask |= NEIGHBOR_MASKS[cur]

    return path


def _beam_search(start, beam_width, max_time, rng, diversity_key_len=4):
    """
    Beam search extending from a start vertex using forbidden_mask representation.
    """
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = 1
    t0 = time.time()

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []
        seen_suffixes = set()

        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)

            if candidates == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb

                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]

                key = tuple(new_path[-diversity_key_len:])
                if key in seen_suffixes:
                    continue
                seen_suffixes.add(key)

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        def _score(entry):
            p, used, forb = entry
            cur = p[-1]
            next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
            return (-len(p), next_opts)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _beam_search_from_path(prefix, beam_width, max_time, rng, diversity_key_len=4):
    """
    Beam search continuing from an existing path prefix.
    """
    if not prefix:
        return _beam_search(0, beam_width, max_time, rng, diversity_key_len)

    start = prefix[-1]
    used_mask = 0
    for v in prefix:
        used_mask |= 1 << v
    forbidden_mask = 0
    for v in prefix[:-1]:
        forbidden_mask |= NEIGHBOR_MASKS[v]

    path = list(prefix)
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = len(path)

    while True:
        if time.time() > max_time:
            break
        if not beam:
            break

        new_beam = []
        seen_suffixes = set()

        for cur_path, cur_used, cur_forb in beam:
            cur = cur_path[-1]
            candidates = NEIGHBOR_MASKS[cur] & ~(cur_used | cur_forb)

            if candidates == 0:
                if len(cur_path) > best_len:
                    best_len = len(cur_path)
                    best_path = list(cur_path)
                continue

            tmp = candidates
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb

                new_path = cur_path + [v]
                new_used = cur_used | (1 << v)
                new_forb = cur_forb | NEIGHBOR_MASKS[cur]

                key = tuple(new_path[-diversity_key_len:])
                if key in seen_suffixes:
                    continue
                seen_suffixes.add(key)

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        def _score(entry):
            p, used, forb = entry
            cur = p[-1]
            next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
            return (-len(p), next_opts)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _local_improve(path, deadline, rng):
    """
    Try to improve path by re-extending from various cut points.
    """
    best_path = list(path)
    best_len = len(path)
    n = len(path)

    if n < 4:
        return best_path

    positions = []
    for i in range(max(1, n // 2), n - 1):
        positions.append(i)
    for i in range(1, max(1, n // 4)):
        positions.append(i)
    rng.shuffle(positions)

    for pos in positions[:15]:
        if time.time() > deadline - 3:
            break
        prefix = path[:pos + 1]
        remaining = deadline - time.time()
        if remaining < 2:
            break
        dur = min(3.0, remaining - 1)
        improved = _beam_search_from_path(prefix, 40, time.time() + dur, rng)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path


def _backtrack_improve(path, deadline, rng):
    """
    Try to improve by backtracking several steps and re-exploring.
    """
    best_path = list(path)
    best_len = len(path)
    n = len(path)

    if n < 5:
        return best_path

    backtrack_amounts = [3, 5, 7, 10, 15]
    rng.shuffle(backtrack_amounts)

    for bt in backtrack_amounts:
        if time.time() > deadline - 5:
            break
        if bt >= n - 1:
            continue
        cut_pos = n - 1 - bt
        prefix = path[:cut_pos + 1]
        remaining = deadline - time.time()
        if remaining < 3:
            break
        dur = min(4.0, remaining - 2)
        improved = _beam_search_from_path(prefix, 60, time.time() + dur, rng)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    rng = random.Random(42)

    best_path = None
    best_len = 0

    # Phase 1: Greedy seed from vertex 0
    greedy = _greedy_extend(0)
    best_path = greedy
    best_len = len(greedy)

    # Phase 2: Beam search from vertex 0 with increasing widths
    beam_configs = [
        (50, 8.0),
        (100, 10.0),
        (200, 12.0),
        (400, 15.0),
    ]
    for bw, duration in beam_configs:
        if time.time() > deadline - 12:
            break
        remaining = deadline - time.time()
        if remaining < duration + 2:
            duration = max(2.0, remaining - 2)
        path = _beam_search(0, bw, time.time() + duration, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 3: Beam search from different starting vertices
    start_vertices = [1, 2, 3, 5, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191]
    for sv in start_vertices:
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        duration = min(4.0, remaining - 2)
        path = _beam_search(sv, 80, time.time() + duration, rng)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path

    # Phase 4: Local improvement on the best path found
    if best_path and len(best_path) > 10:
        if time.time() > deadline - 8:
            pass
        else:
            improved = _local_improve(best_path, deadline, rng)
            if len(improved) > best_len:
                best_len = len(improved)
                best_path = improved

    # Phase 5: Backtrack improvement
    if best_path and len(best_path) > 10 and time.time() < deadline - 6:
        improved = _backtrack_improve(best_path, deadline, rng)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Final validation
    if best_path and _is_valid_snake(best_path):
        return best_path

    # Fallback: simple greedy from vertex 0
    path, used = [0], {0}
    forbidden = 0
    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used | forbidden)
        if candidates == 0:
            return path
        # Pick first candidate
        lsb = candidates & (-candidates)
        v = lsb.bit_length() - 1
        path.append(v)
        used |= (1 << v)
        forbidden |= NEIGHBOR_MASKS[cur]

# EVOLVE-BLOCK-END
