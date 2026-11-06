"""Snake-in-the-box Q_9: forbidden-mask beam search with phased multi-restart and local improvement."""
import time
import random

N = 9
SIZE = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks: NEIGHBOR_MASKS[v] has bit i set iff v^i is a neighbor of v
NEIGHBOR_MASKS = [0] * SIZE
for v in range(SIZE):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ BIT_MASKS[b])
    NEIGHBOR_MASKS[v] = m


def _greedy_extend(start, bit_order=None):
    """Greedy extension from a single start vertex using forbidden_mask."""
    if bit_order is None:
        bit_order = list(range(N))
    path = [start]
    used_mask = 1 << start
    forbidden_mask = 0  # neighbors of all non-last vertices in path

    while True:
        cur = path[-1]
        # Candidates: neighbors of cur, not used, not forbidden
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break

        # Pick candidate with fewest future options (most constrained)
        best_v = None
        best_score = SIZE + 1
        tmp = candidates
        while tmp:
            lsb = tmp & (-tmp)
            v = lsb.bit_length() - 1
            tmp ^= lsb
            # If we go to v, cur becomes non-last, so its neighbors become forbidden
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


def _beam_search(start, beam_width, max_time, rng):
    """
    Beam search extending from a start vertex using forbidden_mask representation.
    State: (path_list, used_mask, forbidden_mask)
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

                # Diversity: avoid identical last-4 suffixes
                key = tuple(new_path[-4:])
                if key in seen_suffixes:
                    continue
                seen_suffixes.add(key)

                new_beam.append((new_path, new_used, new_forb))
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = list(new_path)

        if not new_beam:
            break

        # Sort by length desc, then by next_options asc (more constrained = better)
        def _score(entry):
            p, used, forb = entry
            cur = p[-1]
            next_opts = (NEIGHBOR_MASKS[cur] & ~(used | forb)).bit_count()
            return (-len(p), next_opts)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _beam_search_from_path(prefix, beam_width, max_time, rng):
    """
    Beam search continuing from an existing path prefix.
    prefix: list of vertices forming a valid snake so far.
    """
    if not prefix:
        return _beam_search(0, beam_width, max_time, rng)

    start = prefix[-1]
    used_mask = 0
    for v in prefix:
        used_mask |= 1 << v
    # forbidden_mask: neighbors of all vertices except the last
    forbidden_mask = 0
    for v in prefix[:-1]:
        forbidden_mask |= NEIGHBOR_MASKS[v]

    path = list(prefix)
    beam = [(path, used_mask, forbidden_mask)]
    best_path = list(path)
    best_len = len(path)
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

                key = tuple(new_path[-4:])
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
    For each cut position, take the prefix, run a short beam search, and keep the best.
    """
    best_path = list(path)
    best_len = len(path)
    n = len(path)

    if n < 4:
        return best_path

    # Try several cut positions
    positions = []
    # Bias toward later positions (keep more of the path)
    for i in range(max(1, n // 2), n - 1):
        positions.append(i)
    # Also try some earlier positions
    for i in range(1, max(1, n // 4)):
        positions.append(i)
    rng.shuffle(positions)

    for pos in positions[:20]:
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


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s margin
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
        t0 = time.time()
        result = _beam_search(0, bw, t0 + duration, rng)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result

    # Phase 3: Beam search from the greedy path (extend the greedy result)
    if time.time() < deadline - 15 and best_len < 200:
        remaining = deadline - time.time()
        if remaining > 10:
            dur = min(15.0, remaining - 5)
            result = _beam_search_from_path(best_path, 100, time.time() + dur, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 4: Multi-restart from random starting vertices
    if time.time() < deadline - 15:
        starts = list(range(SIZE))
        rng.shuffle(starts)
        for start in starts[:40]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time()
            if remaining < 4:
                break
            bw = 50
            dur = min(3.5, remaining - 1)
            result = _beam_search(start, bw, time.time() + dur, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 5: Local improvement on the best path found
    if time.time() < deadline - 8:
        improved = _local_improve(best_path, deadline, rng)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 6: Final large beam if time remains
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 10:
            bw = 500
            dur = remaining - 5
            result = _beam_search(0, bw, time.time() + dur, rng)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 7: One more local improvement pass if time allows
    if time.time() < deadline - 5:
        improved = _local_improve(best_path, deadline, rng)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path
