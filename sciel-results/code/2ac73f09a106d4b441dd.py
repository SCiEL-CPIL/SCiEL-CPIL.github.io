"""Snake-in-the-box Q_9: Optimized Beam Search with 2-ply Lookahead and Adaptive Phasing."""
import time
import random

N = 9
SIZE = 1 << N
BIT_MASKS = [1 << b for b in range(N)]

# Precompute neighbor masks
# NEIGHBOR_MASKS[v] is a bitmask of all vertices adjacent to v
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
    forbidden_mask = 0

    while True:
        cur = path[-1]
        candidates = NEIGHBOR_MASKS[cur] & ~(used_mask | forbidden_mask)
        if candidates == 0:
            break

        best_v = None
        best_score = SIZE + 1
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


def _beam_search_with_lookahead(start, beam_width, max_time, rng, alpha=1.0, beta=0.5, gamma=0.1):
    """
    Beam search with 2-ply lookahead scoring.
    alpha: weight for path length
    beta: weight for immediate next options
    gamma: weight for 2-ply lookahead (min options of next step)
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
            
            # 1-ply lookahead: options from cur
            next_opts_mask = NEIGHBOR_MASKS[cur] & ~(used | forb)
            opt_count = next_opts_mask.bit_count()
            
            # 2-ply lookahead: min options from the best next step
            # We estimate the "potential" by looking at the most constrained neighbor
            min_next_opt = SIZE
            if opt_count > 0:
                tmp = next_opts_mask
                # To keep it fast, we only check a few neighbors if there are many
                # But for Q9, max 9 neighbors, so checking all is fine
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    # Calculate options for v
                    v_opts_mask = NEIGHBOR_MASKS[v] & ~(used | (1 << v) | forb)
                    v_opts_count = v_opts_mask.bit_count()
                    if v_opts_count < min_next_opt:
                        min_next_opt = v_opts_count
            
            if min_next_opt == SIZE:
                min_next_opt = 0
                
            return (-len(p) * alpha + opt_count * beta + min_next_opt * gamma)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _beam_search_from_path(prefix, beam_width, max_time, rng, alpha=1.0, beta=0.5, gamma=0.1):
    """Beam search continuing from an existing path prefix."""
    if not prefix:
        return _beam_search_with_lookahead(0, beam_width, max_time, rng, alpha, beta, gamma)

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
            next_opts_mask = NEIGHBOR_MASKS[cur] & ~(used | forb)
            opt_count = next_opts_mask.bit_count()
            
            min_next_opt = SIZE
            if opt_count > 0:
                tmp = next_opts_mask
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    tmp ^= lsb
                    v_opts_mask = NEIGHBOR_MASKS[v] & ~(used | (1 << v) | forb)
                    v_opts_count = v_opts_mask.bit_count()
                    if v_opts_count < min_next_opt:
                        min_next_opt = v_opts_count
            
            if min_next_opt == SIZE:
                min_next_opt = 0
                
            return (-len(p) * alpha + opt_count * beta + min_next_opt * gamma)

        new_beam.sort(key=_score)
        beam = new_beam[:beam_width]

    return best_path


def _local_improve(path, deadline, rng, alpha=1.0, beta=0.5, gamma=0.1):
    """Try to improve path by re-extending from various cut points."""
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

    for pos in positions[:20]:
        if time.time() > deadline - 3:
            break
        prefix = path[:pos + 1]
        remaining = deadline - time.time()
        if remaining < 2:
            break
        dur = min(3.0, remaining - 1)
        improved = _beam_search_from_path(prefix, 50, time.time() + dur, rng, alpha, beta, gamma)
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

    # Phase 1: Greedy seed
    greedy = _greedy_extend(0)
    best_path = greedy
    best_len = len(greedy)

    # Phase 2: Beam search with Lookahead
    # We run multiple configurations. The lookahead makes the search slower per step,
    # so we might use slightly smaller beam widths or shorter durations to fit more iterations.
    
    # Configuration 1: Moderate beam, high lookahead weight
    if time.time() < deadline - 15:
        t0 = time.time()
        duration = 10.0
        result = _beam_search_with_lookahead(0, 80, t0 + duration, rng, 1.0, 0.5, 0.2)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result

    # Configuration 2: Wider beam, lower lookahead weight (faster)
    if time.time() < deadline - 15:
        t0 = time.time()
        duration = 10.0
        result = _beam_search_with_lookahead(0, 150, t0 + duration, rng, 1.0, 0.3, 0.1)
        if len(result) > best_len:
            best_len = len(result)
            best_path = result

    # Phase 3: Extend the best path found
    if time.time() < deadline - 15 and best_len < 200:
        remaining = deadline - time.time()
        if remaining > 10:
            dur = min(15.0, remaining - 5)
            result = _beam_search_from_path(best_path, 100, time.time() + dur, rng, 1.0, 0.5, 0.2)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 4: Multi-restart with varied parameters
    if time.time() < deadline - 15:
        starts = list(range(SIZE))
        rng.shuffle(starts)
        restart_configs = [
            (60, 3.0, 1.0, 0.5, 0.2),
            (100, 2.5, 1.0, 0.3, 0.1),
            (80, 2.5, 1.2, 0.4, 0.15),
        ]
        for start in starts[:20]:
            if time.time() > deadline - 5:
                break
            remaining = deadline - time.time()
            if remaining < 3:
                break
            for bw, dur, alpha, beta, gamma in restart_configs:
                if time.time() > deadline - 3:
                    break
                if remaining < dur:
                    dur = max(1.5, remaining - 1)
                result = _beam_search_with_lookahead(start, bw, time.time() + dur, rng, alpha, beta, gamma)
                if len(result) > best_len:
                    best_len = len(result)
                    best_path = result
                if time.time() > deadline - 5:
                    break

    # Phase 5: Local improvement
    if time.time() < deadline - 8:
        improved = _local_improve(best_path, deadline, rng, 1.0, 0.5, 0.2)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    # Phase 6: Final large beam with tuned parameters
    if time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 10:
            bw = 200
            dur = remaining - 5
            result = _beam_search_with_lookahead(0, bw, time.time() + dur, rng, 1.5, 0.2, 0.1)
            if len(result) > best_len:
                best_len = len(result)
                best_path = result

    # Phase 7: One more local improvement
    if time.time() < deadline - 5:
        improved = _local_improve(best_path, deadline, rng, 1.2, 0.3, 0.1)
        if len(improved) > best_len:
            best_len = len(improved)
            best_path = improved

    return best_path
