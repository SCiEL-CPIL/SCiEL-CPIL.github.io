import time
import random
import math

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0  # Safety margin from 90s limit

# ---------------------------------------------------------------------------
# Pre-compute bit masks at module load (verification is fast bit ops)
# ---------------------------------------------------------------------------
def _build_mask(w, n_bits):
    block = 1 << w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = n_bits // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    return m

_INITIAL_MASKS = [0] * N
for _w in range(N):
    _INITIAL_MASKS[_w] = _build_mask(_w, 1 << N)


def _verify(comparators):
    """Verify the comparator list sorts all 2**N boolean inputs (bitmask method)."""
    state = _INITIAL_MASKS[:]
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
    for i in range(N - 1):
        if (state[i] & ~state[i + 1]) != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Network constructions
# ---------------------------------------------------------------------------
def _odd_even_transposition():
    """Seed network: odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]


def _batcher_odd_even_merge(n):
    """Odd-even merge of n (power-of-2) already-sorted halves. Returns (i,j) list."""
    if n <= 1:
        return []
    result = []

    def _oem(k, offset):
        if k <= 1:
            return
        half_k = k // 2
        # Recursively merge even-indexed and odd-indexed subsequences
        for i in range(offset, offset + k, 2):
            _oem(half_k, i)          # even subsequence
        for i in range(offset + 1, offset + k, 2):
            _oem(half_k, i)          # odd subsequence
        # Compare adjacent pairs
        for i in range(offset, offset + k - 1):
            result.append((i, i + 1))

    _oem(n, 0)
    return result


def _batcher_odd_even_mergesort(n):
    """Batcher's odd-even mergesort for n (power-of-2) wires."""
    if n <= 1:
        return []
    half = n // 2
    left = _batcher_odd_even_mergesort(half)
    right = _batcher_odd_even_mergesort(half)
    right = [(i + half, j + half) for i, j in right]
    merge = _batcher_odd_even_merge(n)
    return left + right + merge


def _build_initial_network():
    """Dense, redundant starting point: Batcher mergesort on next power of 2,
    restricted to N wires. This gives greedy removal a large pool of redundant
    comparators (the seed odd-even transposition is comparatively lean)."""
    p = 1
    while p < N:
        p *= 2
    network = _batcher_odd_even_mergesort(p)
    network = [(i, j) for i, j in network if i < N and j < N]

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for comp in network:
        if comp not in seen:
            seen.add(comp)
            unique.append(comp)
    return unique


def _dedup(net):
    seen = set()
    out = []
    for comp in net:
        if comp not in seen:
            seen.add(comp)
            out.append(comp)
    return out


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS

    current = _build_initial_network()
    if not _verify(current):
        current = _odd_even_transposition()

    best = current[:]
    best_len = len(best)

    # -----------------------------------------------------------------------
    # Phase 1: Greedy removal of redundant comparators (restart from top each hit)
    # -----------------------------------------------------------------------
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current) - 1, -1, -1):
            if time.time() > deadline:
                break
            neighbor = current[:i] + current[i + 1:]
            if _verify(neighbor):
                current = neighbor
                changed = True
                if len(current) < best_len:
                    best = current[:]
                    best_len = len(current)
                break  # restart scan from the top after each removal

    # -----------------------------------------------------------------------
    # Phase 2: Simulated annealing with remove / add / swap / move / slide
    # -----------------------------------------------------------------------
    if time.time() < deadline - 1.0:
        T = 8.0
        T_min = 0.05
        cooling_rate = 0.995
        rng = random.Random(42)

        all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]

        steps_since_improvement = 0
        restart_threshold = 600

        while time.time() < deadline and T > T_min:
            r = rng.random()
            if r < 0.40:
                move_type = 'remove'
            elif r < 0.70:
                move_type = 'add'
            elif r < 0.80:
                move_type = 'swap'
            elif r < 0.90:
                move_type = 'move'
            else:
                move_type = 'slide'

            neighbor = current[:]
            valid_move = True

            if move_type == 'remove' and len(neighbor) > 0:
                idx = rng.randrange(len(neighbor))
                neighbor.pop(idx)
            elif move_type == 'add':
                comp = rng.choice(all_comparators)
                pos = rng.randrange(len(neighbor) + 1)
                neighbor.insert(pos, comp)
            elif move_type == 'swap' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i == j:
                    valid_move = False
                else:
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            elif move_type == 'move' and len(neighbor) >= 2:
                i = rng.randrange(len(neighbor))
                j = rng.randrange(len(neighbor))
                if i == j:
                    valid_move = False
                else:
                    comp = neighbor.pop(i)
                    if i < j:
                        j -= 1
                    neighbor.insert(j, comp)
            elif move_type == 'slide' and len(neighbor) >= 1:
                # Slide a comparator one position forward or backward
                i = rng.randrange(len(neighbor))
                if rng.random() < 0.5:
                    if i < len(neighbor) - 1:
                        neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
                    else:
                        valid_move = False
                else:
                    if i > 0:
                        neighbor[i], neighbor[i - 1] = neighbor[i - 1], neighbor[i]
                    else:
                        valid_move = False
            else:
                valid_move = False

            if not valid_move:
                continue

            if _verify(neighbor):
                delta = len(neighbor) - len(current)
                if delta <= 0 or rng.random() < math.exp(-delta / T):
                    current = neighbor
                    if len(current) < best_len:
                        best = current[:]
                        best_len = len(current)
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

            # Stall detection: reset to best and kick temperature
            if steps_since_improvement > restart_threshold:
                current = best[:]
                T = max(T, 3.0)
                steps_since_improvement = 0

            T *= cooling_rate

    # -----------------------------------------------------------------------
    # Final safety: ensure the returned network is valid
    # -----------------------------------------------------------------------
    if not _verify(best):
        return _odd_even_transposition()
    return best
