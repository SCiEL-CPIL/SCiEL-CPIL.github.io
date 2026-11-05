"""Size-efficient sorting network search on 17 wires."""
import time
import random
import numpy as np

N = 17
M = 1 << N  # 131072

# Precompute the all-inputs matrix once: shape (N, M), dtype uint8.
# wires[w][k] = bit w of input k.
_inputs = np.arange(M, dtype=np.uint32)
_INPUTS = np.empty((N, M), dtype=np.uint8)
for _w in range(N):
    _INPUTS[_w] = ((_inputs >> _w) & 1).astype(np.uint8)


def _verify_fast(network):
    """Verify sorting network over all 2^17 boolean inputs.

    Uses the zero-one principle. Fast path: copy the precomputed input
    matrix, apply each comparator as AND/OR (min/max on booleans), then
    check that the result is non-decreasing on every wire column.
    """
    wires = _INPUTS.copy()
    for (i, j) in network:
        wi = wires[i]
        wj = wires[j]
        # min -> AND, max -> OR (values are 0/1)
        wires[i] = wi & wj
        wires[j] = wi | wj
    # Non-decreasing across wires: no column where wire k has 1 and wire k+1 has 0.
    # diff[w] = wires[w+1] - wires[w]; sorted iff all diff >= 0.
    diff = np.subtract(wires[1:], wires[:-1])
    if np.any(diff < 0):
        return False
    # Conservation is implied by AND/OR on booleans, but be safe:
    # wire 0 must be all 0, wire N-1 all 1 for a fully sorted 0/1 sequence
    # only when the input has all 0s then all 1s -- NOT true for all inputs.
    # So we do NOT check those; the adjacent check is the complete condition.
    return True


def _depth_of_network(network):
    depth = [0] * N
    for (i, j) in network:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth) if depth else 0


def _seed_odd_even():
    """Odd-even transposition sorting network for N wires (seed, 136 comparators)."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin before the 90s kill

    def _time_left():
        return deadline - time.time()

    best_network = _seed_odd_even()
    best_count = len(best_network)

    # Sanity: the seed must be valid.
    if not _verify_fast(best_network):
        # Should never happen; fall back to a trivially correct (large) network.
        # Odd-even is correct, so this is just a guard.
        pass

    rng = random.Random(42)

    # ------------------------------------------------------------------
    # Phase 1: Greedy removal of redundant comparators.
    # Try removing from the END first (later comparators are more likely
    # redundant). Restart the scan after each successful removal.
    # ------------------------------------------------------------------
    changed = True
    while changed and _time_left() > 3:
        changed = False
        # Scan from the end toward the front.
        for idx in range(len(best_network) - 1, -1, -1):
            if _time_left() < 3:
                break
            trial = best_network[:idx] + best_network[idx + 1:]
            if _verify_fast(trial):
                best_network = trial
                best_count = len(trial)
                changed = True
                break  # restart scan; indices shifted

    # ------------------------------------------------------------------
    # Phase 2: If removal stalled, try local mutations to escape local
    # minima (swap adjacent, or replace a comparator with a random one).
    # ------------------------------------------------------------------
    if _time_left() > 10:
        improved = True
        while improved and _time_left() > 5:
            improved = False
            # Try swapping adjacent comparators.
            for idx in range(len(best_network) - 1):
                if _time_left() < 5:
                    break
                trial = list(best_network)
                trial[idx], trial[idx + 1] = trial[idx + 1], trial[idx]
                if _verify_fast(trial):
                    best_network = trial
                    improved = True
                    break
            # Try replacing a comparator with a random valid pair.
            if not improved and _time_left() > 5:
                indices = list(range(len(best_network)))
                rng.shuffle(indices)
                for idx in indices:
                    if _time_left() < 5:
                        break
                    for _ in range(4):
                        if _time_left() < 5:
                            break
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        trial = list(best_network)
                        trial[idx] = (i, j)
                        if _verify_fast(trial):
                            best_network = trial
                            improved = True
                            break

    # ------------------------------------------------------------------
    # Phase 3: Final removal pass (may find more after mutations).
    # ------------------------------------------------------------------
    if _time_left() > 3:
        changed = True
        while changed and _time_left() > 2:
            changed = False
            for idx in range(len(best_network) - 1, -1, -1):
                if _time_left() < 2:
                    break
                trial = best_network[:idx] + best_network[idx + 1:]
                if _verify_fast(trial):
                    best_network = trial
                    best_count = len(trial)
                    changed = True
                    break

    return best_network
