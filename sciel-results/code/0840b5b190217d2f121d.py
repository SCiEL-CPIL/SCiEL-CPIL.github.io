"""Size-efficient 16-wire sorting network via bit-parallel local search."""
import time

N = 16
NUM_INPUTS = 1 << N
DEADLINE_MARGIN = 6.0  # seconds of safety before the 90s wall-clock kill


def _odd_even_network():
    """Odd-even transposition network: 120 comparators for 16 wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _build_target_masks():
    target = [0] * N
    for x in range(NUM_INPUTS):
        target[x.bit_count()] |= (1 << x)
    return target


def _simulate(net, target):
    """Return True iff net sorts all 65536 inputs (bit-parallel)."""
    masks = [0] * N
    for x in range(NUM_INPUTS):
        for w in range(N):
            if (x >> w) & 1:
                masks[w] |= (1 << x)
    for (i, j) in net:
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    for k in range(N):
        if masks[k] != target[k]:
            return False
    return True


def construct():
    start = time.time()
    deadline = start + (90.0 - DEADLINE_MARGIN)

    target = _build_target_masks()

    net = _odd_even_network()
    assert _simulate(net, target), "baseline must be correct"

    best = list(net)
    best_len = len(best)

    # Greedy removal local search with explicit deadline guard.
    # Each candidate deletion is verified by a full bit-parallel simulation.
    while time.time() < deadline:
        if not _simulate(net, target):
            net = list(best)

        progress = False
        i = 0
        while i < len(net) and time.time() < deadline:
            candidate = net[:i] + net[i + 1:]
            if _simulate(candidate, target):
                net = candidate
                progress = True
                # stay at same index; next element shifted in
            else:
                i += 1

        if len(net) < best_len:
            best = list(net)
            best_len = len(net)

        if not progress:
            break

    return best
