import time
import numpy as np

N = 17
NUM_BOOLEANS = 1 << N  # 131072

# Precompute bit masks
_BIT_MASKS = [1 << i for i in range(N)]

def _check_full(network):
    """Full verification: return True iff network sorts all 2^N Boolean inputs."""
    if not network:
        return N == 1
    bits = np.arange(1, NUM_BOOLEANS, dtype=np.uint32)[:, None]
    for (i, j) in network:
        bi = bits & _BIT_MASKS[i]
        bj = bits & _BIT_MASKS[j]
        bits = np.where(bi < bj, bits, bits | bi | bj)
    for i in range(N - 1):
        bi = bits & _BIT_MASKS[i]
        bj = bits & _BIT_MASKS[i + 1]
        if np.any((bi != 0) & (bj == 0)):
            return False
    return True


def _check_single(network, x):
    """Check if a single Boolean input x is sorted by the network. Returns True if sorted."""
    val = x
    for (i, j) in network:
        bi = val & _BIT_MASKS[i]
        bj = val & _BIT_MASKS[j]
        if bi > bj:
            val = val | bi | bj
    # Check sorted: for adjacent pairs, no (1,0) pattern
    for i in range(N - 1):
        if (val & _BIT_MASKS[i]) and not (val & _BIT_MASKS[i + 1]):
            return False
    return True


def _find_bad(network):
    """Find a single Boolean input that the network fails to sort. Returns (x, True) or (None, False)."""
    # Check all inputs; return first failure
    for x in range(1, NUM_BOOLEANS):
        if not _check_single(network, x):
            return (x, True)
    return (None, False)


def _check_fast(network, bad_x=None):
    """
    Fast check: if bad_x is provided and network sorts it, do a full check.
    If bad_x is None, do full check.
    Returns True if network is valid.
    """
    if bad_x is not None:
        if not _check_single(network, bad_x):
            return False
    # Full verification
    return _check_full(network)


# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill

    # Seed: odd-even transposition network (known correct, 136 comparators for N=17)
    def odd_even():
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

    net = odd_even()
    best_net = list(net)
    best_count = len(net)

    if not _check_full(net):
        return best_net

    # Track the "worst" input that was last failing for the current net
    # When we remove a comparator, we can first check if the previously-failing input is now sorted
    bad_x = None

    # Local search: iteratively try to remove comparators
    # Strategy: 
    # 1. Try removing each comparator in a shuffled order
    # 2. For each candidate, first do a quick check on the known-bad input (if any)
    # 3. If that passes, do full verification
    # 4. If removal fails, remember the bad input for next attempt
    
    idx = 0
    order = None
    consecutive_fails = 0
    max_consecutive_fails = len(net) * 3  # after this many fails in a row, re-shuffle

    while time.time() < deadline:
        if order is None or idx >= len(order):
            order = list(range(len(net)))
            np.random.shuffle(order)
            idx = 0
            consecutive_fails = 0

        pos = order[idx]
        idx += 1
        consecutive_fails += 1

        candidate = net[:pos] + net[pos + 1:]
        
        # Quick check: if we have a known bad input, check it first
        if bad_x is not None:
            if not _check_single(candidate, bad_x):
                # Still fails on the same bad input; try next
                continue
        
        # Full check
        if _check_full(candidate):
            net = candidate
            best_net = list(net)
            best_count = len(net)
            bad_x = None  # Reset; we don't know a bad input for the new net
            order = None
            idx = 0
            consecutive_fails = 0
        else:
            # Find a bad input for the current net (not candidate) to use for quick rejection
            # Actually, find bad input for candidate to help reject similar candidates
            bx, _ = _find_bad(candidate)
            if bx is not None:
                bad_x = bx

    # Final verification of best_net
    if not _check_full(best_net):
        # Fallback: return seed
        return odd_even()

    return best_net
# EVOLVE-BLOCK-END
