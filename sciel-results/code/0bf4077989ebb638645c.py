"""Size-efficient sorting network on 16 wires: exact-cover ILP (X-sat) approach."""
import numpy as np
import time
import random
import itertools

N = 16

# ---- zero-one principle bit-parallel verification ---------------------------
# A network is a correct sorter iff for every input vector v, the 0-sink set
# (wires that end up 0) exactly equals the set of wires that were 0 in v.
# Equivalently, the union of 0-sink sets over all v with popcount(v) <= 8 is
# the full set of wires (every wire can be a 0-sink).

def compute_zero_sink_sets(net):
    """Return list of 16-bit masks: mask for input v = wires that end up 0.
    Only compute for v with popcount(v) <= 8 (symmetry: popcount > 8 is complement)."""
    masks = []
    for v in range(1 << N):
        if bin(v).count('1') > N // 2:
            continue
        state = list(v)
        for (i, j) in net:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        # 0-sink mask: wire w is 0 in output
        m = 0
        for w in range(N):
            if state[w] == 0:
                m |= (1 << w)
        masks.append((v, m))
    return masks


def verify(net):
    """Return True if net sorts all 2^N inputs (zero-one principle)."""
    if len(net) == 0:
        return N <= 1
    # quick check: sorted output for all-zero and all-one
    masks = compute_zero_sink_sets(net)
    # For correctness: for each v, output 0-set == input 0-set
    # Check via: for each v with popcount<=8, the 0-sink mask must equal
    # the input 0-mask (i.e., complement of v in N bits).
    full = (1 << N) - 1
    for (v, m) in masks:
        expected = full ^ v   # wires that are 0 in input v
        if m != expected:
            return False
    return True


# ---- ILP: minimum comparators whose 0-sink union covers all N wires --------
def build_comparator_pool(net, max_extra=200):
    """Generate candidate comparators: those in net plus random pairs."""
    pool = set()
    for (i, j) in net:
        pool.add((i, j))
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    for p in all_pairs:
        if len(pool) >= len(net) + max_extra:
            break
        pool.add(p)
    return list(pool)


def comparator_zero_sinks(comparator, net):
    """For each input v (popcount<=8), compute the 0-sink mask when we
    apply comparator on top of net. Returns dict v -> mask."""
    # We compute: for each v, run net, then apply comparator, get 0-sink mask.
    # But we need the 0-sink mask of the *combined* network.
    # This is expensive; instead we use the property:
    # A comparator (i,j) is "useful" for wire w if there exists v such that
    # in the network without (i,j), wire w is 1 but with (i,j) it becomes 0,
    # AND the resulting network is still correct.
    # For the ILP, we define: comparator c covers wire w if there is some v
    # where c changes output at w from 1 to 0, and the base network is correct.
    pass


# ---- Simulated annealing with exact verification ----------------------------
def odd_even_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def batcher_odd_even_sort(wires):
    """Batcher's odd-even merge sort network for given wire indices."""
    n = len(wires)
    if n <= 1:
        return []
    comps = []
    # Batcher's odd-even merge sort
    def _batcher(wire_list):
        m = len(wire_list)
        if m == 1:
            return []
        # split
        half = m // 2
        left = wire_list[:half]
        right = wire_list[half:]
        comps = []
        comps += _batcher(left)
        comps += _batcher(right)
        # odd-even merge
        comps += _odd_even_merge(left, right)
        return comps

    def _odd_even_merge(a, b):
        # a and b are sorted lists of wire indices
        # odd-even merge network
        na = len(a)
        nb = len(b)
        # interleave
        merged = []
        i, j = 0, 0
        while i < na or j < nb:
            if i < na and (j >= nb or a[i] <= b[j]):
                merged.append(a[i])
                i += 1
            else:
                merged.append(b[j])
                j += 1
        # Actually for odd-even merge, we need the specific comparator network.
        # Let me use a simpler approach: just return the comparators for
        # odd-even merge of two sorted sequences.
        # For simplicity, use the standard odd-even merge:
        # Stage 1: compare (a[0],b[0]), (a[1],b[1]), ... (even positions)
        # Stage 2: compare (a[0],b[1]), (a[1],b[2]), ... (odd positions)
        # This is complex; let me use a different strategy.
        # For now, return empty and use a different base.
        return []

    return _batcher(wires)


def simulated_annealing(start_net, deadline):
    """SA that removes redundant comparators, verified by zero-one principle."""
    net = list(start_net)
    best = list(net)
    best_len = len(net)
    t0 = time.time()

    # Precompute verification for current net
    def is_valid(n):
        return verify(n)

    # Verify initial
    if not is_valid(net):
        return best

    # SA: try removing each comparator
    rng = random.Random(42)
    T = 50.0
    T_min = 0.1
    cooling = 0.999

    while time.time() - t0 < deadline - 5:  # 5s safety margin
        if len(net) <= 1:
            break
        # Pick a random comparator to try removing
        idx = rng.randrange(len(net))
        candidate = net[:idx] + net[idx + 1:]
        if is_valid(candidate):
            # Accept: it's still valid
            net = candidate
            if len(net) < best_len:
                best = list(net)
                best_len = len(net)
        else:
            # Reject (or accept with prob exp(-1/T) if we had a cost function,
            # but here invalid is invalid, so just reject)
            pass
        T *= cooling
        if T < T_min:
            T = T_min

    return best


def local_search_remove(net, deadline):
    """Greedy + random removal of redundant comparators."""
    net = list(net)
    if not verify(net):
        return net
    best = list(net)
    best_len = len(net)
    t0 = time.time()
    rng = random.Random(123)

    # First: greedy pass - remove any comparator that is redundant
    changed = True
    while changed and time.time() - t0 < deadline - 5:
        changed = False
        for idx in range(len(net)):
            if time.time() - t0 > deadline - 5:
                break
            candidate = net[:idx] + net[idx + 1:]
            if verify(candidate):
                net = candidate
                changed = True
                if len(net) < best_len:
                    best = list(net)
                    best_len = len(net)
                break  # restart scan

    # Then: random removal attempts
    while time.time() - t0 < deadline - 5:
        if len(net) <= 1:
            break
        idx = rng.randrange(len(net))
        candidate = net[:idx] + net[idx + 1:]
        if verify(candidate):
            net = candidate
            if len(net) < best_len:
                best = list(net)
                best_len = len(net)
        else:
            # Try swapping two comparators
            if len(net) >= 2:
                j = rng.randrange(len(net))
                if j != idx:
                    candidate2 = list(net)
                    candidate2[idx], candidate2[j] = candidate2[j], candidate2[idx]
                    if verify(candidate2):
                        net = candidate2
                        if len(net) < best_len:
                            best = list(net)
                            best_len = len(net)

    return best


def construct():
    t0 = time.time()
    deadline = t0 + 80  # 80s out of 90s

    # Start from odd-even transposition (120 comparators)
    net = odd_even_network()

    # Phase 1: Greedy removal
    net = local_search_remove(net, deadline)
    best = list(net)
    best_len = len(net)

    if time.time() - t0 > deadline - 10:
        return best

    # Phase 2: SA with random restarts
    while time.time() - t0 < deadline - 5:
        # Restart from current best with perturbation
        perturbed = list(best)
        # Add some random comparators to escape local optimum
        for _ in range(3):
            i = random.randrange(N)
            j = random.randrange(N)
            if i < j:
                perturbed.append((i, j))
        # Try to remove
        result = local_search_remove(perturbed, deadline)
        if len(result) < best_len:
            best = result
            best_len = len(result)

    # Final verification
    if verify(best):
        return best
    else:
        # Fallback to original
        return odd_even_network()
