"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # safety margin under 90s kill


def _build_seed_network():
    """Odd-even transposition network for N wires."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _init_wires():
    """Initialize wire bitmasks for all Boolean inputs.
    
    For N wires, there are 2^N boolean inputs.
    wires[i] is a bitmask of length 2^N where bit k is 1 iff input k has a 1 on wire i.
    """
    total = 1 << N
    wires = []
    for i in range(N):
        val = 0
        block = 1 << i
        period = 2 * block
        for start in range(0, total, period):
            mask = ((1 << block) - 1) << (start + block)
            val |= mask
        wires.append(val)
    return wires


def _compute_expected():
    """Compute expected final wire values after sorting.
    
    After sorting, wire i (0-indexed, 0 is smallest) should contain 1s for all inputs
    where the (N-i)-th smallest value is 1.
    Since we're dealing with boolean inputs, the sorted output on wire i is 1 iff
    the input has at least (N-i) ones.
    """
    expected = []
    total = 1 << N
    for i in range(N):
        val = 0
        threshold = N - i
        for k in range(threshold, total):
            if k.bit_count() >= threshold:
                val |= (1 << k)
        expected.append(val)
    return expected


def _verify_network_bitmask(comparators, wires_init=None, expected=None):
    """Verify network sorts all Boolean inputs using bitmask simulation.
    
    This is the fastest verification method using Python big integers.
    Each wire is represented as a single big integer where bit k indicates
    the value on that wire for input k.
    """
    if wires_init is None:
        wires_init = _init_wires()
    if expected is None:
        expected = _compute_expected()
    
    wires = wires_init[:]
    for (a, b) in comparators:
        ma = wires[a] & wires[b]
        mb = wires[a] | wires[b]
        wires[a] = ma
        wires[b] = mb
    
    for i in range(N):
        if wires[i] != expected[i]:
            return False
    return True


def _verify_network_numpy(comps):
    """Vectorized zero-one verification over all 2^N boolean inputs using NumPy.
    
    Returns True iff the comparator network sorts every boolean input.
    """
    total = 1 << N
    masks = np.arange(total, dtype=np.uint32)
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        # swap iff bit i is 1 AND bit j is 0
        cond = (masks & bi) & ~(masks & bj)
        if np.any(cond):
            swap_mask = np.uint32((1 << i) | (1 << j))
            masks = np.where(cond, masks ^ swap_mask, masks)
    # sorted iff mask & (mask+1) == 0  (i.e. mask is of form (1<<k)-1)
    return bool(np.all((masks & (masks + 1)) == 0))


def _depth_of_network(comps):
    """Compute parallel depth (longest path) of the network."""
    if not comps:
        return 0
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)


def _greedy_remove(comps, wires_init, expected, deadline):
    """Repeatedly try to remove each comparator until a full pass removes none.
    Returns the reduced list."""
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            new_comps = comps[:idx] + comps[idx + 1:]
            if _verify_network_bitmask(new_comps, wires_init, expected):
                comps = new_comps
                improved = True
                break  # restart scan from the end
    return comps


def _try_remove(comps, idx, wires_init, expected, deadline):
    """Try removing comparator at index idx. Returns new comps if valid, else None."""
    if time.time() > deadline:
        return None
    new_comps = comps[:idx] + comps[idx+1:]
    if _verify_network_bitmask(new_comps, wires_init, expected):
        return new_comps
    return None


def _try_swap(comps, i, j, wires_init, expected, deadline):
    """Try swapping comparators at positions i and j."""
    if time.time() > deadline:
        return None
    new_comps = list(comps)
    new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
    if _verify_network_bitmask(new_comps, wires_init, expected):
        return new_comps
    return None


def _try_insert(comps, pos, i, j, wires_init, expected, deadline):
    """Try inserting comparator (i,j) at position pos."""
    if time.time() > deadline:
        return None
    new_comps = comps[:pos] + [(i, j)] + comps[pos:]
    if _verify_network_bitmask(new_comps, wires_init, expected):
        return new_comps
    return None


def _local_search(comps, wires_init, expected, rng, deadline):
    """Local search with swaps, insertions, and re-removal to escape minima."""
    comps = list(comps)
    n0 = len(comps)
    while time.time() < deadline:
        moved = False
        # Try a few random swaps (reorder) and re-remove
        for _ in range(3):
            if time.time() > deadline:
                break
            n = len(comps)
            if n < 2:
                break
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            cand = comps[:]
            cand[i], cand[j] = cand[j], cand[i]
            if _verify_network_bitmask(cand, wires_init, expected):
                reduced = _greedy_remove(cand, wires_init, expected, deadline)
                if len(reduced) < len(comps):
                    comps = reduced
                    moved = True
                    break
                elif len(reduced) == len(comps) and rng.random() < 0.3:
                    # accept a same-size reorder to change the landscape
                    comps = reduced
                    moved = True
                    break
        # Try inserting a comparator then re-removing (escape move)
        if not moved and time.time() < deadline:
            for _ in range(2):
                if time.time() > deadline:
                    break
                a = rng.randint(0, N - 1)
                b = rng.randint(a + 1, N)
                pos = rng.randint(0, len(comps) + 1)
                cand = comps[:pos] + [(a, b)] + comps[pos:]
                if _verify_network_bitmask(cand, wires_init, expected):
                    reduced = _greedy_remove(cand, wires_init, expected, deadline)
                    if len(reduced) < len(comps):
                        comps = reduced
                        moved = True
                        break
                    elif len(reduced) == len(comps) and rng.random() < 0.2:
                        comps = reduced
                        moved = True
                        break
        if not moved:
            # No improvement in this iteration, try a few more random perturbations
            perturbations = 0
            while not moved and perturbations < 5 and time.time() < deadline:
                perturbations += 1
                n = len(comps)
                if n < 2:
                    break
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
                if i == j:
                    continue
                cand = comps[:]
                cand[i], cand[j] = cand[j], cand[i]
                if _verify_network_bitmask(cand, wires_init, expected):
                    reduced = _greedy_remove(cand, wires_init, expected, deadline)
                    if len(reduced) < len(comps):
                        comps = reduced
                        moved = True
                    elif rng.random() < 0.1:
                        comps = reduced
                        moved = True
            if not moved:
                break
    
    return comps


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Build seed network
    comps = _build_seed_network()
    
    # Precompute verification data
    wires_init = _init_wires()
    expected = _compute_expected()
    
    # Verify seed is correct
    if not _verify_network_bitmask(comps, wires_init, expected):
        return comps
    
    best_comps = comps[:]
    best_count = len(comps)
    
    # Phase 1: Greedy removal
    comps = _greedy_remove(comps, wires_init, expected, deadline - 20)
    if len(comps) < best_count:
        best_count = len(comps)
        best_comps = comps[:]
    
    # Phase 2: Local search
    if time.time() < deadline - 10:
        rng = random.Random(42)
        comps = _local_search(comps, wires_init, expected, rng, deadline - 5)
        if len(comps) < best_count:
            best_count = len(comps)
            best_comps = comps[:]
    
    # Phase 3: Final greedy removal pass
    if time.time() < deadline - 2:
        comps = _greedy_remove(comps, wires_init, expected, deadline - 1)
        if len(comps) < best_count:
            best_count = len(comps)
            best_comps = comps[:]
    
    # Verify final result
    if not _verify_network_bitmask(best_comps, wires_init, expected):
        # Shouldn't happen, but return seed anyway
        return _build_seed_network()
    
    return best_comps
# EVOLVE-BLOCK-END
