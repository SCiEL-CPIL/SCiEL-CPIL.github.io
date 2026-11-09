import time

N = 17

def _build_seed_network():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _init_wires():
    """Initialize wire bitmasks for all Boolean inputs."""
    wires = []
    total = 1 << N
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
    """Compute expected final wire values after sorting."""
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
    """Verify network sorts all Boolean inputs using bitmask simulation."""
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

def _depth_of_network(comparators):
    """Compute parallel depth of the network."""
    depth = [0] * N
    for (a, b) in comparators:
        d = max(depth[a], depth[b]) + 1
        depth[a] = d
        depth[b] = d
    return max(depth)

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 5 second safety margin
    
    # Build seed network
    comps = _build_seed_network()
    
    # Precompute
    wires_init = _init_wires()
    expected = _compute_expected()
    
    # Verify seed is correct
    assert _verify_network_bitmask(comps, wires_init, expected)
    
    best_comps = comps[:]
    best_count = len(comps)
    
    # Greedy removal: try removing each comparator
    changed = True
    while changed and time.time() < DEADLINE:
        changed = False
        i = 0
        while i < len(comps) and time.time() < DEADLINE:
            # Try removing comparator i
            new_comps = comps[:i] + comps[i+1:]
            if _verify_network_bitmask(new_comps, wires_init, expected):
                comps = new_comps
                changed = True
                # Don't increment i, try removing next one at same position
            else:
                i += 1
        # If we made changes, we might be able to remove more
        if len(comps) < best_count:
            best_count = len(comps)
            best_comps = comps[:]
    
    # Try more aggressive local search if time permits
    if time.time() < DEADLINE:
        # Try swapping adjacent comparators
        for _round in range(50):
            if time.time() > DEADLINE:
                break
            improved = False
            for i in range(len(comps) - 1):
                if time.time() > DEADLINE:
                    break
                # Try swapping comps[i] and comps[i+1]
                new_comps = comps[:]
                new_comps[i], new_comps[i+1] = comps[i+1], comps[i]
                if _verify_network_bitmask(new_comps, wires_init, expected):
                    # Check if this actually helps (it's same count, so skip for now)
                    pass
            if not improved:
                break
    
    return best_comps
