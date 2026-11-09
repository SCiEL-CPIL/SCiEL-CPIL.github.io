"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # safety margin under 90s kill

def _build_seed_network():
    """Odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _verify_network_bitmask(comps):
    """Verify sorting network using zero-one principle with bitmask simulation.
    Returns True if network sorts all 2^N boolean inputs."""
    # For each boolean input (represented as integer mask), simulate the network
    # and check if output is sorted (all 1s before all 0s, i.e., low bits are 1s)
    
    # Precompute: for each comparator (i,j), the mask operations
    # If bit i is 0 and bit j is 1, swap them
    # min goes to i, max goes to j
    # So: if (mask has bit i=0, bit j=1) -> swap
    # i.e., if (mask & (1<<i)) == 0 and (mask & (1<<j)) != 0:
    #   mask = mask | (1<<i) & ~(1<<j)  -- wait, let me think more carefully
    
    # After compare-exchange on (i,j):
    #   wire_i gets min, wire_j gets max
    # If we think of 1 > 0:
    #   If wire_i=0, wire_j=1: after -> wire_i=1, wire_j=0 (swap)
    #   If wire_i=1, wire_j=0: after -> wire_i=0, wire_j=1 (swap)  
    #   Wait no: min goes to i, max goes to j
    #   If wire_i=1, wire_j=0: min=0->i, max=1->j: wire_i=0, wire_j=1 (swap)
    #   If wire_i=0, wire_j=1: min=0->i, max=1->j: wire_i=0, wire_j=1 (no change)
    #   If wire_i=0, wire_j=0: no change
    #   If wire_i=1, wire_j=1: no change
    
    # So swap happens iff wire_i=1 and wire_j=0
    # In bitmask: if (mask & (1<<i)) and not (mask & (1<<j)):
    #   mask ^= (1<<i) | (1<<j)
    
    # A sorted output for N wires means: some k low bits are 1, rest are 0
    # i.e., mask = (1<<k) - 1 for some k in 0..N
    # Equivalently: mask == 0 or mask == (1<<N)-1 or mask has form 0b00...011...1
    
    # Better: check if mask is of form (1<<k)-1 for some k
    # i.e., mask & (mask+1) == 0 (power of 2 minus 1, or 0)
    # Actually: sorted means all 1s are in the lower positions
    # mask & (mask+1) == 0 works for 0 and for (1<<k)-1
    
    # Precompute sorted masks
    sorted_masks = set()
    for k in range(N + 1):
        sorted_masks.add((1 << k) - 1)
    
    total = 1 << N
    # Process all inputs
    for inp in range(total):
        mask = inp
        for (i, j) in comps:
            bi = 1 << i
            bj = 1 << j
            if (mask & bi) and not (mask & bj):
                mask ^= bi | bj
        if mask not in sorted_masks:
            return False
    return True

def _verify_network_bitmask_fast(comps):
    """Faster verification using vectorized bitmask simulation with NumPy."""
    total = 1 << N
    # Process all 2^N inputs at once using vectorized operations
    masks = np.arange(total, dtype=np.uint32)
    
    for (i, j) in comps:
        bi = np.uint32(1 << i)
        bj = np.uint32(1 << j)
        # Swap condition: bit i is 1 AND bit j is 0
        cond = (masks & bi) & ~(masks & bj)
        # Apply swap where condition is true
        swap_mask = np.uint32((1 << i) | (1 << j))
        masks = np.where(cond, masks ^ swap_mask, masks)
    
    # Check if all masks are sorted: mask & (mask+1) == 0
    # For uint32, mask+1 wraps around at 2^32, but our masks are < 2^17 so fine
    result = (masks & (masks + 1)) == 0
    return bool(np.all(result))

def _depth_of_network(comps):
    """Compute the parallel depth (longest path) of the comparator network."""
    if not comps:
        return 0
    # depth[i] = earliest time wire i is free
    depth = [0] * N
    for (i, j) in comps:
        d = max(depth[i], depth[j]) + 1
        depth[i] = d
        depth[j] = d
    return max(depth)

def _try_remove(comps, idx, deadline):
    """Try removing comparator at index idx. Returns new comps if valid, else None."""
    if time.time() > deadline:
        return None
    new_comps = comps[:idx] + comps[idx+1:]
    if _verify_network_bitmask_fast(new_comps):
        return new_comps
    return None

def _try_swap(comps, i, j, deadline):
    """Try swapping comparators at positions i and j."""
    if time.time() > deadline:
        return None
    new_comps = list(comps)
    new_comps[i], new_comps[j] = new_comps[j], new_comps[i]
    if _verify_network_bitmask_fast(new_comps):
        return new_comps
    return None

def _try_insert(comps, pos, i, j, deadline):
    """Try inserting comparator (i,j) at position pos."""
    if time.time() > deadline:
        return None
    new_comps = comps[:pos] + [(i, j)] + comps[pos:]
    if _verify_network_bitmask_fast(new_comps):
        return new_comps
    return None

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with seed network
    comps = _build_seed_network()
    best_comps = list(comps)
    best_len = len(comps)
    
    # Verify seed
    if not _verify_network_bitmask_fast(best_comps):
        # Shouldn't happen, but return seed anyway
        return best_comps
    
    # Phase 1: Greedy removal - repeatedly try to remove comparators
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(comps)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            new_comps = _try_remove(comps, idx, deadline)
            if new_comps is not None:
                comps = new_comps
                best_comps = list(comps)
                best_len = len(comps)
                improved = True
                break  # Restart scan from beginning
    
    # Phase 2: Local search with swaps and insertions
    if time.time() < deadline - 5:
        # Try random swaps
        import random
        rng = random.Random(42)
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            n = len(comps)
            if n < 2:
                break
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            new_comps = _try_swap(comps, i, j, deadline)
            if new_comps is not None:
                comps = new_comps
                best_comps = list(comps)
                best_len = len(comps)
        
        # Try inserting useful comparators and then removing others
        # First, try to find redundant comparators after structural changes
        if time.time() < deadline - 5:
            # Try inserting a random comparator, then doing removal pass
            for attempt in range(50):
                if time.time() > deadline - 3:
                    break
                i = rng.randint(0, N - 1)
                j = rng.randint(i + 1, N)
                pos = rng.randint(0, len(comps) + 1)
                new_comps = _try_insert(comps, pos, i, j, deadline)
                if new_comps is not None:
                    # Now try to remove from this extended network
                    extended = list(new_comps)
                    # Try removing the newly added one or others
                    for rem_idx in range(len(extended) - 1, -1, -1):
                        if time.time() > deadline - 2:
                            break
                        rc = _try_remove(extended, rem_idx, deadline)
                        if rc is not None:
                            if len(rc) < len(comps):
                                comps = rc
                                best_comps = list(comps)
                                best_len = len(comps)
                            break
                    # Also try full removal pass on extended
                    if time.time() < deadline - 5:
                        ext_improved = True
                        while ext_improved and time.time() < deadline - 3:
                            ext_improved = False
                            for rem_idx in range(len(extended) - 1, -1, -1):
                                if time.time() > deadline - 2:
                                    break
                                rc = _try_remove(extended, rem_idx, deadline)
                                if rc is not None:
                                    extended = rc
                                    ext_improved = True
                                    break
                        if len(extended) < best_len:
                            best_comps = list(extended)
                            best_len = len(extended)
                            comps = list(extended)
    
    # Final: try one more removal pass
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for idx in range(len(comps) - 1, -1, -1):
                if time.time() > deadline - 1:
                    break
                new_comps = _try_remove(comps, idx, deadline)
                if new_comps is not None:
                    comps = new_comps
                    best_comps = list(comps)
                    best_len = len(comps)
                    improved = True
                    break
    
    return best_comps
# EVOLVE-BLOCK-END
