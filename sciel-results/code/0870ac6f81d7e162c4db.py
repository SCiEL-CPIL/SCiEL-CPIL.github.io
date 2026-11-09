"""Size-efficient sorting networks on 17 wires."""
import time
import random

N = 17

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

N = 17

# Precompute the initial mask vectors for fast verification
# Each wire w has a bitmask where bit k is 1 iff input k has bit w set
# We'll represent masks as integers for speed

def _precompute_masks(n_wires):
    """Precompute the initial masks for all wires."""
    total_inputs = 1 << n_wires
    masks = []
    for w in range(n_wires):
        m = 0
        for k in range(total_inputs):
            if (k >> w) & 1:
                m |= (1 << k)
        masks.append(m)
    return masks

# Cache the precomputed masks
_MASKS_CACHE = {}

def _get_masks(n_wires):
    if n_wires not in _MASKS_CACHE:
        _MASKS_CACHE[n_wires] = _precompute_masks(n_wires)
    return _MASKS_CACHE[n_wires]

def _verify_network_fast(network, n_wires):
    """Fast verification using precomputed masks."""
    masks = list(_get_masks(n_wires))
    for i, j in network:
        if i >= j or i < 0 or j >= n_wires:
            return False
        mi = masks[i]
        mj = masks[j]
        masks[i] = mi & mj
        masks[j] = mi | mj
    # Check if sorted: for each adjacent pair, masks[w] & ~masks[w+1] should be 0
    for w in range(n_wires - 1):
        if masks[w] & ~masks[w + 1]:
            return False
    return True

def _build_odd_even_merge_sort(n_wires):
    """Build an odd-even merge sort network."""
    def _merge_sort(n, network):
        if n <= 1:
            return network
        if n == 2:
            network.append((0, 1))
            return network
        mid = n // 2
        # Sort first half and second half
        network = _merge_sort(mid, network)
        # Sort second half (shifted by mid)
        for i in range(mid, n, 2):
            if i + 1 < n:
                network.append((i, i + 1))
        for i in range(mid + 1, n, 2):
            if i + 1 < n:
                network.append((i, i + 1))
        # Actually, let's use a simpler approach: odd-even transposition
        # For now, just use odd-even transposition as seed
        return network
    
    # Use odd-even transposition as a reliable seed
    network = []
    for phase in range(n_wires):
        if phase % 2 == 0:
            for i in range(0, n_wires - 1, 2):
                network.append((i, i + 1))
        else:
            for i in range(1, n_wires - 1, 2):
                network.append((i, i + 1))
    return network

def _greedy_remove(network, deadline):
    """Aggressive greedy removal trying multiple positions."""
    removed = 0
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removals in a strategic order: from end to start
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network, removed
            # Quick check: if this comparator is a no-op, skip verification
            # Remove and verify
            new_net = network[:i] + network[i+1:]
            if _verify_network_fast(new_net, N):
                network = new_net
                removed += 1
                improved = True
                break
    return network, removed

def _try_swap(network, idx1, idx2, deadline):
    """Try swapping two comparators and cleaning up."""
    if idx1 == idx2:
        return None
    new_net = network[:]
    new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
    if not _verify_network_fast(new_net, N):
        return None
    cleaned, _ = _greedy_remove(new_net, deadline)
    return cleaned

def _try_relocate(network, idx, new_pos, deadline):
    """Try relocating a comparator to a new position."""
    if idx == new_pos:
        return None
    comp = network[idx]
    new_net = network[:idx] + network[idx+1:]
    if new_pos > idx:
        new_pos -= 1
    new_net.insert(new_pos, comp)
    if not _verify_network_fast(new_net, N):
        return None
    cleaned, _ = _greedy_remove(new_net, deadline)
    return cleaned

def _try_modify(network, idx, new_i, new_j, deadline):
    """Try modifying a comparator to connect different wires."""
    if new_i >= new_j or new_i < 0 or new_j >= N:
        return None
    new_net = network[:]
    new_net[idx] = (new_i, new_j)
    if not _verify_network_fast(new_net, N):
        return None
    cleaned, _ = _greedy_remove(new_net, deadline)
    return cleaned

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 90s limit
    
    # Start with odd-even transposition
    network = _build_odd_even_merge_sort(N)
    best_network = list(network)
    best_size = len(network)
    
    # Initial aggressive cleanup
    network, _ = _greedy_remove(network, deadline)
    if len(network) < best_size:
        best_size = len(network)
        best_network = list(network)
    
    # Local search with multiple strategies
    rng = random.Random(42)
    while time.time() < deadline - 3:
        strategy = rng.randint(0, 3)
        candidate = None
        
        if strategy == 0 and len(best_network) > 1:
            # Try swapping two random comparators
            i1 = rng.randint(0, len(best_network) - 1)
            i2 = rng.randint(0, len(best_network) - 1)
            if i1 != i2:
                candidate = _try_swap(best_network, i1, i2, deadline)
        
        elif strategy == 1 and len(best_network) > 0:
            # Try relocating a comparator
            idx = rng.randint(0, len(best_network) - 1)
            new_pos = rng.randint(0, len(best_network) - 1)
            candidate = _try_relocate(best_network, idx, new_pos, deadline)
        
        elif strategy == 2 and len(best_network) > 0:
            # Try modifying a comparator
            idx = rng.randint(0, len(best_network) - 1)
            old_i, old_j = best_network[idx]
            # Try nearby wires
            new_i = max(0, min(N - 2, old_i + rng.choice([-1, 0, 1])))
            new_j = max(1, min(N - 1, old_j + rng.choice([-1, 0, 1])))
            if new_i < new_j:
                candidate = _try_modify(best_network, idx, new_i, new_j, deadline)
        
        elif strategy == 3 and len(best_network) > 0:
            # Try removing a random comparator (if it's redundant)
            idx = rng.randint(0, len(best_network) - 1)
            new_net = best_network[:idx] + best_network[idx+1:]
            if _verify_network_fast(new_net, N):
                candidate, _ = _greedy_remove(new_net, deadline)
        
        if candidate is not None and len(candidate) < best_size:
            best_size = len(candidate)
            best_network = candidate
    
    return best_network

# EVOLVE-BLOCK-END
