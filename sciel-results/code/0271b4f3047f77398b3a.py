import time
import random
from functools import lru_cache

N = 18
TOTAL_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START

# Precomputed constants for 0-1 principle guided search
_TARGETS = None
_INITIALS = None
# For each pair (i,j), precompute:
# - fix_mask[i][j]: bitmask of inputs where wire i has 1 and wire j has 0 (i.e., the comparator would fix this input)
# - swap_mask[i][j]: bitmask of inputs where the comparator changes the state (i has 1, j has 0)
_FIX_MASKS = None
_SWAP_MASKS = None

def _precompute_targets():
    """Precompute the target state for each wire for all 2^N inputs."""
    targets = [0] * N
    popcounts = [0] * (1 << N)
    for m in range(1, 1 << N):
        popcounts[m] = popcounts[m >> 1] + (m & 1)
        
    masks_by_popcount = [0] * (N + 1)
    for m in range(1 << N):
        k = popcounts[m]
        masks_by_popcount[k] |= (1 << m)
        
    for i in range(N):
        threshold = N - i
        target_mask = 0
        for k in range(threshold, N + 1):
            target_mask |= masks_by_popcount[k]
        targets[i] = target_mask
        
    return targets

def _precompute_initials():
    """Precompute initial states for all 2^N inputs."""
    initials = [0] * N
    for i in range(N):
        mask = 0
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        initials[i] = mask
    return initials

def _precompute_fix_masks():
    """Precompute fix and swap masks for each comparator pair."""
    global _FIX_MASKS, _SWAP_MASKS
    _FIX_MASKS = [[0] * N for _ in range(N)]
    _SWAP_MASKS = [[0] * N for _ in range(N)]
    
    if _INITIALS is None:
        _init_constants()
    
    for i in range(N):
        for j in range(i + 1, N):
            si = _INITIALS[i]
            sj = _INITIALS[j]
            # Comparator fixes input m if si has 1 and sj has 0 at bit m
            fix = si & ~sj
            # Comparator changes state if si != sj, i.e., si XOR sj
            swap = si ^ sj
            _FIX_MASKS[i][j] = fix
            _SWAP_MASKS[i][j] = swap

def _init_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is None:
        _TARGETS = _precompute_targets()
        _INITIALS = _precompute_initials()
        _precompute_fix_masks()

def _verify_network_bitparallel(comparators):
    """Verifies the network using bit-parallel simulation."""
    if _TARGETS is None:
        _init_constants()
    
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _compute_unsorted_mask(states):
    """Compute which inputs are still unsorted given current wire states."""
    unsorted = 0
    for i in range(N):
        if states[i] != _TARGETS[i]:
            unsorted |= (states[i] ^ _TARGETS[i])
    return unsorted

def _guided_constructive_search(deadline):
    """
    Constructive search guided by the 0-1 principle.
    Start from an empty network and greedily add comparators that fix the most unsorted inputs.
    """
    if _TARGETS is None:
        _init_constants()
    
    states = list(_INITIALS)
    network = []
    
    while time.time() < deadline:
        # Find unsorted inputs
        unsorted = 0
        for i in range(N):
            if states[i] != _TARGETS[i]:
                unsorted |= (states[i] ^ _TARGETS[i])
        
        if unsorted == 0:
            break  # Sorted!
        
        # Find the comparator that fixes the most unsorted inputs
        best_fix = 0
        best_comp = None
        for i in range(N):
            for j in range(i + 1, N):
                si = states[i]
                sj = states[j]
                # Only consider if this comparator would change something
                if si == sj:
                    continue
                # Count how many unsorted inputs this fixes
                fix = si & ~sj & unsorted
                if fix > best_fix:
                    best_fix = fix
                    best_comp = (i, j)
        
        if best_comp is None or best_fix == 0:
            # No single comparator can fix any unsorted input - shouldn't happen for a valid network
            # But if we're stuck, break
            break
        
        i, j = best_comp
        network.append((i, j))
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
    
    return network

def _local_search_prune(comps, deadline):
    """
    Aggressive local search: try to remove each comparator one by one.
    Use bit-parallel verification for speed.
    """
    comps = list(comps)
    n = len(comps)
    
    while time.time() < deadline:
        removed_any = False
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline:
                return comps
            
            # Try removing comparator at idx
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                n = len(comps)
                removed_any = True
                break  # Restart from end after removal
        
        if not removed_any:
            break
    
    return comps

def _swap_mutation(comps, idx1, idx2):
    """Swap two comparators in the network."""
    c = list(comps)
    c[idx1], c[idx2] = c[idx2], c[idx1]
    return c

def _local_search_with_swaps(comps, deadline, max_iters=200):
    """
    Local search that tries removals and swaps to escape local minima.
    """
    best = list(comps)
    best_size = len(best)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Try removals
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_bitparallel(candidate):
                best = candidate
                best_size -= 1
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try swaps (to escape local minima)
        # Try a limited number of random swaps
        for _ in range(20):
            if time.time() > deadline:
                break
            if len(best) < 2:
                break
            i = random.randint(0, len(best) - 2)
            j = random.randint(i + 1, len(best) - 1)
            swapped = _swap_mutation(best, i, j)
            
            # Check if swap helps: verify and try to prune
            if _verify_network_bitparallel(swapped):
                pruned = _local_search_prune(swapped, deadline)
                if len(pruned) < best_size:
                    best = pruned
                    best_size = len(best)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best

def construct():
    _init_constants()
    
    # 90s limit, keep safety margin
    deadline = time.time() + 80.0
    
    # Strategy: 
    # 1. Start with seed network
    # 2. Aggressive pruning
    # 3. Guided constructive search to find alternative networks
    # 4. Local search with swaps to escape local minima
    
    best = _seed_network()
    best_size = len(best)
    
    # Step 1: Prune the seed
    best = _local_search_prune(best, deadline)
    best_size = len(best)
    
    # Step 2: Try guided constructive search
    if time.time() < deadline:
        constructive = _guided_constructive_search(deadline)
        if constructive:
            pruned_constructive = _local_search_prune(constructive, deadline)
            if len(pruned_constructive) < best_size:
                best = pruned_constructive
                best_size = len(best)
    
    # Step 3: Local search with swaps on best
    if time.time() < deadline:
        best = _local_search_with_swaps(best, deadline)
        best_size = len(best)
    
    # Step 4: Final aggressive prune
    if time.time() < deadline:
        best = _local_search_prune(best, deadline)
    
    return best

# EVOLVE-BLOCK-END
