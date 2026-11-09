import time
import random

N = 18

# EVOLVE-BLOCK-START

# We use Python big integers to represent the state of all 2^N inputs.
# Bit 'm' in the integer corresponds to input mask 'm'.
# Wire i state is an integer M_i.

def _precompute_targets():
    """
    Precompute the target state for each wire for all 2^N inputs.
    """
    targets = [0] * N
    # Precompute popcount for all masks
    popcounts = [0] * (1 << N)
    for m in range(1, 1 << N):
        popcounts[m] = popcounts[m >> 1] + (m & 1)
        
    # Initialize masks for each possible popcount k
    masks_by_popcount = [0] * (N + 1)
    for m in range(1 << N):
        k = popcounts[m]
        masks_by_popcount[k] |= (1 << m)
        
    # Construct targets
    for i in range(N):
        threshold = N - i
        target_mask = 0
        for k in range(threshold, N + 1):
            target_mask |= masks_by_popcount[k]
        targets[i] = target_mask
        
    return targets

def _precompute_initials():
    """
    Precompute initial states for all 2^N inputs.
    """
    initials = [0] * N
    for i in range(N):
        mask = 0
        for m in range(1 << N):
            if (m >> i) & 1:
                mask |= (1 << m)
        initials[i] = mask
    return initials

_TARGETS = None
_INITIALS = None

def _init_constants():
    global _TARGETS, _INITIALS
    if _TARGETS is None:
        _TARGETS = _precompute_targets()
        _INITIALS = _precompute_initials()

def _verify_network_bitparallel(comparators):
    """
    Verifies the network using bit-parallel simulation.
    """
    if _TARGETS is None:
        _init_constants()
    
    # Copy initials
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
        
    # Check against targets
    for i in range(N):
        if states[i] != _TARGETS[i]:
            return False
    return True

def _hamming_distance_bitparallel(comparators):
    """
    Compute the total Hamming distance between the network output and targets
    across all 2^N inputs. Returns an integer (number of bit positions where
    output differs from target).
    """
    if _TARGETS is None:
        _init_constants()
    
    states = list(_INITIALS)
    
    for i, j in comparators:
        si = states[i]
        sj = states[j]
        states[i] = si & sj
        states[j] = si | sj
    
    total_hd = 0
    for i in range(N):
        diff = states[i] ^ _TARGETS[i]
        total_hd += diff.bit_count()
    return total_hd

def _seed_network():
    """Odd-Even Transposition Sort."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _prune(comps, deadline):
    """
    Aggressively remove redundant comparators.
    We try to remove comparators from the end first, as they are often redundant.
    If a removal fails, we move to the previous one.
    """
    comps = list(comps)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from the end first
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() > deadline:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network_bitparallel(candidate):
                comps = candidate
                improved = True
                break
    return comps

def _greedy_construct(deadline):
    """
    Greedily build a sorting network by starting from an empty network and
    adding the comparator that most reduces total Hamming distance to targets.
    Uses the 0-1 principle: we only need to check Boolean inputs.
    """
    if _TARGETS is None:
        _init_constants()
    
    comps = []
    current_hd = _hamming_distance_bitparallel(comps)
    
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        if current_hd == 0:
            break
        
        best_comp = None
        best_hd = current_hd
        
        for comp in all_comps:
            candidate = comps + [comp]
            hd = _hamming_distance_bitparallel(candidate)
            if hd < best_hd:
                best_hd = hd
                best_comp = comp
            if time.time() > deadline:
                break
        
        if best_comp is None or best_hd >= current_hd:
            # No single comparator reduces HD; try random insertion
            if time.time() > deadline:
                break
            comp = random.choice(all_comps)
            pos = random.randint(0, len(comps))
            candidate = comps[:pos] + [comp] + comps[pos:]
            hd = _hamming_distance_bitparallel(candidate)
            if hd < current_hd:
                comps = candidate
                current_hd = hd
            else:
                break
        else:
            # Insert the best comparator at a position that minimizes HD
            # Try inserting at several positions
            best_pos = len(comps)
            best_pos_hd = best_hd
            for pos in range(len(comps) + 1):
                candidate = comps[:pos] + [best_comp] + comps[pos:]
                hd = _hamming_distance_bitparallel(candidate)
                if hd < best_pos_hd:
                    best_pos_hd = hd
                    best_pos = pos
                if time.time() > deadline:
                    break
            comps = comps[:best_pos] + [best_comp] + comps[best_pos:]
            current_hd = best_pos_hd
    
    return comps

def construct():
    _init_constants()
    
    # 90s limit, keep 5s safety margin
    deadline = time.time() + 85.0
    rng = random.Random(42)
    
    # Phase 1: Greedy construction from empty network
    greedy_deadline = time.time() + 40.0
    best = _greedy_construct(greedy_deadline)
    
    # If greedy didn't produce a valid network, fall back to seed
    if not _verify_network_bitparallel(best):
        best = _seed_network()
    
    # Phase 2: Prune the network
    prune_deadline = time.time() + 70.0
    best = _prune(best, prune_deadline)
    
    # Phase 3: Local search - try removing comparators
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try to remove any comparator
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_network_bitparallel(candidate):
                best = candidate
                improved = True
                break # Restart scan from end
    
    return best

# EVOLVE-BLOCK-END
