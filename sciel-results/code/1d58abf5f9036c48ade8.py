"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import math

_INITIAL_BITSETS = None

def _init_bitsets():
    """
    Initialize 18 bitsets, each of length 2^N.
    bitset[k] has bit i set if the i-th input (0 to 2^N-1) has a 1 in wire k.
    """
    total_bits = 1 << N
    bitsets = [0] * N
    
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        # Mask for one period: lower half 0s, upper half 1s
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        
        # Repeat pattern to fill total_bits
        # Use doubling
        current_val = pattern
        current_len = period
        
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
            
        bitsets[k] = current_val

    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network_full(comparators):
    """
    Check if network sorts all 2^N Boolean inputs using bitset simulation.
    """
    if not comparators:
        return N <= 1
        
    # Get a copy of the initial bitsets
    bitsets = list(_get_initial_bitsets())
    
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
        
    # Check if sorted:
    # For all k < l, bitsets[k] must be a subset of bitsets[l].
    # i.e., bitsets[k] & ~bitsets[l] == 0.
    # It is sufficient to check adjacent wires: k and k+1.
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
            
    return True

def _is_sorting_network_sampled(comparators, K=2048, seed=42):
    """
    Fast check using K random 0-1 inputs.
    Returns True if all K inputs are sorted.
    """
    if not comparators:
        return N <= 1
    
    # Generate K random 0-1 inputs
    # We pack them into N integers, one per wire.
    # bit w of integer wire[k] corresponds to input w.
    rng = random.Random(seed)
    wire_bits = [0] * N
    for w in range(K):
        for k in range(N):
            if rng.randint(0, 1):
                wire_bits[k] |= (1 << w)
                
    # Simulate
    for i, j in comparators:
        bi = wire_bits[i]
        bj = wire_bits[j]
        wire_bits[i] = bi & bj
        wire_bits[j] = bi | bj
        
    # Check sorted
    for k in range(N - 1):
        if wire_bits[k] & ~wire_bits[k + 1]:
            return False
    return True

def _greedy_reduce(comparators, deadline, use_full=False):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    Uses sampled check for speed unless use_full is True.
    """
    current = list(comparators)
    
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        
        # Try removing comparator at i
        saved = current[i]
        current.pop(i)
        
        if use_full:
            valid = _is_sorting_network_full(current)
        else:
            valid = _is_sorting_network_sampled(current)
            
        if valid:
            # Keep it removed
            pass
        else:
            # Restore
            current.insert(i, saved)
            i += 1
        
    return current

def _simulated_annealing(start_network, deadline):
    """
    Simulated Annealing to find a smaller sorting network.
    Moves: Remove, Add, Swap, Move.
    Uses sampled verification for speed.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    # SA Parameters
    T = 2.0
    cooling_rate = 0.99
    
    max_idx = N - 1
    
    while time.time() < deadline and T > 0.05:
        r = random.random()
        neighbor = list(current)
        
        if r < 0.4:
            # Try Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
            else:
                continue
        elif r < 0.7:
            # Try Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
        elif r < 0.85:
            # Try Swap
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                else:
                    continue
            else:
                continue
        else:
            # Try Move
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                else:
                    continue
            else:
                continue
        
        if _is_sorting_network_sampled(neighbor):
            new_size = len(neighbor)
            
            if new_size < best_size:
                best = neighbor
                best_size = new_size
            
            if delta < 0:
                current = neighbor
            elif delta == 0:
                current = neighbor
            else:
                if random.random() < math.exp(-1.0 / T):
                    current = neighbor
        
        T *= cooling_rate
        
    return best

def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_sampled = list(seed_network)
    best_full = list(seed_network)
    
    start_time = time.time()
    deadline = start_time + 50  # 10s margin for safety
    
    # 1. Initial Greedy reduction (fast)
    best_sampled = _greedy_reduce(best_sampled, deadline)
    
    # Verify full
    if _is_sorting_network_full(best_sampled):
        best_full = list(best_sampled)
    else:
        # Fallback to seed if reduction broke it (shouldn't happen with valid seed)
        best_full = list(seed_network)
        
    # 2. Simulated Annealing loops
    last_full_check = time.time()
    
    while time.time() < deadline:
        # Decide start network
        if random.random() < 0.6:
            start_net = list(best_sampled)
        else:
            start_net = list(seed_network)
            num_noise = random.randint(5, 15)
            for _ in range(num_noise):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            
            reduce_deadline = min(deadline, time.time() + 2)
            start_net = _greedy_reduce(start_net, reduce_deadline)
            
            if time.time() > deadline:
                break
        
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best_sampled):
            best_sampled = candidate
            # Greedy reduce again
            if time.time() < deadline:
                quick_deadline = min(deadline, time.time() + 1)
                best_sampled = _greedy_reduce(best_sampled, quick_deadline)
                
        # Periodic full verification
        if time.time() - last_full_check > 5.0:
            last_full_check = time.time()
            if _is_sorting_network_full(best_sampled):
                best_full = list(best_sampled)
            # If it fails, we keep the previous best_full
    
    # Final check
    if time.time() < deadline:
        if _is_sorting_network_full(best_sampled):
            best_full = list(best_sampled)
            
    return best_full
# EVOLVE-BLOCK-END
