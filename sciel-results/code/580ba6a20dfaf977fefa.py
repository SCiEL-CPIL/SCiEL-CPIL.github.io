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

def _is_sorting_network(comparators):
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

def _greedy_reduce(comparators, deadline):
    """
    Greedy reduction: try removing each comparator, keep if valid.
    Optimized to avoid list slicing overhead where possible.
    """
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        
        # Try removing comparator at i
        saved = current[i]
        current.pop(i)
        
        if _is_sorting_network(current):
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
    Moves: Remove, Slide, Replace, Move, Swap, Add.
    Includes stall handling with reheating and perturbation.
    """
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    
    # SA Parameters
    T = 2.0
    cooling_rate = 0.995
    min_T = 0.05
    
    steps_since_improvement = 0
    max_stall_steps = 100
    
    while time.time() < deadline and T > min_T:
        # Generate a move
        r = random.random()
        neighbor = list(current)
        move_valid = False
        delta = 0
        
        # Determine move type
        # Probabilities tuned for minimization:
        # Remove: 35%, Slide: 20%, Replace: 15%, Move: 10%, Swap: 10%, Add: 10%
        if r < 0.35:
            # Try Remove
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
                move_valid = True
        elif r < 0.55:
            # Try Slide: Change (i, j) to (i+1, j+1) or (i-1, j-1)
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                i, j = neighbor[idx]
                # Try shifting up or down
                if random.random() < 0.5:
                    # Shift down (increase indices)
                    if i + 1 < N - 1 and j + 1 < N:
                        ni, nj = i + 1, j + 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
                else:
                    # Shift up (decrease indices)
                    if i - 1 >= 0 and j - 1 > 0:
                        ni, nj = i - 1, j - 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
        elif r < 0.70:
            # Try Replace: change the pair of a random comparator
            if len(neighbor) > 0:
                idx = random.randint(0, len(neighbor) - 1)
                # Bias towards adjacent or short distance
                if random.random() < 0.6:
                    dist = random.randint(1, 3)
                else:
                    dist = random.randint(1, N - 1)
                a = random.randint(0, N - 1 - dist)
                b = a + dist
                neighbor[idx] = (a, b)
                delta = 0
                move_valid = True
        elif r < 0.80:
            # Try Move (move a comparator to a new position)
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                    move_valid = True
        elif r < 0.90:
            # Try Swap (exchange two comparators)
            if len(neighbor) > 1:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                    move_valid = True
        else:
            # Try Add
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
            move_valid = True
        
        if not move_valid:
            T *= cooling_rate
            continue

        # Check validity
        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            
            if new_size < best_size:
                best = neighbor
                best_size = new_size
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
            
            # Acceptance criterion
            if delta < 0:
                # Improvement
                current = neighbor
            elif delta == 0:
                # Neutral
                current = neighbor
            else:
                # Worse (Add)
                if random.random() < math.exp(-1.0 / T):
                    current = neighbor
        
        # Stall handling: If stuck for a while, kick the temperature up
        if steps_since_improvement > max_stall_steps:
            T = max(T * 1.5, 0.5)
            steps_since_improvement = 0
            # Also, maybe accept a random perturbation to current
            if random.random() < 0.3 and len(current) > 0:
                # Perturb current by adding a few random comparators
                for _ in range(random.randint(1, 3)):
                    i = random.randint(0, N - 2)
                    j = random.randint(i + 1, N - 1)
                    current.append((i, j))
        
        # Cool down
        T *= cooling_rate
        
    return best

def construct():
    # Start with odd-even transposition network
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best = list(seed_network)
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety
    
    # 1. Greedy reduction to get a good starting point
    best = _greedy_reduce(best, deadline)
    
    # 2. Simulated Annealing with restarts
    while time.time() < deadline:
        # Restart SA from current best or perturbed seed
        if random.random() < 0.5:
            start_net = list(best)
        else:
            start_net = list(seed_network)
            # Perturb seed
            for _ in range(random.randint(0, 10)):
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                start_net.append((i, j))
            start_net = _greedy_reduce(start_net, deadline)
        
        candidate = _simulated_annealing(start_net, deadline)
        
        if len(candidate) < len(best):
            best = candidate
            
    return best
# EVOLVE-BLOCK-END
