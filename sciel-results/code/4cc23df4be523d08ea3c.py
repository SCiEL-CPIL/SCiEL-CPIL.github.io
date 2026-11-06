"""
Size-efficient sorting networks on 17 wires.
Combines bitmask-based validation (for speed) with a hybrid search:
1. Greedy deletion to quickly reduce size from the seed.
2. Simulated Annealing to escape local minima and find smaller networks.
"""
import time
import random
import math

N = 17
TOTAL_INPUTS = 1 << N  # 131072

# Precompute initial state bitmasks for verification.
# state[i] is a bitmask where bit k is 1 if input k has a 1 at wire i.
# We can construct these efficiently.
INITIAL_STATE = [0] * N
for i in range(N):
    mask = 0
    bit_val = 1 << i
    # Pattern for wire i: blocks of 2^i zeros, 2^i ones
    # We can use the iterative shift method or direct construction.
    # Direct construction is faster for precomputation.
    block = 1 << i
    val = 0
    for m in range(block, TOTAL_INPUTS, 2 * block):
        bits_to_set = (1 << block) - 1
        val |= bits_to_set << m
    INITIAL_STATE[i] = val

def _is_valid(network):
    """
    Check if network sorts all 0/1 inputs using zero-one principle.
    Uses bitmask simulation for speed.
    """
    if not network:
        return N <= 1
    
    # Copy initial state
    state = list(INITIAL_STATE)
    n = N
    
    # Apply comparators
    # Optimization: local variable binding
    st = state
    for i, j in network:
        si = st[i]
        sj = st[j]
        st[i] = si & sj
        st[j] = si | sj
        
    # Check if state is sorted: S[0] subseteq S[1] subseteq ... subseteq S[N-1]
    # This is equivalent to checking S[i] & ~S[i+1] == 0 for all i
    for i in range(n - 1):
        if st[i] & ~st[i + 1]:
            return False
            
    return True

def _construct_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _greedy_delete(network, deadline):
    """
    Attempt to remove redundant comparators one by one.
    Returns the reduced network.
    """
    current = list(network)
    
    while time.time() < deadline:
        removed_any = False
        # Try removing each comparator in random order
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            # Try removing this comparator
            test_network = current[:idx] + current[idx+1:]
            if _is_valid(test_network):
                current = test_network
                removed_any = True
                break
        
        if not removed_any:
            break
            
    return current

def construct():
    """
    Construct a sorting network with as few comparators as possible.
    Uses a combination of greedy deletion and simulated annealing.
    """
    start_time = time.time()
    # 90s limit, keep 10s margin for safety
    deadline = start_time + 80.0
    
    # Start with Odd-Even Transposition
    current_network = _construct_odd_even_transposition()
    best_network = list(current_network)
    best_len = len(current_network)
    
    # Verify start
    if not _is_valid(current_network):
        return best_network

    # Phase 1: Greedy Deletion
    # This is fast and often finds a local minimum quickly
    greedy_deadline = start_time + 20.0  # Spend first 20s on greedy
    current_network = _greedy_delete(current_network, greedy_deadline)
    
    if len(current_network) < best_len:
        best_network = list(current_network)
        best_len = len(current_network)
    
    current_len = len(current_network)
    
    # Phase 2: Simulated Annealing
    # Use the remaining time to explore the search space more broadly
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    min_T = 0.1
    
    while time.time() < deadline:
        if T < min_T:
            # Reheat or stop? Stopping is safer for finding global best
            break
            
        # Generate a candidate move
        move_type = random.random()
        
        if move_type < 0.4:
            # Delete a random comparator
            if len(current_network) > 10: # Don't delete too many
                idx = random.randint(0, len(current_network) - 1)
                candidate = current_network[:idx] + current_network[idx+1:]
                new_len = current_len - 1
            else:
                continue
        elif move_type < 0.6:
            # Add a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            # Insert at random position
            idx = random.randint(0, len(current_network))
            candidate = current_network[:idx] + [(i, j)] + current_network[idx:]
            new_len = current_len + 1
        elif move_type < 0.8:
            # Swap two comparators
            if len(current_network) < 2:
                continue
            idx1 = random.randint(0, len(current_network) - 1)
            idx2 = random.randint(0, len(current_network) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current_network)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            new_len = current_len
        else:
            # Modify a comparator (change wires)
            if len(current_network) == 0:
                continue
            idx = random.randint(0, len(current_network) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            candidate = list(current_network)
            candidate[idx] = (i, j)
            new_len = current_len
        
        # Verify candidate
        if not _is_valid(candidate):
            # Invalid, reject
            continue
            
        # Accept or reject based on SA
        delta_cost = new_len - current_len
        if delta_cost < 0:
            # Improvement
            current_network = candidate
            current_len = new_len
            if current_len < best_len:
                best_network = list(current_network)
                best_len = current_len
        else:
            # Worse or same, accept with probability exp(-delta/T)
            if delta_cost == 0:
                # Accept with some probability to allow drift
                if random.random() < 0.5:
                    current_network = candidate
            else:
                prob = math.exp(-delta_cost / T)
                if random.random() < prob:
                    current_network = candidate
                    current_len = new_len
        
        # Cool down
        T *= cooling_rate
        
    return best_network

# EVOLVE-BLOCK-END
