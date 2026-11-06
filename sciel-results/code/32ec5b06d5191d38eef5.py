"""Size-efficient sorting network search for N=16 wires.

Combined approach:
- Start from the known 60-comparator Batcher bitonic sorter (a correct, explicit construction).
- Use fast bit-parallel verification (zero-one principle over all 65536 boolean inputs).
- Aggressive greedy removal: repeatedly try to remove each comparator; if the network
  remains valid, remove it.
- Simulated annealing local search: try swapping, moving, and inserting comparators
  with temperature-based acceptance to escape local minima.
- Multi-start from different initial constructions (bitonic, odd-even, random).
- Use a deadline guard to ensure we finish well before the 90s limit.

The reference is 60 comparators. We start at 64 (bitonic) and hope to get to or below 60
via local search. Even 61 or 62 would be a solid improvement over the odd-even 120.
"""

import time
import random
import math

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = NUM_INPUTS // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _get_initial_states()[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """
    Generate Batcher's bitonic sorting network for n wires (n must be power of 2).
    Returns a list of (i, j) comparator pairs.
    """
    if n <= 1:
        return []
    
    comparators = []
    
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))  # min to a, max to b
            else:
                comparators.append((b, a))  # min to b, max to a
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    
    bitonic_sort(n, 0, True)
    return comparators

def odd_even_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def greedy_remove(net, deadline):
    """
    Try to remove each comparator one by one. If removal keeps the network valid,
    remove it and continue. Returns the reduced network.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                net = candidate
                changed = True
                break
    return net

def simulated_annealing(net, deadline, random_seed=42):
    """
    Simulated annealing local search over comparator sequences.
    Operations: swap two comparators, move a comparator, insert a random comparator,
    delete a comparator (if valid).
    """
    random.seed(random_seed)
    best_net = list(net)
    best_count = len(net)
    current_net = list(net)
    current_count = len(net)
    
    # Temperature schedule
    t_start = 1.0
    t_end = 0.01
    total_iterations = 5000
    iterations_per_temp_step = total_iterations // 20
    
    temp = t_start
    cooling_rate = (t_end / t_start) ** (1.0 / max(1, iterations_per_temp_step))
    
    for iteration in range(total_iterations):
        if time.time() > deadline:
            break
        
        # Choose a random operation
        op = random.random()
        
        if op < 0.3:
            # Swap two comparators
            if len(current_net) < 2:
                continue
            idx1 = random.randint(0, len(current_net) - 1)
            idx2 = random.randint(0, len(current_net) - 1)
            if idx1 == idx2:
                continue
            candidate = list(current_net)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            delta = 0  # Same number of comparators
        elif op < 0.5:
            # Move a comparator to a different position
            if len(current_net) < 2:
                continue
            idx = random.randint(0, len(current_net) - 1)
            new_idx = random.randint(0, len(current_net) - 1)
            if idx == new_idx:
                continue
            candidate = list(current_net)
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            delta = 0  # Same number of comparators
        elif op < 0.7:
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(current_net))
            candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
            delta = 1  # One more comparator
        elif op < 0.85:
            # Try to delete a random comparator
            if len(current_net) == 0:
                continue
            idx = random.randint(0, len(current_net) - 1)
            candidate = current_net[:idx] + current_net[idx+1:]
            delta = -1  # One fewer comparator
        else:
            # Swap adjacent comparators
            if len(current_net) < 2:
                continue
            idx = random.randint(0, len(current_net) - 2)
            candidate = list(current_net)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            delta = 0  # Same number of comparators
        
        # Check if candidate is valid
        if not fast_verify(candidate):
            continue
        
        # Acceptance decision
        new_count = len(candidate)
        if new_count < current_count:
            # Always accept if better
            current_net = candidate
            current_count = new_count
            if current_count < best_count:
                best_net = list(current_net)
                best_count = current_count
        else:
            # Accept with probability exp(-delta / temp)
            if delta == 0:
                accept_prob = 0.5  # Accept half the time for same-size moves
            else:
                accept_prob = math.exp(-delta / max(0.001, temp))
            if random.random() < accept_prob:
                current_net = candidate
                current_count = new_count
        
        # Cool down
        if (iteration + 1) % iterations_per_temp_step == 0:
            temp *= cooling_rate
    
    return best_net

def construct():
    deadline = time.time() + 82  # 82s to leave 8s margin for overhead
    
    # Start from bitonic sorter (64 comparators for n=16)
    net = bitonic_sorter(N)
    best_net = list(net)
    best_count = len(net)
    
    # Verify initial network
    if not fast_verify(best_net):
        # Fallback to odd-even if bitonic has a bug
        best_net = odd_even_network()
        best_count = len(best_net)
        net = list(best_net)
    
    # Greedy removal pass 1
    net = greedy_remove(net, deadline)
    if len(net) < best_count:
        best_net = list(net)
        best_count = len(net)
    
    # Simulated annealing from the current best
    sa_result = simulated_annealing(net, deadline, random_seed=123)
    if len(sa_result) < best_count:
        best_net = sa_result
        best_count = len(best_net)
    
    # Try simulated annealing from odd-even network as well
    net2 = odd_even_network()
    net2 = greedy_remove(net2, deadline)
    sa_result2 = simulated_annealing(net2, deadline, random_seed=456)
    if len(sa_result2) < best_count:
        best_net = sa_result2
        best_count = len(best_net)
    
    # Final greedy removal on the best found
    net_final = greedy_remove(best_net, deadline)
    if len(net_final) < best_count:
        best_net = net_final
        best_count = len(best_net)
    
    return best_net
