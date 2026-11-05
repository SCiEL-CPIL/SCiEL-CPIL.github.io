"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Start from Batcher's bitonic sorter (64 comparators for 16 wires), which is a known correct network.
2. Use fast bit-parallel verification via the zero-one principle.
3. Aggressive greedy removal: repeatedly try to remove each comparator (in various orders) until stuck.
4. Random restarts with insertion: insert a random comparator at a random position, then greedy-remove.
   This helps escape local minima.
5. Also try swapping adjacent comparators and moving comparators to new positions.
6. Time-budgeted: return the best network found before the deadline.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial state patterns
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
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

def batcher_network():
    """Generate Batcher's bitonic sorting network for N=16 wires."""
    comparators = []
    
    def bitonic_merge(start, size, direction):
        """Merge a bitonic sequence of length 'size' starting at 'start'.
        direction=True means ascending, False means descending."""
        if size == 1:
            return
        half = size // 2
        bitonic_merge(start, half, True)
        bitonic_merge(start + half, half, False)
        # Now merge the two halves
        for i in range(half):
            if direction:
                comparators.append((start + i, start + i + half))
            else:
                comparators.append((start + i + half, start + i))
    
    def bitonic_sort(start, size, direction):
        """Sort a sequence of length 'size' starting at 'start'."""
        if size == 1:
            return
        half = size // 2
        bitonic_sort(start, half, True)
        bitonic_sort(start + half, half, False)
        bitonic_merge(start, size, direction)
    
    bitonic_sort(0, N, True)
    return comparators

def odd_even_network():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    random.seed(42)
    
    # Start from Batcher's bitonic sorter (64 comparators)
    net = batcher_network()
    best_net = list(net)
    best_count = len(net)
    
    def greedy_remove(net, time_limit=None):
        """Try to remove each comparator; repeat until no more removals possible."""
        changed = True
        while changed:
            if time_limit is not None and time.time() > time_limit:
                break
            changed = False
            # Try removing from end to start (often more effective)
            for idx in range(len(net) - 1, -1, -1):
                if time_limit is not None and time.time() > time_limit:
                    break
                candidate = net[:idx] + net[idx+1:]
                if fast_verify(candidate):
                    net = candidate
                    changed = True
                    break
        return net
    
    # Initial greedy removal
    net = greedy_remove(net, deadline)
    best_net = list(net)
    best_count = len(net)
    
    # Try multiple strategies:
    # 1. Random insertion + greedy removal
    # 2. Swap adjacent comparators + greedy removal
    # 3. Move a comparator to a different position + greedy removal
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        current_time = time.time()
        remaining = deadline - current_time
        
        # Choose a random strategy
        strategy = random.random()
        
        if strategy < 0.4:
            # Strategy 1: Insert a random comparator, then greedy remove
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [(i, j)] + net[pos:]
            
            # Quick check: is the candidate even potentially useful?
            # Just do greedy removal on it
            candidate = greedy_remove(candidate, deadline)
            
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
                
        elif strategy < 0.7:
            # Strategy 2: Swap two random comparators (or adjacent ones)
            if len(net) >= 2:
                idx1 = random.randint(0, len(net) - 2)
                idx2 = idx1 + random.randint(1, min(5, len(net) - idx1 - 1))
                candidate = list(net)
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                
                candidate = greedy_remove(candidate, deadline)
                
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
                    
        else:
            # Strategy 3: Move a comparator to a different position
            if len(net) >= 2:
                idx = random.randint(0, len(net) - 1)
                new_pos = random.randint(0, len(net) - 1)
                if new_pos != idx:
                    candidate = list(net)
                    comp = candidate.pop(idx)
                    # Adjust new_pos if idx < new_pos
                    if idx < new_pos:
                        new_pos -= 1
                    candidate.insert(new_pos, comp)
                    
                    candidate = greedy_remove(candidate, deadline)
                    
                    if len(candidate) < len(net):
                        net = candidate
                        best_net = list(net)
                        best_count = len(net)
        
        # If we haven't improved in a while, try a more aggressive restart
        # by starting from a different initial network
        if iterations % 50 == 0 and time.time() < deadline - 10:
            # Try starting from odd-even and doing greedy removal
            net2 = odd_even_network()
            net2 = greedy_remove(net2, deadline)
            if len(net2) < best_count:
                net = net2
                best_net = list(net)
                best_count = len(net)
    
    # Final verification
    if not fast_verify(best_net):
        # Fallback to a known-good network
        best_net = batcher_network()
        best_net = greedy_remove(best_net, deadline)
    
    return best_net
