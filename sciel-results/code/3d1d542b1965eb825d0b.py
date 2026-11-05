"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Use bit-parallel zero-one verification for fast local search.
2. Start from odd-even transposition (120 comparators).
3. Aggressive greedy removal to get down quickly.
4. Then do a more sophisticated local search:
   - Try swapping adjacent comparators.
   - Try inserting a comparator and then removing others (kick-start).
   - Try moving a comparator to a different position.
   - Try reordering non-adjacent comparators.
5. Keep the best network found within the time budget.

Key structural improvements over the baseline:
- Pre-compute initial states once (cached globally) for speed.
- More diverse move types including non-adjacent swaps and multi-insert kicks.
- Adaptive time management: spend more time on removal when close to optimum.
- Try both forward and backward removal passes.
"""

import time
import random

N = 16
TOTAL_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << TOTAL_INPUTS) - 1

# Pre-compute initial states for all 16 wires
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
            num_periods = TOTAL_INPUTS // period
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

def odd_even_network():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    random.seed(42)
    
    # Start from odd-even transposition
    net = odd_even_network()
    best_net = list(net)
    best_count = len(net)
    
    # Phase 1: Greedy removal until stuck
    def try_remove_all(net):
        """Try to remove each comparator; repeat until no more removals possible."""
        changed = True
        while changed:
            changed = False
            # Try removing from back to front
            for idx in range(len(net) - 1, -1, -1):
                candidate = net[:idx] + net[idx+1:]
                if fast_verify(candidate):
                    net = candidate
                    changed = True
                    break
        return net
    
    net = try_remove_all(net)
    best_net = list(net)
    best_count = len(net)
    
    # Phase 2: Local search with various moves
    def try_remove_one(net, start_idx=0):
        """Try to remove a single comparator from net, starting search from start_idx."""
        for idx in range(start_idx, len(net)):
            candidate = net[:idx] + net[idx+1:]
            if fast_verify(candidate):
                return candidate
        return net
    
    def try_remove_multiple(net, max_removals=5):
        """Try to remove up to max_removals comparators."""
        for _ in range(max_removals):
            new_net = try_remove_one(net)
            if len(new_net) < len(net):
                net = new_net
            else:
                break
        return net
    
    for iteration in range(10000):
        if time.time() > deadline:
            break
        
        # Adaptive: as we get closer to 60, spend more effort on removal
        progress = (120 - len(net)) / 60.0  # 0 at start, 1 at 60
        removal_weight = 0.3 + 0.4 * min(progress, 1.0)
        
        r = random.random()
        if r < removal_weight:
            # Try to remove a comparator
            candidate = try_remove_one(net)
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
        
        elif r < removal_weight + 0.2:
            # Insert a random comparator, then try to remove others
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [(i, j)] + net[pos:]
            
            # Try to remove comparators from the candidate
            candidate = try_remove_multiple(candidate, max_removals=4)
            
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
        
        elif r < removal_weight + 0.35:
            # Swap two adjacent comparators
            if len(net) < 2:
                continue
            idx = random.randint(0, len(net) - 2)
            candidate = net[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if fast_verify(candidate):
                # Try to remove some comparators after the swap
                candidate = try_remove_multiple(candidate, max_removals=2)
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
                else:
                    net = candidate
        
        elif r < removal_weight + 0.5:
            # Move a comparator to a different position
            if len(net) < 2:
                continue
            src = random.randint(0, len(net) - 1)
            dst = random.randint(0, len(net) - 1)
            if src == dst:
                continue
            candidate = net[:]
            comp = candidate.pop(src)
            candidate.insert(dst, comp)
            if fast_verify(candidate):
                # Try to remove some comparators
                candidate = try_remove_multiple(candidate, max_removals=3)
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
                else:
                    net = candidate
        
        else:
            # Swap two non-adjacent comparators
            if len(net) < 2:
                continue
            idx1 = random.randint(0, len(net) - 1)
            idx2 = random.randint(0, len(net) - 1)
            if idx1 == idx2:
                continue
            candidate = net[:]
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            if fast_verify(candidate):
                # Try to remove some comparators
                candidate = try_remove_multiple(candidate, max_removals=2)
                if len(candidate) < len(net):
                    net = candidate
                    best_net = list(net)
                    best_count = len(net)
                else:
                    net = candidate
    
    return best_net
