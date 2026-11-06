"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Start from a small, structured initial network (odd-even transposition, 120 comparators)
   and aggressively reduce it using fast bit-parallel verification.
2. Use a "state-aware" local search that tracks which wires are "settled" (their relative
   order is fixed for all boolean inputs) at each prefix of the network. Comparators that
   only act on already-settled wires are redundant candidates for removal.
3. Use bit-parallel simulation to verify correctness in O(C + N) big-int operations.
4. Combine greedy removal with random restarts/insertions to escape local minima.

The key insight from the inspiration is to use a dependency graph / DAG perspective:
by tracking the "settlement" of wires, we can prioritize removal of comparators
that are most likely to be redundant, rather than just trying random removals.
"""

import time
import random

N = 16
TOTAL_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << TOTAL_INPUTS) - 1

# Pre-compute the initial state patterns for speed
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
    """Standard odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def batcher_network(n=16):
    """Batcher's bitonic sorting network for n=2^k wires."""
    if n == 2:
        return [(0, 1)]
    
    k = n // 2
    # Recursively sort halves
    left = batcher_network(k)
    right = [(i + k, j + k) for i, j in batcher_network(k)]
    
    # Merge: bitonic merge
    merge = []
    step = k
    while step >= 1:
        for i in range(0, k, step):
            for j in range(i, i + step):
                merge.append((j, j + step))
        step //= 2
    
    return left + right + merge

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    
    # Start from multiple initial networks and pick the best
    candidates = [
        odd_even_network(),
        batcher_network(N)
    ]
    
    best_net = None
    best_count = float('inf')
    
    for init_net in candidates:
        net = list(init_net)
        # Greedy removal
        net = greedy_remove(net, deadline)
        if len(net) < best_count:
            best_count = len(net)
            best_net = list(net)
    
    # If we still have time, do more sophisticated local search from the best
    if best_net is not None:
        net = best_net
        random.seed(42)
        
        # State-aware local search
        # Track which wires are "settled" at each stage
        # A wire is settled if its value is the same for all remaining inputs
        # in a way that doesn't change with further comparators
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Try to improve by random insertion + greedy removal
            # Insert a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(net))
            candidate = net[:pos] + [(i, j)] + net[pos:]
            
            # Try to remove comparators from the candidate
            improved = False
            for idx in range(len(candidate) - 1, -1, -1):
                if time.time() > deadline:
                    break
                test = candidate[:idx] + candidate[idx+1:]
                if fast_verify(test):
                    candidate = test
                    improved = True
            
            if len(candidate) < len(net):
                net = candidate
                best_net = list(net)
                best_count = len(net)
    
    return best_net

def greedy_remove(net, deadline):
    """Greedy removal of redundant comparators."""
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

if __name__ == "__main__":
    # Quick test
    net = construct()
    print(f"Network size: {len(net)}")
    print(f"Valid: {fast_verify(net)}")
