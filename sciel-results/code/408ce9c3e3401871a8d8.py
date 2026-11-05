<<<<<<< SEARCH
"""Size-efficient sorting network search for N=16 wires.

DIRECT approach:
- Start from the known 64-comparator Batcher bitonic sorter (a correct, explicit construction).
- Use fast bit-parallel verification (zero-one principle over all 65536 boolean inputs).
- Aggressive greedy removal: repeatedly try to remove each comparator; if the network
  remains valid, remove it. This is cheap and often reduces size significantly.
- Then do a more sophisticated local search: try inserting a comparator at a random
  position, then do another greedy removal pass. Keep the best network found.
- Use a deadline guard to ensure we finish well before the 90s limit.

The reference is 60 comparators. We start at 64 (bitonic) and hope to get to or below 60
via local search. Even 61 or 62 would be a solid improvement over the odd-even 120.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

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
=======
"""Size-efficient sorting network search for N=16 wires.

DIRECT approach:
- Start from the known 64-comparator Batcher bitonic sorter (a correct, explicit construction).
- Use fast bit-parallel verification (zero-one principle over all 65536 boolean inputs).
- Aggressive greedy removal with randomized order to escape local minima.
- Targeted local search: insert comparators at high-value positions, then greedy remove.
- Use a deadline guard to ensure we finish well before the 90s limit.

The reference is 60 comparators. We start at 64 (bitonic) and hope to get to or below 60
via local search. Even 61 or 62 would be a solid improvement over the odd-even 120.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536
ALL_ONES = (1 << NUM_INPUTS) - 1

# Pre-compute initial states at module load for speed
_initial_states = []
for _i in range(N):
    _val = 0
    _period = 1 << (_i + 1)
    _half = 1 << _i
    _period_val = ((1 << _half) - 1) << _half
    _num_periods = NUM_INPUTS // _period
    _shift = 0
    for _ in range(_num_periods):
        _val |= _period_val << _shift
        _shift += _period
    _initial_states.append(_val)

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = _initial_states[:]
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True
>>>>>>> REPLACE

