import time
import random

N = 16
DEADLINE = 85.0  # Safety margin for 90s limit

_INITIAL_BITSETS = None

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = []
        for i in range(N):
            mask = 1 << i
            bitset = 0
            for k in range(1 << N):
                if k & mask:
                    bitset |= (1 << k)
            _INITIAL_BITSETS.append(bitset)
    return _INITIAL_BITSETS

def _verify_network(net):
    initial = _get_initial_bitsets()
    state = initial[:]
    
    for i, j in net:
        new_i = state[i] & state[j]
        new_j = state[i] | state[j]
        state[i] = new_i
        state[j] = new_j
        
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
            
    return True

def _base_60_network():
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def construct():
    start_time = time.time()
    
    base = _base_60_network()
    # Ensure base is valid
    if not _verify_network(base):
        return base

    best_net = base[:]
    best_len = len(base)
    current_net = base[:]
    
    rng = random.Random(12345)
    
    while time.time() < DEADLINE:
        removed = False
        
        # Systematic deletion attempt
        # We iterate backwards to avoid index shifting issues during single pass, 
        # but since we break on first success, it's fine.
        for i in range(len(current_net) - 1, -1, -1):
            if time.time() > DEADLINE: break
            
            trial = current_net[:i] + current_net[i+1:]
            if _verify_network(trial):
                current_net = trial
                removed = True
                if len(current_net) < best_len:
                    best_net = current_net[:]
                    best_len = len(current_net)
                break
        
        # If no removal found, try random swaps or insertions to escape local minima
        if not removed:
            # Check time before expensive random ops
            if time.time() < DEADLINE - 5:
                move_type = rng.random()
                if move_type < 0.7:
                    # Swap
                    if len(current_net) > 1:
                        i, j = rng.sample(range(len(current_net)), 2)
                        trial = current_net[:]
                        trial[i], trial[j] = trial[j], trial[i]
                        if _verify_network(trial):
                            current_net = trial
                else:
                    # Insert
                    i, j = rng.sample(range(N), 2)
                    if i > j: i, j = j, i
                    pos = rng.randrange(len(current_net) + 1)
                    trial = current_net[:pos] + [(i, j)] + current_net[pos:]
                    if _verify_network(trial):
                        current_net = trial

    return best_net
