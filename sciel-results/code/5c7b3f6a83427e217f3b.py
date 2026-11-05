"""Size-efficient sorting network search for N=16 wires.

Strategy:
1. Start from Batcher's Bitonic Sorter (80 comparators).
2. Use local search with greedy removal to reduce comparator count.
3. Use bit-parallel verification with caching.
"""

import time
import random

N = 16
ALL_ONES = (1 << (1 << N)) - 1

_initial_states = None
_verify_cache = {}

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            val = 0
            period = 1 << (i + 1)
            half = 1 << i
            period_val = ((1 << half) - 1) << half
            num_periods = (1 << N) // period
            shift = 0
            for _ in range(num_periods):
                val |= period_val << shift
                shift += period
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    state = list(_get_initial_states())
    
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def verify_cached(network):
    key = tuple(network)
    if key in _verify_cache:
        return _verify_cache[key]
    res = fast_verify(network)
    _verify_cache[key] = res
    return res

def bitonic_merge(n):
    if n == 1:
        return []
    half = n // 2
    comps = []
    for i in range(half):
        comps.append((i, i + half))
    left = bitonic_merge(half)
    right = bitonic_merge(half)
    right_offset = [(i + half, j + half) for (i, j) in right]
    comps.extend(left)
    comps.extend(right_offset)
    return comps

def bitonic_sort(n):
    if n == 1:
        return []
    half = n // 2
    net1 = bitonic_sort(half)
    
    net2_asc = bitonic_sort(half)
    net2_desc = [(half - 1 - j, half - 1 - i) for (i, j) in net2_asc]
    net2_desc = [(i + half, j + half) for (i, j) in net2_desc]
    
    merge_net = bitonic_merge(n)
    
    return net1 + net2_desc + merge_net

def construct():
    deadline = time.time() + 85
    
    net = bitonic_sort(N)
    best_net = list(net)
    best_count = len(net)
    
    def greedy_remove(net):
        changed = True
        while changed:
            changed = False
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > deadline:
                    return net
                candidate = net[:idx] + net[idx+1:]
                if verify_cached(candidate):
                    net = candidate
                    changed = True
        return net
    
    net = greedy_remove(net)
    best_net = list(net)
    best_count = len(net)
    
    random.seed(42)
    
    while time.time() < deadline:
        move_type = random.randint(0, 2)
        
        if len(net) > 0:
            if move_type == 0 and len(net) >= 2:
                i, j = random.sample(range(len(net)), 2)
                candidate = list(net)
                candidate[i], candidate[j] = candidate[j], candidate[i]
            elif move_type == 1:
                i = random.randint(0, len(net) - 1)
                j = random.randint(0, len(net) - 1)
                if i == j:
                    continue
                comp = net[i]
                candidate = net[:i] + net[i+1:]
                candidate.insert(j, comp)
            elif move_type == 2:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                pos = random.randint(0, len(net))
                candidate = net[:pos] + [(i, j)] + net[pos:]
            else:
                continue
        else:
            continue
            
        candidate = greedy_remove(candidate)
        
        if len(candidate) < len(net):
            net = candidate
            best_net = list(net)
            best_count = len(net)
            
    return best_net
