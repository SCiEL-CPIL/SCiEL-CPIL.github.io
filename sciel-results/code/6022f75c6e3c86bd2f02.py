import time
import random
import sys

# Global cache for priority sets
_PRIORITY_CACHE = {}

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 310.0  # Leave 20s safety margin before 330s kill


def _int_to_tuple(idx, n):
    """Convert an integer index to a tuple of digits in base 3."""
    v = []
    x = idx
    for _ in range(n):
        v.append(x % 3)
        x //= 3
    return tuple(v)


def _tuple_to_int(v):
    """Convert a tuple of digits in base 3 to an integer index."""
    idx = 0
    for i, d in enumerate(reversed(v)):
        idx += d * (3 ** i)
    return idx


def _greedy_cap_set(n, seed, time_limit, priority_mode="random"):
    """
    Greedy cap set with various priority modes.
    """
    start = time.time()
    total = 3 ** n
    
    # Precompute all vectors
    all_vecs = [_int_to_tuple(i, n) for i in range(total)]
    
    # Create ordering based on priority mode
    if priority_mode == "random":
        rng = random.Random(seed)
        order = list(range(total))
        rng.shuffle(order)
    elif priority_mode == "zero_count":
        zeros = [sum(1 for d in v if d == 0) for v in all_vecs]
        order = list(range(total))
        rng = random.Random(seed)
        groups = {}
        for i in range(total):
            z = zeros[i]
            if z not in groups:
                groups[z] = []
            groups[z].append(i)
        order = []
        for z in sorted(groups.keys(), reverse=True):
            rng.shuffle(groups[z])
            order.extend(groups[z])
    elif priority_mode == "nonzero_count":
        nonzeros = [sum(1 for d in v if d != 0) for v in all_vecs]
        order = list(range(total))
        rng = random.Random(seed)
        groups = {}
        for i in range(total):
            nz = nonzeros[i]
            if nz not in groups:
                groups[nz] = []
            groups[nz].append(i)
        order = []
        for nz in sorted(groups.keys()):
            rng.shuffle(groups[nz])
            order.extend(groups[nz])
    elif priority_mode == "max_digit":
        maxd = [max(v) for v in all_vecs]
        order = list(range(total))
        rng = random.Random(seed)
        groups = {}
        for i in range(total):
            md = maxd[i]
            if md not in groups:
                groups[md] = []
            groups[md].append(i)
        order = []
        for md in sorted(groups.keys()):
            rng.shuffle(groups[md])
            order.extend(groups[md])
    elif priority_mode == "hybrid_zero_nonzero":
        zeros = [sum(1 for d in v if d == 0) for v in all_vecs]
        nonzeros = [sum(1 for d in v if d != 0) for v in all_vecs]
        scores = [z * 2 - nz for z, nz in zip(zeros, nonzeros)]
        order = list(range(total))
        rng = random.Random(seed)
        groups = {}
        for i in range(total):
            s = scores[i]
            if s not in groups:
                groups[s] = []
            groups[s].append(i)
        order = []
        for k in sorted(groups.keys(), reverse=True):
            rng.shuffle(groups[k])
            order.extend(groups[k])
    elif priority_mode == "coord_sum":
        sums = [sum(v) for v in all_vecs]
        order = list(range(total))
        rng = random.Random(seed)
        groups = {}
        for i in range(total):
            s = sums[i]
            if s not in groups:
                groups[s] = []
            groups[s].append(i)
        order = []
        for k in sorted(groups.keys()):
            rng.shuffle(groups[k])
            order.extend(groups[k])
    else:
        order = list(range(total))
    
    S = set()
    forbidden = set()
    
    for idx in order:
        if time.time() - start > time_limit:
            break
        if idx in forbidden:
            continue
        
        S.add(idx)
        v = all_vecs[idx]
        
        for a_idx in S:
            if a_idx == idx:
                continue
            a = all_vecs[a_idx]
            w = tuple((2 * v[i] - a[i]) % 3 for i in range(n))
            w_idx = _tuple_to_int(w)
            forbidden.add(w_idx)
    
    return S


def priority(vector, n):
    if n not in _PRIORITY_CACHE:
        # Perform search for this n
        time_limit = 250.0  # Leave time for overhead
        
        best_set = set()
        best_size = 0
        
        start_search = time.time()
        
        # Try multiple strategies with different seeds
        strategies = [
            ("random", [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]),
            ("zero_count", [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]),
            ("nonzero_count", [42, 123, 456, 789, 101, 202]),
            ("max_digit", [42, 123, 456, 789]),
            ("hybrid_zero_nonzero", [42, 123, 456, 789, 101, 202]),
            ("coord_sum", [42, 123, 456, 789]),
        ]
        
        for mode, seeds in strategies:
            for seed in seeds:
                if time.time() - start_search > time_limit:
                    break
                remaining = time_limit - (time.time() - start_search)
                if remaining < 5.0:
                    break
                
                # Use a fixed time per run, capped by remaining time
                per_run_time = min(4.0, remaining)
                S = _greedy_cap_set(n, seed, per_run_time, mode)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S
        
        # Convert best_set (indices) to set of tuples
        best_tuples = set()
        for idx in best_set:
            best_tuples.add(_int_to_tuple(idx, n))
        
        _PRIORITY_CACHE[n] = best_tuples
    
    return 1.0 if vector in _PRIORITY_CACHE[n] else 0.0
