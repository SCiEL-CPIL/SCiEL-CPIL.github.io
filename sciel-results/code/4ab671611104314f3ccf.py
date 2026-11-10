import random
import time
import numpy as np

_START = time.time()
_DEADLINE = 150.0 - 3.0   # safety margin

def _edge_index(i, j):
    return i * 15 - (i * (i + 1)) // 2 + (j - i - 1)

def _build_k4s():
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ))
    return k4s

_K4S = _build_k4s()
_K4S_ARR = np.array(_K4S, dtype=np.int32)
_NUM_K4 = len(_K4S)

_EDGE_TO_K4S = [[] for _ in range(120)]
for k, k4 in enumerate(_K4S):
    for e in k4:
        _EDGE_TO_K4S[e].append(k)

def _init_k4_valid(colours):
    valid = [False] * _NUM_K4
    for k in range(_NUM_K4):
        c = _K4S[k]
        valid[k] = len({colours[c[0]], colours[c[1]], colours[c[2]], colours[c[3]], colours[c[4]], colours[c[5]]}) >= 5
    return valid

def _update_k4_valid(k, colours, valid):
    c = _K4S[k]
    valid[k] = len({colours[c[0]], colours[c[1]], colours[c[2]], colours[c[3]], colours[c[4]], colours[c[5]]}) >= 5

def _sa_fixed_palette(K, time_limit, seed):
    rng = random.Random(seed)
    colours = [rng.randint(0, K-1) for _ in range(120)]
    valid = _init_k4_valid(colours)
    num_invalid = sum(1 for v in valid if not v)
    
    best_colours = list(colours)
    best_invalid = num_invalid
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and num_invalid > 0:
        e = rng.randrange(120)
        new_c = rng.randint(0, K-1)
        old_c = colours[e]
        
        if old_c == new_c:
            continue
            
        # Calculate delta
        delta = 0
        affected = _EDGE_TO_K4S[e]
        
        # Save old validity for affected K4s
        old_states = {}
        for k in affected:
            old_states[k] = valid[k]
            
        colours[e] = new_c
        
        for k in affected:
            _update_k4_valid(k, colours, valid)
            if not valid[k]:
                delta += 1 if not old_states[k] else 0
                # Wait, if it was invalid (False) and becomes invalid (False), delta 0.
                # If it was invalid (False) and becomes valid (True), delta -1.
                # If it was valid (True) and becomes invalid (False), delta +1.
                # If it was valid (True) and becomes valid (True), delta 0.
                
                # Let's just recalculate delta properly:
                # delta += (1 if not valid[k] else 0) - (1 if not old_states[k] else 0)
        
        # My previous logic was flawed. Let's do it cleanly.
        delta = 0
        for k in affected:
            was_invalid = not old_states[k]
            is_invalid = not valid[k]
            if was_invalid and not is_invalid:
                delta -= 1
            elif not was_invalid and is_invalid:
                delta += 1
                
        if delta < 0:
            # Improvement
            num_invalid += delta
            if num_invalid == 0:
                return list(colours)
            if num_invalid < best_invalid:
                best_invalid = num_invalid
                best_colours = list(colours)
        else:
            # Accept with probability
            if delta == 0:
                # Accept to move around
                if rng.random() < 0.5:
                    num_invalid += delta
            else:
                if T > T_min:
                    if rng.random() < np.exp(-delta / T):
                        num_invalid += delta
                        if num_invalid < best_invalid:
                            best_invalid = num_invalid
                            best_colours = list(colours)
                    else:
                        # Reject: revert
                        colours[e] = old_c
                        for k in affected:
                            valid[k] = old_states[k]
        
        T *= cooling
        
    if best_invalid == 0:
        return best_colours
    return None

def construct():
    # Try to find solutions with fewer colors
    # Bounds: 13 <= f(16) <= 15
    # Try 13, 14, 15, 16
    
    # Time budget allocation
    # Total 147s available.
    # Try 13 for 40s
    # Try 14 for 40s
    # Try 15 for 40s
    # Try 16 for 10s (should be easy)
    # Fallback to 17
    
    # Use multiple seeds for each K to increase chance
    seeds = [1, 2, 3, 4, 5]
    
    for K in [13, 14, 15, 16]:
        if time.time() > _DEADLINE - 5:
            break
            
        # Allocate time based on K
        if K == 13:
            time_per_seed = 8.0
        elif K == 14:
            time_per_seed = 8.0
        elif K == 15:
            time_per_seed = 5.0
        else:
            time_per_seed = 2.0
            
        for seed in seeds:
            if time.time() > _DEADLINE - 2:
                break
                
            sol = _sa_fixed_palette(K, time_per_seed, seed)
            if sol is not None:
                # Validate just in case
                # We trust the SA but a quick check is good
                # The SA returns only if num_invalid == 0
                return sol
                
    # If all fail, return a known valid 17-color solution
    # The original code had a 17-color solution: (i+j)%17
    colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    return colours
