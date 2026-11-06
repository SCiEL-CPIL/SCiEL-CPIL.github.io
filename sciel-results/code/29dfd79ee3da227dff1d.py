import time
import random
import numpy as np

N = 32
MAX_VAL = 512
MAX_D = 255
TIME_LIMIT = 115  # Safety margin

def score_bitset(bits):
    """Count distinct d>0 such that a, a+d, a+2d all in set."""
    count = 0
    # We can optimize this by checking only up to MAX_D
    for d in range(1, MAX_D + 1):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count

def bits_to_list(bits):
    return [i for i in range(1, MAX_VAL + 1) if bits & (1 << i)]

def list_to_bits(lst):
    bits = 0
    for x in lst:
        bits |= (1 << x)
    return bits

def greedy_construct():
    """Construct a set by greedily adding the element with max marginal gain."""
    bits = 0
    used = set()
    current_score = 0
    
    for _ in range(N):
        best_gain = -1
        best_val = None
        # Optimization: Only check values that might be useful?
        # For now, check all 1..512
        for v in range(1, MAX_VAL + 1):
            if v in used:
                continue
            new_bits = bits | (1 << v)
            new_score = score_bitset(new_bits)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_val = v
        
        if best_val is None:
            break
        bits |= (1 << best_val)
        used.add(best_val)
        current_score = best_gain + current_score
        
    return bits, current_score

def local_search_polish(bits, score, time_budget=5.0):
    """Exhaustive single-swap local search until no improvement."""
    best_bits = bits
    best_score = score
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        in_set = []
        out_set = []
        for v in range(1, MAX_VAL + 1):
            if best_bits & (1 << v):
                in_set.append(v)
            else:
                out_set.append(v)
        
        if not in_set or not out_set:
            break
        
        best_swap_gain = 0
        best_swap = None
        
        # Try all swaps
        for v_remove in in_set:
            for v_add in out_set:
                new_bits = (best_bits & ~(1 << v_remove)) | (1 << v_add)
                new_score = score_bitset(new_bits)
                gain = new_score - best_score
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (v_remove, v_add)
        
        if best_swap is not None and best_swap_gain > 0:
            v_remove, v_add = best_swap
            best_bits = (best_bits & ~(1 << v_remove)) | (1 << v_add)
            best_score += best_swap_gain
        else:
            break
    
    return best_bits, best_score

def simulated_annealing(initial_bits, initial_score, time_budget, seed=None):
    """SA with swap moves."""
    if seed is not None:
        random.seed(seed)
    
    bits = initial_bits
    current_score = initial_score
    best_bits = bits
    best_score = current_score
    
    start_time = time.time()
    temp = 10.0
    cooling = 0.99995
    
    def get_in_out(bits):
        in_set = []
        out_set = []
        for v in range(1, MAX_VAL + 1):
            if bits & (1 << v):
                in_set.append(v)
            else:
                out_set.append(v)
        return in_set, out_set
    
    while time.time() - start_time < time_budget:
        in_set, out_set = get_in_out(bits)
        if not in_set or not out_set:
            break
        
        v_remove = random.choice(in_set)
        v_add = random.choice(out_set)
        new_bits = (bits & ~(1 << v_remove)) | (1 << v_add)
        
        new_score = score_bitset(new_bits)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
            bits = new_bits
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_bits = bits
        
        temp *= cooling
        if temp < 0.01:
            break
    
    return best_bits, best_score

def generate_initializations():
    candidates = []
    
    # 1. Greedy
    bits, score = greedy_construct()
    candidates.append((bits, score))
    
    # 2. Consecutive
    bits = list_to_bits(list(range(1, 33)))
    candidates.append((bits, score_bitset(bits)))
    
    # 3. Random
    for _ in range(3):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        bits = list_to_bits(lst)
        candidates.append((bits, score_bitset(bits)))
        
    # 4. Clustered
    cluster = random.sample(range(1, 100), 16)
    spread = random.sample(range(300, 513), 16)
    bits = list_to_bits(cluster + spread)
    candidates.append((bits, score_bitset(bits)))
    
    # 5. Grid-like
    # 4 blocks of 8
    grid = []
    for i in range(4):
        base = i * 16 + 1
        grid.extend(range(base, base + 8))
    bits = list_to_bits(grid)
    candidates.append((bits, score_bitset(bits)))
    
    return candidates

def construct():
    start_time = time.time()
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)
    
    best_bits = None
    best_score = 0
    
    # Phase 1: Initialize and Polish
    candidates = generate_initializations()
    
    for bits, score in candidates:
        if time_left() < 10:
            break
        # Polish each candidate
        ls_bits, ls_score = local_search_polish(bits, score, min(5, time_left() / 5))
        if ls_score > best_score:
            best_score = ls_score
            best_bits = ls_bits
    
    if time_left() < 5:
        return bits_to_list(best_bits)
    
    # Phase 2: Simulated Annealing from best and restarts
    # We have time_left() seconds.
    # Split into rounds.
    num_rounds = 10
    sa_time_per_round = (time_left() * 0.8) / num_rounds
    polish_time_per_round = (time_left() * 0.2) / num_rounds
    
    current_best_bits = best_bits
    current_best_score = best_score
    
    for r in range(num_rounds):
        if time_left() < 2:
            break
        
        # SA from current best
        sa_bits, sa_score = simulated_annealing(current_best_bits, current_best_score, sa_time_per_round, seed=42 + r)
        
        # Polish the SA result
        ls_bits, ls_score = local_search_polish(sa_bits, sa_score, polish_time_per_round)
        
        if ls_score > current_best_score:
            current_best_score = ls_score
            current_best_bits = ls_bits
            best_score = ls_score
            best_bits = ls_bits
        else:
            # If SA didn't improve, maybe restart from a random point?
            # Or just continue.
            pass
            
    # Final Polish
    if time_left() > 1:
        final_bits, final_score = local_search_polish(best_bits, best_score, max(0, time_left() - 0.5))
        if final_score > best_score:
            best_bits = final_bits
            best_score = final_score
            
    return bits_to_list(best_bits)
