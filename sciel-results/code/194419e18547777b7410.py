"""Rewritten program: bitset-accelerated greedy + simulated annealing for 32-element 3-AP difference maximization."""
import time
import random
import numpy as np

# Constants
N = 32          # number of elements
MAX_VAL = 512   # max value
MAX_D = 255     # max possible difference (512-1)/2 = 255.5 -> 255
TIME_LIMIT = 115  # safety margin before 120s limit

def score_bitset(bits):
    """
    Given a bitset (int) representing the set, compute the number of distinct d>0
    such that there exists a with a, a+d, a+2d all in the set.
    
    For each d, we check if (bits & (bits >> d) & (bits >> (2*d))) != 0.
    This is fast because it's just bit shifts and ANDs.
    """
    count = 0
    for d in range(1, MAX_D + 1):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count

def bits_to_list(bits):
    """Convert bitset to list of integers (1-indexed positions)."""
    result = []
    for i in range(1, MAX_VAL + 1):
        if bits & (1 << i):
            result.append(i)
    return result

def list_to_bits(lst):
    """Convert list of integers to bitset."""
    bits = 0
    for x in lst:
        bits |= (1 << x)
    return bits

def greedy_construct():
    """
    Greedy construction: build the set one element at a time,
    choosing the element that maximizes the marginal gain in distinct differences.
    """
    bits = 0
    current_score = 0
    used = set()
    
    for _ in range(N):
        best_gain = -1
        best_val = None
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
        current_score = score_bitset(bits)
    
    return bits, current_score

def simulated_annealing(initial_bits, initial_score, time_budget):
    """
    Simulated annealing on the bitset representation.
    Moves: add a random element (if size < N), remove a random element (if size > N),
    or swap a random element with another.
    """
    bits = initial_bits
    current_score = initial_score
    best_bits = bits
    best_score = current_score
    
    # Count bits
    def popcount(x):
        return bin(x).count('1')
    
    current_size = popcount(bits)
    
    start_time = time.time()
    temp = 10.0
    cooling = 0.99995
    
    while time.time() - start_time < time_budget:
        # Generate a neighbor
        move_type = random.random()
        if move_type < 0.3 and current_size < N:
            # Add a random element
            candidates = [v for v in range(1, MAX_VAL + 1) if not (bits & (1 << v))]
            if not candidates:
                continue
            v = random.choice(candidates)
            new_bits = bits | (1 << v)
        elif move_type < 0.5 and current_size > N:
            # Remove a random element
            candidates = [v for v in range(1, MAX_VAL + 1) if (bits & (1 << v))]
            if not candidates:
                continue
            v = random.choice(candidates)
            new_bits = bits & ~(1 << v)
        else:
            # Swap: remove one, add another
            in_set = [v for v in range(1, MAX_VAL + 1) if (bits & (1 << v))]
            out_set = [v for v in range(1, MAX_VAL + 1) if not (bits & (1 << v))]
            if not in_set or not out_set:
                continue
            v_remove = random.choice(in_set)
            v_add = random.choice(out_set)
            new_bits = (bits & ~(1 << v_remove)) | (1 << v_add)
        
        new_score = score_bitset(new_bits)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / max(temp, 1e-10)):
            bits = new_bits
            current_score = new_score
            current_size = popcount(bits)
            
            if current_score > best_score:
                best_score = current_score
                best_bits = bits
        
        temp *= cooling
        if temp < 0.01:
            break
    
    return best_bits, best_score

def local_search_polish(bits, score, time_budget):
    """
    Local search: try all single-element swaps and keep the best improvement.
    Repeat until no improvement or time runs out.
    """
    best_bits = bits
    best_score = score
    start_time = time.time()
    
    def popcount(x):
        return bin(x).count('1')
    
    while time.time() - start_time < time_budget:
        improved = False
        in_set = [v for v in range(1, MAX_VAL + 1) if (best_bits & (1 << v))]
        out_set = [v for v in range(1, MAX_VAL + 1) if not (best_bits & (1 << v))]
        
        if not in_set or not out_set:
            break
        
        best_swap_gain = 0
        best_swap = None
        
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
            improved = True
        
        if not improved:
            break
    
    return best_bits, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Greedy construction
    bits, score = greedy_construct()
    best_bits = bits
    best_score = score
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT:
        return bits_to_list(best_bits)
    
    # Phase 2: Multiple rounds of SA + local search polish
    remaining = TIME_LIMIT - elapsed
    num_rounds = 5
    sa_budget = remaining * 0.6 / num_rounds
    polish_budget = remaining * 0.4 / num_rounds
    
    for r in range(num_rounds):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # SA from current best
        sa_bits, sa_score = simulated_annealing(best_bits, best_score, sa_budget)
        if sa_score > best_score:
            best_bits = sa_bits
            best_score = sa_score
        
        # Local search polish
        ls_bits, ls_score = local_search_polish(best_bits, best_score, polish_budget)
        if ls_score > best_score:
            best_bits = ls_bits
            best_score = ls_score
    
    # Final check
    if time.time() - start_time > TIME_LIMIT - 2:
        return bits_to_list(best_bits)
    
    # One more polish if time allows
    final_bits, final_score = local_search_polish(best_bits, best_score, max(0, TIME_LIMIT - (time.time() - start_time) - 1))
    if final_score > best_score:
        best_bits = final_bits
        best_score = final_score
    
    return bits_to_list(best_bits)
