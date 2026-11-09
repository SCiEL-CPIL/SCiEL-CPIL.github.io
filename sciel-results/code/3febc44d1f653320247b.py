import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 45.0

def can_add_fast(bitset, x):
    """
    Check if adding x to the set creates a 3-AP.
    Returns True if x can be added.
    """
    # Case 1: x is first term: x, x+d, x+2d
    d_max = (N - x) // 2
    for d in range(1, d_max + 1):
        if (bitset & (1 << (x + d - 1))) and (bitset & (1 << (x + 2*d - 1))):
            return False
    
    # Case 2: x is middle term: x-d, x, x+d
    d_max = min(x - 1, N - x)
    for d in range(1, d_max + 1):
        if (bitset & (1 << (x - d - 1))) and (bitset & (1 << (x + d - 1))):
            return False
    
    # Case 3: x is last term: x-2d, x-d, x
    d_max = x // 2
    for d in range(1, d_max + 1):
        m = x - 2*d
        if m >= 1 and (bitset & (1 << (m - 1))) and (bitset & (1 << (x - d - 1))):
            return False
    
    return True

def greedy_construct(order):
    """Greedy construction with given order."""
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add_fast(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

def local_search(bitset, score, deadline):
    """Local search to improve the solution."""
    best_bitset = bitset
    best_score = score
    current_bitset = bitset
    current_score = score
    
    recip = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = Fraction(1, i)
    
    # Phase 1: Add all possible elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (current_bitset & (1 << (x - 1))):
                if can_add_fast(current_bitset, x):
                    current_bitset |= (1 << (x - 1))
                    current_score += recip[x]
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset
    
    # Phase 2: Simulated annealing with swaps
    current_temp = 0.05
    cooling_rate = 0.9998
    min_temp = 1e-12
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            current_bitset = best_bitset
            current_score = best_score
            current_temp = max(current_temp * 1.5, 0.03)
            no_improve = 0
        
        # Try a swap: remove one element, add another
        in_set = [i for i in range(1, N + 1) if current_bitset & (1 << (i - 1))]
        not_in_set = [i for i in range(1, N + 1) if not (current_bitset & (1 << (i - 1)))]
        
        if not in_set or not not_in_set:
            break
        
        x_rem = random.choice(in_set)
        x_add = random.choice(not_in_set)
        
        trial = current_bitset & ~(1 << (x_rem - 1))
        if can_add_fast(trial, x_add):
            trial |= (1 << (x_add - 1))
            delta = recip[x_add] - recip[x_rem]
            
            if delta > 0:
                current_bitset = trial
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_bitset = current_bitset
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                if random.random() < math.exp(float(delta) / current_temp):
                    current_bitset = trial
                    current_score += delta
                    no_improve += 1
                else:
                    no_improve += 1
        else:
            no_improve += 1
        
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp
    
    return best_bitset, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Multiple greedy strategies
    best_bitset = 0
    best_score = Fraction(0, 1)
    
    # Strategy 1: Smallest first
    order1 = list(range(1, N + 1))
    b, s = greedy_construct(order1)
    if s > best_score:
        best_score = s
        best_bitset = b
    
    # Strategy 2: Randomized small-first
    for trial in range(30):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Bias towards small numbers
        order.sort(key=lambda x: x + random.random() * 20)
        b, s = greedy_construct(order)
        if s > best_score:
            best_score = s
            best_bitset = b
    
    # Strategy 3: Weighted random
    for trial in range(20):
        if time.time() > deadline - 15:
            break
        order = []
        remaining = list(range(1, N + 1))
        while remaining:
            # Weight by reciprocal value
            weights = [1.0 / x for x in remaining]
            total = sum(weights)
            r = random.random() * total
            cum = 0
            for i, x in enumerate(remaining):
                cum += weights[i]
                if r <= cum:
                    order.append(x)
                    remaining.pop(i)
                    break
        b, s = greedy_construct(order)
        if s > best_score:
            best_score = s
            best_bitset = b
    
    # Local search
    best_bitset, best_score = local_search(best_bitset, best_score, deadline)
    
    # Final polish
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add_fast(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    improved = True
    
    result = [x for x in range(1, N + 1) if best_bitset & (1 << (x - 1))]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
