import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 50.0  # Safety margin before 60s kill

# Precompute AP conflict masks for fast bitset operations
# conflict_mask[x] is a bitmask where bit i is set if element (i+1) conflicts with x
# i.e., if adding x would create a 3-AP with element (i+1) and some other element
# Actually, let's precompute for each x, the set of elements that would form a 3-AP with x
# If both elements in a pair are present, adding x creates a 3-AP
# Better: for each x, precompute a list of "conflict pairs" - pairs (a,b) such that {x,a,b} is a 3-AP
# Then can_add checks if any such pair is fully contained in the bitset

ap_pairs = [[] for _ in range(N + 1)]

def precompute_ap_pairs():
    for x in range(1, N + 1):
        pairs = set()
        # Case 1: x is first term: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if y <= N and z <= N:
                pairs.add((y, z))
        
        # Case 2: x is middle term: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        
        # Case 3: x is last term: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                pairs.add((y, z))
        
        ap_pairs[x] = list(pairs)

precompute_ap_pairs()

# Precompute reciprocal values as floats for speed
recip_float = [0.0] * (N + 1)
for i in range(1, N + 1):
    recip_float[i] = 1.0 / i

def can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset creates a 3-AP.
    Returns True if x can be added without creating a 3-AP.
    """
    for a, b in ap_pairs[x]:
        # Check if both a and b are in the bitset
        if (bitset & (1 << (a - 1))) and (bitset & (1 << (b - 1))):
            return False
    return True

def greedy_construct(order):
    """
    Greedy construction: add elements in the given order if they don't create a 3-AP.
    Returns (bitset, score_float) where score_float is the sum of reciprocals as float.
    """
    bitset = 0
    score_float = 0.0
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score_float += recip_float[x]
    return bitset, score_float

def compute_exact_score(bitset):
    """Compute exact Fraction score from bitset."""
    score = Fraction(0, 1)
    for x in range(1, N + 1):
        if bitset & (1 << (x - 1)):
            score += Fraction(1, x)
    return score

def local_search(bitset, score_float, deadline):
    """
    Local search to improve the solution using float scores for speed.
    Returns (best_bitset, best_score_float) as (int, float).
    """
    best_bitset = bitset
    best_score_float = score_float
    current_bitset = bitset
    current_score_float = score_float
    
    # Simulated Annealing parameters
    current_temp = 0.03
    cooling_rate = 0.9998
    min_temp = 1e-12
    no_improve_count = 0
    max_no_improve = 3000
    
    # Precompute list of elements not in set for faster random selection
    def get_not_in_set(bitset):
        """Return a list of elements not in the bitset."""
        result = []
        for i in range(1, N + 1):
            if not (bitset & (1 << (i - 1))):
                result.append(i)
        return result
    
    def random_elem_in(bitset):
        """Get a random element from the bitset."""
        if bitset == 0:
            return -1
        # Build list once
        elems = []
        for i in range(1, N + 1):
            if bitset & (1 << (i - 1)):
                elems.append(i)
        if not elems:
            return -1
        return random.choice(elems)

    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset to best and increase temperature
            current_bitset = best_bitset
            current_score_float = best_score_float
            current_temp = max(current_temp * 1.5, 0.03)
            no_improve_count = 0
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an element
            x_add = random.randint(1, N)
            if not (current_bitset & (1 << (x_add - 1))):
                if can_add(current_bitset, x_add):
                    new_bitset = current_bitset | (1 << (x_add - 1))
                    delta = recip_float[x_add]
                    current_bitset = new_bitset
                    current_score_float += delta
                    if current_score_float > best_score_float:
                        best_score_float = current_score_float
                        best_bitset = current_bitset
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type < 0.75:
            # Try to remove an element
            x_rem = random_elem_in(current_bitset)
            if x_rem != -1:
                new_bitset = current_bitset & ~(1 << (x_rem - 1))
                delta = -recip_float[x_rem]
                # Accept removal with probability based on temperature
                if random.random() < math.exp(delta / current_temp):
                    current_bitset = new_bitset
                    current_score_float += delta
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        else:
            # Swap: remove one, add another
            x_rem = random_elem_in(current_bitset)
            if x_rem != -1:
                x_add = random.randint(1, N)
                if not (current_bitset & (1 << (x_add - 1))):
                    # Check if we can swap
                    trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                    if can_add(trial_bitset, x_add):
                        trial_bitset |= (1 << (x_add - 1))
                        delta = recip_float[x_add] - recip_float[x_rem]
                        
                        if delta > 0:
                            current_bitset = trial_bitset
                            current_score_float += delta
                            if current_score_float > best_score_float:
                                best_score_float = current_score_float
                                best_bitset = trial_bitset
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            if random.random() < math.exp(delta / current_temp):
                                current_bitset = trial_bitset
                                current_score_float += delta
                                no_improve_count += 1
                            else:
                                no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        current_temp *= cooling_rate
        if current_temp < min_temp:
            current_temp = min_temp
    
    return best_bitset, best_score_float

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Multiple greedy strategies
    strategies = []
    
    # Strategy 1: Standard greedy (smallest numbers first) - maximizes reciprocal sum
    strategies.append(list(range(1, N + 1)))
    
    # Strategy 2: Randomized greedy with small bias
    for trial in range(15):
        order = list(range(1, N + 1))
        # Perturb the order with some randomness
        for i in range(len(order) - 1, 0, -1):
            if random.random() < 0.3:
                j = max(0, i - random.randint(1, 5))
                order[i], order[j] = order[j], order[i]
        strategies.append(order)
    
    # Strategy 3: Greedy by reciprocal value with randomness
    for trial in range(15):
        order = []
        remaining = set(range(1, N + 1))
        while remaining:
            # Choose from the smallest 30% with some randomness
            sorted_remaining = sorted(remaining)
            k = max(1, len(sorted_remaining) * 3 // 10)
            candidates = sorted_remaining[:k]
            x = random.choice(candidates)
            order.append(x)
            remaining.remove(x)
        strategies.append(order)
    
    # Run all strategies and keep the best
    best_bitset = 0
    best_score_float = 0.0
    
    for order in strategies:
        if time.time() > deadline - 15:
            break
        bitset, score_float = greedy_construct(order)
        if score_float > best_score_float:
            best_score_float = score_float
            best_bitset = bitset
    
    # Local search on the best solution
    best_bitset, best_score_float = local_search(best_bitset, best_score_float, deadline)
    
    # Final polish: try to add any remaining numbers that can be added
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if not (best_bitset & (1 << (x - 1))):
                if can_add(best_bitset, x):
                    best_bitset |= (1 << (x - 1))
                    improved = True
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if best_bitset & (1 << (x - 1)):
            result.append(x)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
