import time
import random
import math
from fractions import Fraction

N = 256
TIME_LIMIT = 50.0  # Safety margin before 60s kill

# Precompute AP pairs for each x
# ap_pairs[x] is a list of tuples (a, b) such that {x, a, b} is a 3-AP
# and a, b != x
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
    Returns (bitset, score) where score is the exact sum of reciprocals as a Fraction.
    """
    bitset = 0
    score = Fraction(0, 1)
    for x in order:
        if can_add(bitset, x):
            bitset |= (1 << (x - 1))
            score += Fraction(1, x)
    return bitset, score

def local_search(bitset, score, deadline):
    """
    Local search to improve the solution.
    Returns (best_bitset, best_score) as (int, Fraction).
    """
    best_bitset = bitset
    best_score = score
    current_bitset = bitset
    current_score = score
    
    # Precompute reciprocals as Fractions
    recip = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = Fraction(1, i)
    
    # Precompute list of elements in bitset for faster access? 
    # Bitset iteration is fast enough for N=256.
    
    # Simulated Annealing parameters
    current_temp = 0.05
    cooling_rate = 0.9999
    min_temp = 1e-12
    no_improve_count = 0
    max_no_improve = 2000
    
    # Helper to get a random element from a bitset
    def random_elem(bitset):
        if bitset == 0:
            return -1
        # Pick a random position and check if set
        # Since N=256, we can just iterate or pick random 1..256 and retry
        # Iterating is safer
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
            current_score = best_score
            current_temp = max(current_temp * 2, 0.05)
            no_improve_count = 0
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add an element
            # Find elements not in set
            # We can generate a random x and check if it's in set
            x_add = random.randint(1, N)
            if not (current_bitset & (1 << (x_add - 1))):
                if can_add(current_bitset, x_add):
                    new_bitset = current_bitset | (1 << (x_add - 1))
                    delta = recip[x_add]
                    if delta > 0:
                        current_bitset = new_bitset
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                    else:
                        # Accept worse moves with probability
                        if random.random() < math.exp(float(delta) / current_temp):
                            current_bitset = new_bitset
                            current_score += delta
                            no_improve_count += 1
                        else:
                            no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type < 0.7:
            # Try to remove an element
            x_rem = random_elem(current_bitset)
            if x_rem != -1:
                new_bitset = current_bitset & ~(1 << (x_rem - 1))
                delta = -recip[x_rem]
                # Always accept removal if it leads to a better overall solution after re-adding
                # Since delta is negative, we use SA acceptance
                if random.random() < math.exp(float(delta) / current_temp):
                    current_bitset = new_bitset
                    current_score += delta
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        else:
            # Swap: remove one, add another
            x_rem = random_elem(current_bitset)
            if x_rem != -1:
                x_add = random.randint(1, N)
                if not (current_bitset & (1 << (x_add - 1))):
                    # Check if we can swap
                    # Remove x_rem, add x_add
                    trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                    if can_add(trial_bitset, x_add):
                        trial_bitset |= (1 << (x_add - 1))
                        delta = recip[x_add] - recip[x_rem]
                        
                        if delta > 0:
                            current_bitset = trial_bitset
                            current_score += delta
                            if current_score > best_score:
                                best_score = current_score
                                best_bitset = trial_bitset
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                        else:
                            if random.random() < math.exp(float(delta) / current_temp):
                                current_bitset = trial_bitset
                                current_score += delta
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
    
    return best_bitset, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Reciprocals for numbers 1 to 256
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
    
    # Multiple greedy strategies
    strategies = []
    
    # Strategy 1: Standard greedy (smallest numbers first)
    strategies.append(list(range(1, N + 1)))
    
    # Strategy 2: Randomized greedy with small bias
    for trial in range(20):
        order = list(range(1, N + 1))
        # Perturb the order with some randomness
        for i in range(len(order) - 1, 0, -1):
            # With some probability, swap with a nearby element
            if random.random() < 0.3:
                j = max(0, i - random.randint(1, 5))
                order[i], order[j] = order[j], order[i]
        strategies.append(order)
    
    # Strategy 3: Reverse greedy (largest first) - sometimes helps
    strategies.append(list(range(N, 0, -1)))
    
    # Strategy 4: Greedy by reciprocal density (small numbers but skip some)
    for trial in range(20):
        order = []
        # Prioritize small numbers but with some randomness
        remaining = set(range(1, N + 1))
        while remaining:
            # Choose from the smallest 20% with some randomness
            sorted_remaining = sorted(remaining)
            k = max(1, len(sorted_remaining) // 5)
            candidates = sorted_remaining[:k]
            x = random.choice(candidates)
            order.append(x)
            remaining.remove(x)
        strategies.append(order)
    
    # Run all strategies and keep the best
    best_bitset = 0
    best_score = Fraction(0, 1)
    
    for order in strategies:
        if time.time() > deadline - 10:
            break
        bitset, score = greedy_construct(order)
        if score > best_score:
            best_score = score
            best_bitset = bitset
    
    # Local search on the best solution
    best_bitset, best_score = local_search(best_bitset, best_score, deadline)
    
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
