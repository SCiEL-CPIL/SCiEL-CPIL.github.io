import time
import random
import math

def calc_score(A):
    """
    Compute the score for a set A: max(|A+A|, |A*A|).
    """
    if not A:
        return 0
    A = list(A)
    sumset = set(a + b for a in A for b in A)
    prodset = set(a * b for a in A for b in A)
    return max(len(sumset), len(prodset))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 20
    max_val = 10000
    
    # Helper to generate smooth numbers
    def generate_smooth_numbers(limit, max_exp=10):
        smooths = set()
        for i in range(1, limit + 1):
            x = i
            # Check if i is 5-smooth (2, 3, 5)
            while x % 2 == 0: x //= 2
            while x % 3 == 0: x //= 3
            while x % 5 == 0: x //= 5
            if x == 1:
                smooths.add(i)
        return sorted(smooths)
    
    # Generate diverse initial sets
    initial_sets = []
    
    # 1. Small integers
    initial_sets.append(list(range(1, 21)))
    
    # 2. Smooth numbers (first 20)
    smooths = generate_smooth_numbers(200)
    if len(smooths) >= 20:
        initial_sets.append(smooths[:20])
    else:
        initial_sets.append(list(range(1, 21)))
        
    # 3. Geometric-like progression (powers of 2, then fill)
    geo = []
    val = 1
    while len(geo) < 20 and val <= max_val:
        geo.append(val)
        val *= 2
    # Fill remaining with small numbers not in geo
    for i in range(1, max_val):
        if len(geo) >= 20: break
        if i not in geo:
            geo.append(i)
    initial_sets.append(sorted(geo))
    
    # 4. Random set
    random.seed(42)
    random_set = random.sample(range(1, max_val), 20)
    initial_sets.append(sorted(random_set))
    
    # 5. Another random set with different seed
    random.seed(123)
    random_set2 = random.sample(range(1, 500), 20)
    initial_sets.append(sorted(random_set2))

    best_global_set = None
    best_global_score = float('inf')
    
    # Simulated Annealing parameters
    # We will run SA for each initial set, or just pick the best one and run longer?
    # Better: run SA on the best initial set found by quick evaluation, 
    # or run parallel-ish SA on a couple of good seeds.
    # Given 85s budget, we can afford a robust SA run.
    
    # First, evaluate initial sets to find the best starting point
    for s in initial_sets:
        if len(s) != 20 or len(set(s)) != 20:
            continue
        score = calc_score(s)
        if score < best_global_score:
            best_global_score = score
            best_global_set = list(s)
            
    # If we haven't found a valid set (shouldn't happen), fallback
    if best_global_set is None:
        best_global_set = list(range(1, 21))
        best_global_score = calc_score(best_global_set)
        
    current_set = list(best_global_set)
    current_score = best_global_score
    
    # Simulated Annealing Loop
    T = 1000.0
    T_min = 0.1
    alpha = 0.995 # Cooling rate
    
    # Candidate pool for swaps
    # We want to consider numbers that might improve the structure.
    # Small numbers (1-100) and smooth numbers are good candidates.
    # Also, numbers that are multiples of existing elements.
    
    # Pre-compute a static pool of "interesting" numbers
    # 1-200 and smooth numbers up to 1000
    interesting_numbers = set(range(1, 201))
    smooths_1000 = generate_smooth_numbers(1000)
    interesting_numbers.update(smooths_1000)
    # Add some random large numbers
    random.seed(999)
    for _ in range(100):
        interesting_numbers.add(random.randint(1, max_val))
        
    interesting_list = sorted(list(interesting_numbers))
    
    step_count = 0
    while time.time() < deadline and T > T_min:
        # Generate a move: swap one element in current_set with one from interesting_list
        # To avoid O(n * pool_size) every step, we can pick a random index and a random candidate
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        # Pick a new value
        # Bias towards small numbers and smooth numbers?
        # Just pick randomly from interesting_list for now, but ensure it's not in current_set
        # We might need to check if new_val is already in current_set.
        # Since n=20 is small, checking membership is O(n).
        
        # Optimization: Pre-compute set of current elements for O(1) check
        current_set_arr = set(current_set)
        
        # Try a few random candidates if the first is in set
        new_val = None
        for _ in range(10): # Try up to 10 times to find a valid new_val
            cand = random.choice(interesting_list)
            if cand not in current_set_arr:
                new_val = cand
                break
        
        if new_val is None:
            T *= alpha
            continue
            
        # Perform swap
        current_set[idx] = new_val
        
        # Calculate new score
        new_score = calc_score(current_set)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_global_score:
                best_global_score = new_score
                best_global_set = list(current_set)
        else:
            # Deterioration
            # Accept with probability exp(-delta / T)
            if delta == 0:
                current_score = new_score
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                else:
                    # Reject: revert
                    current_set[idx] = old_val
                    
        T *= alpha
        step_count += 1
        
        # Check time
        if time.time() > deadline:
            break
            
    # Final verification
    if best_global_set is None or len(best_global_set) != 20:
        return list(range(1, 21))
        
    # Ensure distinct positive integers <= 10000
    final_set = sorted(list(set(best_global_set)))
    if len(final_set) < 20:
        # This shouldn't happen if we maintained size 20, but just in case
        return list(range(1, 21))
    if any(x > 10000 for x in final_set):
        return list(range(1, 21))
        
    return final_set
