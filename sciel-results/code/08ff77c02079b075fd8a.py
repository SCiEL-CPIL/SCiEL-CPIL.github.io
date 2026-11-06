import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    n = 256
    
    # Precompute weights as floats for fast comparison
    weights = [0.0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = 1.0 / i
        
    # Helper to check if x can be added to a set represented by a boolean array
    # in_set is a list of bools of length n+1
    def can_add(in_set, x):
        # Iterate over all elements currently in the set
        # To make this faster, we maintain a list of elements in the set
        # But passing the list is easier for the closure
        pass # Placeholder, we will implement inside the class or closure with state

    # We will use a class-like structure or closures to manage state for speed
    # But since we need to return a list, let's stick to functions.
    
    # To optimize can_add, we need the list of elements in the set.
    # Let's define a function that takes the list of elements and the candidate x.
    def is_valid_addition(elements, x):
        # elements is a set or list of ints
        # We use a set for O(1) lookup, but iteration is O(N)
        # The check logic:
        # For each y in elements:
        #   1. 2x - y in elements?
        #   2. 2y - x in elements?
        #   3. (x+y)/2 in elements?
        
        # Optimization: Use a boolean array for membership
        # But we need to pass it.
        pass

    # Let's use a boolean array `in_set` for O(1) membership checks.
    # And a list `current_elements` for iteration.
    
    def check_add(in_set, current_elements, x):
        for y in current_elements:
            c1 = 2 * x - y
            if 1 <= c1 <= n and c1 != x and c1 != y and in_set[c1]:
                return False
            c2 = 2 * y - x
            if 1 <= c2 <= n and c2 != x and c2 != y and in_set[c2]:
                return False
            if (x + y) % 2 == 0:
                c3 = (x + y) // 2
                if 1 <= c3 <= n and c3 != x and c3 != y and in_set[c3]:
                    return False
        return True

    def get_weight(elements):
        return sum(weights[x] for x in elements)

    def get_exact_weight(elements):
        return sum(Fraction(1, x) for x in elements)

    # Initial solution: Greedy
    def greedy_init():
        S = []
        in_set = [False] * (n + 1)
        # Try adding in order of decreasing weight (increasing index? No, 1/x decreases)
        # So 1, 2, 3... is descending weight.
        for x in range(1, n + 1):
            if check_add(in_set, S, x):
                S.append(x)
                in_set[x] = True
        return S, in_set

    best_set, _ = greedy_init()
    best_score = get_exact_weight(best_set)
    best_float_score = get_weight(best_set)
    
    # Local Search / Simulated Annealing
    # We will run multiple restarts
    
    num_restarts = 10
    max_steps = 5000
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize current state
        # Option 1: Start from best_set
        # Option 2: Start from a random valid set
        
        if r == 0:
            current_set, current_in_set = greedy_init()
        else:
            # Random initialization
            # Start with a random subset and repair
            current_set = set()
            current_in_set = [False] * (n + 1)
            
            # Add random elements until no more can be added or size limit
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            for x in candidates:
                if len(current_set) > 120: break
                if check_add(current_in_set, list(current_set), x):
                    current_set.add(x)
                    current_in_set[x] = True
            
            # Ensure it's not empty
            if not current_set:
                current_set = {1}
                current_in_set[1] = True
        
        current_elements = list(current_set)
        current_score = get_weight(current_elements)
        
        # Simulated Annealing parameters
        T = 0.05
        T_min = 0.001
        cooling = 0.99
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline - 1.0:
                break
                
            # Generate a random move
            # Types: Add, Remove, Swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            accepted = False
            new_score = current_score
            
            if move_type == 'add':
                # Pick a random x not in set
                # To speed up, we can pick from a list of non-members
                # But creating that list is O(N).
                # Just pick random x in 1..n
                x = random.randint(1, n)
                if not current_in_set[x]:
                    if check_add(current_in_set, current_elements, x):
                        # Valid add
                        new_score = current_score + weights[x]
                        delta = new_score - current_score
                        if delta > 0 or random.random() < pow(2.71828, -delta / T):
                            # Accept
                            current_set.add(x)
                            current_in_set[x] = True
                            current_elements.append(x)
                            current_score = new_score
                            accepted = True
                            
            elif move_type == 'remove':
                # Pick a random y in set
                if current_elements:
                    y = random.choice(current_elements)
                    # Remove is always valid
                    new_score = current_score - weights[y]
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.71828, -delta / T):
                        # Accept
                        current_set.remove(y)
                        current_in_set[y] = False
                        current_elements.remove(y)
                        current_score = new_score
                        accepted = True
                        
            elif move_type == 'swap':
                # Pick y in set, x not in set
                if current_elements:
                    y = random.choice(current_elements)
                    x = random.randint(1, n)
                    if not current_in_set[x]:
                        # Check if x can be added to set - {y}
                        # Temporarily remove y
                        current_in_set[y] = False
                        temp_elements = [z for z in current_elements if z != y]
                        
                        if check_add(current_in_set, temp_elements, x):
                            new_score = current_score - weights[y] + weights[x]
                            delta = new_score - current_score
                            if delta > 0 or random.random() < pow(2.71828, -delta / T):
                                # Accept
                                current_set.remove(y)
                                current_set.add(x)
                                current_in_set[y] = False
                                current_in_set[x] = True
                                current_elements.remove(y)
                                current_elements.append(x)
                                current_score = new_score
                                accepted = True
                            else:
                                # Reject, restore y
                                current_in_set[y] = True
                        else:
                            # Invalid, restore y
                            current_in_set[y] = True
                            
            if accepted:
                # Update best if improved
                if current_score > best_float_score:
                    # Verify exact score?
                    # For speed, we can just store the set and calculate exact at the end
                    # But we need to compare exact scores if floats are close
                    # Given the scale, float precision should be okay for comparison
                    # unless we are extremely close.
                    # Let's calculate exact score only if float score is close to best
                    if abs(current_score - best_float_score) < 1e-5:
                        exact_cur = get_exact_weight(current_elements)
                        if exact_cur > best_score:
                            best_score = exact_cur
                            best_set = set(current_elements)
                            best_float_score = current_score
                    else:
                        if current_score > best_float_score:
                            best_set = set(current_elements)
                            best_score = get_exact_weight(best_set)
                            best_float_score = current_score
                            
            steps += 1
            T *= cooling
            if T < T_min:
                break
                
    return sorted(best_set)
