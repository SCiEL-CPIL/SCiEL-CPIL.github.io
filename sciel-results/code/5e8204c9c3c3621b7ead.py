import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10 seconds margin for safety
    
    all_numbers = list(range(2, 41))
    
    def compute_products(s):
        prods = {1}
        for x in s:
            new_prods = set()
            for p in prods:
                new_prods.add(x * p)
            prods = new_prods
        return prods
    
    def can_add(prods, x):
        """Check if x can be added to the set with current products prods."""
        for p in prods:
            if x * p in prods:
                return False, None
        new_prods = set(prods)
        for p in prods:
            new_prods.add(x * p)
        return True, new_prods
    
    # Start with the known good 16-element set
    base_set = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    best_set = list(base_set)
    best_size = len(best_set)
    
    # Try multiple strategies to extend the set
    strategies = []
    
    # Strategy 1: Greedy in increasing order
    strategies.append(('increasing', list(range(2, 41))))
    
    # Strategy 2: Greedy in decreasing order
    strategies.append(('decreasing', list(range(40, 1, -1))))
    
    # Strategy 3: Primes first, then composites
    primes_in_range = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites_in_range = [n for n in range(2, 41) if n not in primes_in_range]
    strategies.append(('primes_first', primes_in_range + composites_in_range))
    
    # Strategy 4: Random orders
    random.seed(42)
    for i in range(20):
        order = list(range(2, 41))
        random.shuffle(order)
        strategies.append((f'random_{i}', order))
    
    for name, order in strategies:
        if time.time() > deadline:
            break
        
        # Start from base set and try to add remaining elements
        current_set = list(base_set)
        current_prods = compute_products(current_set)
        base_set_set = set(current_set)
        
        remaining = [x for x in order if x not in base_set_set]
        
        for x in remaining:
            if time.time() > deadline:
                break
            valid, new_prods = can_add(current_prods, x)
            if valid:
                current_set.append(x)
                current_prods = new_prods
                if len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(current_set)
    
    # Try a more systematic approach: start from scratch with different initial seeds
    # Use a local search / hill climbing approach
    
    def local_search(initial_set, max_iterations=1000):
        """Try to improve the set by swapping elements."""
        if time.time() > deadline:
            return initial_set
        
        current_set = list(initial_set)
        current_prods = compute_products(current_set)
        current_size = len(current_set)
        best_local = list(current_set)
        best_local_size = current_size
        
        all_set = set(range(2, 41))
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            current_set_set = set(current_set)
            
            # Try to add any element not in current set
            for x in all_set - current_set_set:
                valid, new_prods = can_add(current_prods, x)
                if valid:
                    current_set.append(x)
                    current_prods = new_prods
                    current_size += 1
                    improved = True
                    if current_size > best_local_size:
                        best_local = list(current_set)
                        best_local_size = current_size
                    break  # Restart the loop after adding one element
            
            if not improved:
                # Try swapping: remove one element, add another
                for i, elem_to_remove in enumerate(current_set):
                    if time.time() > deadline:
                        break
                    
                    # Temporarily remove elem_to_remove
                    temp_set = current_set[:i] + current_set[i+1:]
                    temp_prods = compute_products(temp_set)
                    
                    for x in all_set - set(temp_set):
                        if x == elem_to_remove:
                            continue
                        valid, new_prods = can_add(temp_prods, x)
                        if valid:
                            # Perform the swap
                            current_set = temp_set + [x]
                            current_prods = new_prods
                            improved = True
                            if len(current_set) > best_local_size:
                                best_local = list(current_set)
                                best_local_size = len(current_set)
                            break
                    if improved:
                        break
        
        return best_local
    
    # Run local search from the best set found so far
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iterations=500)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Try local search from different starting points
    if time.time() < deadline:
        # Start with just primes
        primes_only = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        improved_set = local_search(primes_only, max_iterations=300)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
    
    # Try local search from random subsets
    if time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            random_subset = random.sample(range(2, 41), 15)
            improved_set = local_search(random_subset, max_iterations=200)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
    
    # Final verification and return
    # Ensure all elements are in range 2..40 and distinct
    best_set = [x for x in best_set if 2 <= x <= 40]
    best_set = list(set(best_set))  # Remove duplicates
    best_set.sort()
    
    # Verify the set is valid
    final_prods = compute_products(best_set)
    if len(final_prods) != (1 << len(best_set)):
        # Invalid set, fall back to base set
        best_set = list(base_set)
    
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
