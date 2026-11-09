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

    def is_valid(s):
        prods = compute_products(s)
        return len(prods) == (1 << len(s))

    best_set = []
    best_size = 0
    
    # Known good set
    base_set = [2, 3, 4, 5, 7, 9, 11, 13, 16, 17, 19, 23, 25, 29, 31, 37]
    if is_valid(base_set):
        best_set = list(base_set)
        best_size = len(best_set)
        
    def local_search(initial_set, max_iterations=2000):
        """Local search with add, swap, and 2-swap moves."""
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
            
            # 1. Try to add
            for x in list(all_set - current_set_set):
                valid, new_prods = can_add(current_prods, x)
                if valid:
                    current_set.append(x)
                    current_prods = new_prods
                    current_size += 1
                    improved = True
                    if current_size > best_local_size:
                        best_local = list(current_set)
                        best_local_size = current_size
                    break # Restart after add
                    
            if improved:
                continue
                
            # 2. Try 1-swap
            swapped = False
            for i, elem_to_remove in enumerate(current_set):
                if time.time() > deadline:
                    break
                temp_set = current_set[:i] + current_set[i+1:]
                temp_prods = compute_products(temp_set)
                
                for x in all_set - set(temp_set):
                    if x == elem_to_remove:
                        continue
                    valid, new_prods = can_add(temp_prods, x)
                    if valid:
                        current_set = temp_set + [x]
                        current_prods = new_prods
                        swapped = True
                        if len(current_set) > best_local_size:
                            best_local = list(current_set)
                            best_local_size = len(current_set)
                        break
                if swapped:
                    break
                    
            if swapped:
                continue
                
            # 3. Try 2-swap
            # Remove 2, add 2
            # This is expensive, so only try if stuck
            if current_size < 15: # Only if small enough to be fast? No, 2-swap is O(N^2 * 2^k)
                pass # Skip 2-swap for now if too slow, or implement carefully
                
        return best_local

    # Run local search from base set
    if time.time() < deadline:
        improved_set = local_search(best_set, max_iterations=1000)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)
            
    # Run local search from random sets
    random.seed(42)
    for i in range(50):
        if time.time() > deadline:
            break
        # Start with random subset of size 10
        random_subset = random.sample(range(2, 41), 10)
        # Ensure it's valid? If not, local search might fail or take long.
        # Better to start with a valid set.
        # Let's greedily build a valid set of size 10
        valid_subset = []
        valid_prods = {1}
        for x in random_subset:
            v, np = can_add(valid_prods, x)
            if v:
                valid_subset.append(x)
                valid_prods = np
            if len(valid_subset) == 10:
                break
                
        if len(valid_subset) > 0:
            improved_set = local_search(valid_subset, max_iterations=500)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
                
    # Final verification
    best_set = [x for x in best_set if 2 <= x <= 40]
    best_set = list(set(best_set))
    best_set.sort()
    
    if is_valid(best_set):
        return best_set
    else:
        return list(base_set)

if __name__ == "__main__":
    res = construct()
    print(f"Constructed set size: {len(res)}")
    print(res)
