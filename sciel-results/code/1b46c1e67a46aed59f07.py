import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Known good solution: {66, ..., 100}, size 35
    best_set = set(range(66, 101))
    best_size = 35
    
    def is_valid(s):
        """Check if no subset of s sums to 200."""
        sums = {0}
        for x in s:
            new = {y + x for y in sums if y + x <= 200}
            if 200 in new:
                return False
            sums |= new
        return True
    
    def compute_reachable(s):
        """Compute all subset sums up to 200 for set s."""
        reachable = {0}
        for x in s:
            new_sums = {y + x for y in reachable if y + x <= 200}
            reachable |= new_sums
        return reachable
    
    # Strategy 1: Conflict graph approach
    # Build conflict graph where vertices are 1-100
    # Two vertices conflict if they are part of a subset summing to 200
    # Actually, we need to be more careful: a set S is valid if no subset of S sums to 200
    # This is equivalent to: for every subset T of S, sum(T) != 200
    
    # Alternative approach: Greedy with conflict detection
    # Start with all elements, remove those that cause conflicts
    
    # Let's try a different approach: 
    # 1. Start with empty set
    # 2. Add elements in an order that minimizes conflicts
    # 3. Use the insight that larger numbers have fewer conflicts with small numbers
    
    # Actually, let's use the approach from the current program but enhanced:
    # Start with {49..100}, remove bad elements, then add small elements
    
    # Enhanced greedy: prioritize elements with fewest conflicts
    def count_conflicts(x, current_set):
        """Count how many elements in current_set form a subset summing to 200 with x."""
        if 200 - x in current_set:
            return 1  # Direct pair conflict
        # Check for triplets: need y, z in current_set such that x + y + z = 200
        # i.e., y + z = 200 - x
        target = 200 - x
        count = 0
        for y in current_set:
            z = target - y
            if z in current_set and z != y and z > y:  # Avoid double counting
                count += 1
        return count
    
    # Strategy: Build set greedily, prioritizing elements with fewest conflicts
    # Try multiple orders and keep the best
    
    orders = [
        list(range(100, 0, -1)),  # Large to small
        list(range(1, 101)),      # Small to large
        list(range(50, 0, -1)) + list(range(100, 50, -1)),  # Mid-large to small, then large
        list(range(100, 50, -1)) + list(range(50, 0, -1)),  # Large to mid-large, then small
    ]
    
    # Add random orders
    for _ in range(20):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        s = set()
        reachable = {0}
        
        for x in order:
            # Check if adding x is valid
            if (200 - x) not in reachable:
                s.add(x)
                new_sums = {y + x for y in reachable if y + x <= 200}
                reachable |= new_sums
        
        if len(s) > best_size:
            if is_valid(s):
                best_size = len(s)
                best_set = s
    
    # Local search: try to improve by swapping elements
    def try_local_search(initial_set, max_iterations=500):
        """Try to improve the set via local search moves."""
        s = set(initial_set)
        reachable = compute_reachable(s)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iteration += 1
            
            # Move 1: Add a single element
            for x in range(1, 101):
                if x not in s:
                    if (200 - x) not in reachable:
                        s.add(x)
                        new_sums = {y + x for y in reachable if y + x <= 200}
                        reachable |= new_sums
                        improved = True
                        break
            
            if improved:
                continue
            
            # Move 2: Swap one out, two in
            for x_out in list(s):
                if time.time() > deadline:
                    return s, compute_reachable(s)
                
                # Temporarily remove x_out
                s.remove(x_out)
                # Recompute reachable without x_out
                reachable_no_x = compute_reachable(s)
                
                # Try to add two elements
                added = []
                for x_in1 in range(1, 101):
                    if x_in1 not in s and x_in1 != x_out:
                        if (200 - x_in1) not in reachable_no_x:
                            s.add(x_in1)
                            added.append(x_in1)
                            new_sums1 = {y + x_in1 for y in reachable_no_x if y + x_in1 <= 200}
                            reachable_no_x |= new_sums1
                            
                            for x_in2 in range(1, 101):
                                if x_in2 not in s and x_in2 != x_out and x_in2 != x_in1:
                                    if (200 - x_in2) not in reachable_no_x:
                                        s.add(x_in2)
                                        added.append(x_in2)
                                        new_sums2 = {y + x_in2 for y in reachable_no_x if y + x_in2 <= 200}
                                        reachable_no_x |= new_sums2
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
                
                if not improved:
                    # Revert
                    s.add(x_out)
                    reachable = compute_reachable(s)
                    added = []
                
                if improved:
                    reachable = compute_reachable(s)
                    break
        
        return s, compute_reachable(s)
    
    # Apply local search to best set
    improved_set, improved_reachable = try_local_search(best_set)
    if len(improved_set) > best_size:
        if is_valid(improved_set):
            best_size = len(improved_set)
            best_set = improved_set
    
    # Final verification
    if is_valid(best_set):
        return list(best_set)
    else:
        # Fallback to known good solution
        return list(range(66, 101))
