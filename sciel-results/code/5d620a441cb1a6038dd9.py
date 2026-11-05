import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Known good solution: {66, ..., 100}, size 35
    best_set = set(range(66, 101))
    best_size = 35
    
    def is_valid(s):
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

    # Initial greedy attempts with different orders to find a potentially better starting point
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 50, -1)) + list(range(50, 0, -1))
    ]
    
    # Add a few random orders
    for _ in range(5):
        rand_order = list(range(1, 101))
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
            
        s = set()
        reachable = {0}
        
        for x in order:
            # Check if adding x creates sum 200
            if (200 - x) not in reachable:
                s.add(x)
                new_sums = {y + x for y in reachable if y + x <= 200}
                reachable |= new_sums
        
        if len(s) > best_size and is_valid(s):
            best_size = len(s)
            best_set = s

    # Local Search to improve best_set
    # We try to increase size by swapping elements.
    # Move: Remove 1, Add 2 (net +1)
    # Move: Remove 2, Add 3 (net +1)
    
    current_set = set(best_set)
    current_size = best_size
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        if iterations > 50: # Limit iterations to avoid infinite loops on bad paths
            break
            
        reachable = compute_reachable(current_set)
        
        # Try Move 1: Remove x, Add y, z
        for x in list(current_set):
            if time.time() > deadline:
                return list(best_set)
            
            # Temporarily remove x
            current_set.remove(x)
            # Recompute reachable without x
            reachable_no_x = compute_reachable(current_set)
            
            # Try to add two elements y, z
            added = []
            for y in range(1, 101):
                if y not in current_set and y != x:
                    if (200 - y) not in reachable_no_x:
                        # Tentatively add y
                        current_set.add(y)
                        reachable_y = compute_reachable(current_set)
                        
                        for z in range(1, 101):
                            if z not in current_set and z != x and z != y:
                                if (200 - z) not in reachable_y:
                                    # Valid swap
                                    current_set.add(z)
                                    added = [y, z]
                                    break
                        if added:
                            break
                        else:
                            # Undo y
                            current_set.remove(y)
                            
            if added:
                # Update best
                new_size = len(current_set)
                if new_size > best_size:
                    best_size = new_size
                    best_set = set(current_set)
                    improved = True
                    break # Restart search with new best
            else:
                # Restore x
                current_set.add(x)
                
    # Final verification
    if is_valid(best_set):
        return list(best_set)
    else:
        return list(range(66, 101))
