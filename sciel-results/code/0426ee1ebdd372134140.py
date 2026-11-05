import sys
import time
import random

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s buffer
    
    # Prescribed elements
    prescribed = {1, 2, 5, 11}
    
    # Initial set: Just the prescribed elements
    current_set = set(prescribed)
    current_sums = set()
    for i, x in enumerate(sorted(prescribed)):
        for y in sorted(prescribed)[i:]:
            current_sums.add(x + y)
            
    best_set = list(current_set)
    best_size = len(current_set)
    
    # Candidates are 1..284 excluding prescribed
    all_candidates = list(range(1, 285))
    
    def check_add(x, c_set, c_sums):
        """Check if x can be added to c_set without sum collisions."""
        if x in c_set:
            return False
        # Check 2*x
        if 2 * x in c_sums:
            return False
        # Check x + y for all y in c_set
        for y in c_set:
            if x + y in c_sums:
                return False
        return True

    def add_to_state(x, c_set, c_sums):
        """Add x to the state, updating sums."""
        c_set.add(x)
        c_sums.add(2 * x)
        for y in c_set:
            if y != x:
                c_sums.add(x + y)

    def remove_from_state(x, c_set, c_sums):
        """Remove x from the state, updating sums."""
        c_set.remove(x)
        c_sums.remove(2 * x)
        for y in c_set:
            c_sums.remove(x + y)

    # Simulated Annealing Parameters
    # We want to maximize size.
    # Energy = -size.
    # Move 1: Add valid x. Delta E = -1. Always accept.
    # Move 2: Remove non-prescribed y. Delta E = +1. Accept with prob exp(-1/T).
    # Move 3: Swap y (non-prescribed) with x (not in set). If valid, Delta E = 0. Accept with prob 1 (or exp(0/T)=1).
    #         Actually, if we swap and it's valid, size is same. We might want to accept these to wander.
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # To make it faster, we precompute or just iterate.
    # Number of iterations
    max_iters = 500000
    iters_done = 0
    
    while time.time() - start_time < timeout and iters_done < max_iters:
        iters_done += 1
        
        # 1. Try to add a random element
        if random.random() < 0.5:
            # Pick random candidate not in set
            # Efficient way: pick random from 1..284, if in set, try again (few retries)
            added = False
            for _ in range(10): # retry limit
                x = random.randint(1, 284)
                if x not in current_set:
                    if check_add(x, current_set, current_sums):
                        add_to_state(x, current_set, current_sums)
                        added = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        break
            if added:
                continue # Accept and next iteration
        
        # 2. Try to remove a random non-prescribed element
        elif random.random() < 0.3:
            # Get list of non-prescribed elements in current_set
            removable = [x for x in current_set if x not in prescribed]
            if removable:
                y = random.choice(removable)
                # Calculate probability
                # Delta E = 1 (size decreases)
                if random.random() < (2.71828 ** (-1.0 / T)):
                    remove_from_state(y, current_set, current_sums)
                    # Note: We don't update best here as size decreased
        
        # 3. Try to swap a non-prescribed element with a new one
        else:
            removable = [x for x in current_set if x not in prescribed]
            if removable:
                y = random.choice(removable)
                # Try to find an x that can replace y
                swapped = False
                for _ in range(10):
                    x = random.randint(1, 284)
                    if x in current_set or x == y:
                        continue
                    # Check if (S - {y}) U {x} is valid
                    # Temporarily remove y
                    remove_from_state(y, current_set, current_sums)
                    if check_add(x, current_set, current_sums):
                        add_to_state(x, current_set, current_sums)
                        swapped = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        break
                    else:
                        # Restore y
                        add_to_state(y, current_set, current_sums)
                if swapped:
                    continue
                # If not swapped, state is unchanged (y was restored or never removed if x invalid check failed early? 
                # Wait, my logic above removes y, checks x. If x fails, restores y. So state is consistent.)
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: Restart temperature if stuck? 
            # For now, just keep going or reset T occasionally if no improvement
            # Simple restart: if T is min, maybe reset T to 0.5 and randomize set slightly?
            # Let's add a simple perturbation if T is low
            if iters_done % 10000 == 0:
                # Randomly remove 2-3 elements and try to add back
                removable = [x for x in current_set if x not in prescribed]
                for _ in range(min(3, len(removable))):
                    y = random.choice(removable)
                    remove_from_state(y, current_set, current_sums)
                    removable.remove(y)
                # Try to add some back
                for _ in range(5):
                    x = random.randint(1, 284)
                    if x not in current_set and check_add(x, current_set, current_sums):
                        add_to_state(x, current_set, current_sums)
                T = 0.5

    return best_set
