import random
import time

def construct():
    n = 512
    
    # Step 1: Base-3 digit-based construction
    # Select numbers in [1, n] whose base-3 representation contains no digit '2'
    mask = 0
    for x in range(1, n + 1):
        num = x
        valid = True
        while num > 0:
            if num % 3 == 2:
                valid = False
                break
            num //= 3
        if valid:
            mask |= (1 << x)
            
    def can_add(x, m):
        # Case 1: x is 1st term
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if ((m >> (x + d)) & 1) and ((m >> (x + 2*d)) & 1) and ((m >> (x + 3*d)) & 1):
                return False
        # Case 2: x is 2nd term
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if ((m >> (x - d)) & 1) and ((m >> (x + d)) & 1) and ((m >> (x + 2*d)) & 1):
                return False
        # Case 3: x is 3rd term
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if ((m >> (x - 2*d)) & 1) and ((m >> (x - d)) & 1) and ((m >> (x + d)) & 1):
                return False
        # Case 4: x is 4th term
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if ((m >> (x - 3*d)) & 1) and ((m >> (x - 2*d)) & 1) and ((m >> (x - d)) & 1):
                return False
        return True

    def get_size(m):
        return bin(m).count('1')

    # Greedy Add
    def greedy_add(m, order=None):
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not ((m >> x) & 1):
                if can_add(x, m):
                    m |= (1 << x)
        return m

    # Initial Greedy Pass
    mask = greedy_add(mask)
    best_mask = mask
    best_size = get_size(mask)
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try removing each element and re-greedy-ing
        # To speed up, we can try a subset or randomize
        # Let's try all elements in the current set
        current_elements = [x for x in range(1, n + 1) if (mask >> x) & 1]
        
        for y in current_elements:
            if time.time() - start_time > time_limit:
                break
                
            # Remove y
            trial_mask = mask & ~(1 << y)
            
            # Greedy add with random order to escape local minima
            # Using a random permutation might be expensive to generate, 
            # but n=512 is small.
            # Alternatively, just use 1..n order first. 
            # Randomization helps. Let's use a fixed random seed for reproducibility? 
            # No, just use random.
            order = list(range(1, n + 1))
            random.shuffle(order)
            
            trial_mask = greedy_add(trial_mask, order)
            
            new_size = get_size(trial_mask)
            
            if new_size > best_size:
                best_size = new_size
                best_mask = trial_mask
                mask = trial_mask
                improved = True
                # Update current_elements? It's costly. 
                # We can just break and restart the loop over elements 
                # to ensure we are working with the new set.
                break
                
        if not improved:
            # If no improvement found by removing one element, 
            # maybe try a random kick?
            # Or just stop?
            # Let's try a few random kicks (remove 2 elements) if time permits?
            # For now, if no improvement in a full pass, we might be stuck.
            # Let's try randomizing the order of the greedy add in the initial step 
            # and restart?
            
            # Simpler: If no improvement, break.
            # But we might have missed something due to the greedy order.
            # Let's do a few random restarts of the whole greedy process 
            # from the base set if we get stuck?
            # No, the base set is small.
            
            # Let's try "kick": remove a random element, add random element?
            # No, that decreases size.
            
            # Let's just break. The "remove one, greedy add" is quite powerful.
            break

    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return sorted(result)
