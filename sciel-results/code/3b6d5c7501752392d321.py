"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import lcm

    # Use a deadline slightly before the 90s kill to ensure we return in time
    deadline = time.time() + 85.0
    n = 128

    # Precompute lcm table for speed
    # lcm(a, b) is symmetric
    lcm_table = {}
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            val = lcm(a, b)
            lcm_table[(a, b)] = val
            if a != b:
                lcm_table[(b, a)] = val

    def get_lcm(a, b):
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return lcm_table.get((a, b), lcm(a, b))

    # Precompute blocking pair masks for each x
    # x is blocked by pair (a,b) if lcm(a,b) == lcm(a,x) == lcm(b,x)
    # We store the bitmask of the pair (a,b) for each x
    blocking_pair_masks = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        pairs = set()
        for a in range(1, n + 1):
            if a == x:
                continue
            for b in range(a + 1, n + 1):
                if b == x:
                    continue
                if get_lcm(a, b) == get_lcm(a, x) == get_lcm(b, x):
                    # Store the pair (min, max) to avoid duplicates
                    min_ab, max_ab = (a, b) if a < b else (b, a)
                    pairs.add((min_ab, max_ab))
        
        # Convert pairs to bitmasks
        for a, b in pairs:
            mask = (1 << (a - 1)) | (1 << (b - 1))
            blocking_pair_masks[x].append(mask)

    def can_add(x, mask):
        """Check if x can be added to the set represented by mask."""
        for pm in blocking_pair_masks[x]:
            if (mask & pm) == pm:
                return False
        return True

    def greedy_with_order(order, initial_mask=0):
        """Greedy construction given an order."""
        selected_mask = initial_mask
        for x in order:
            if (selected_mask >> (x - 1)) & 1:
                continue
            if can_add(x, selected_mask):
                selected_mask |= (1 << (x - 1))
        return selected_mask

    def refill(mask):
        """Greedy refill: add all elements that can be added, in natural order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not ((mask >> (x - 1)) & 1):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        changed = True
        return mask

    def mask_to_list(mask):
        """Convert bitmask to sorted list of elements."""
        return [i + 1 for i in range(n) if (mask >> i) & 1]

    # 1. Initial Greedy Attempts
    best_mask = 0
    best_size = 0
    
    # Natural order
    order1 = list(range(1, n + 1))
    m1 = greedy_with_order(order1)
    if bin(m1).count('1') > best_size:
        best_mask = m1
        best_size = bin(m1).count('1')
        
    # Reverse order
    order2 = list(range(n, 0, -1))
    m2 = greedy_with_order(order2)
    if bin(m2).count('1') > best_size:
        best_mask = m2
        best_size = bin(m2).count('1')
        
    # Degree-based order (fewest blocking pairs first)
    deg_order = sorted(range(1, n + 1), key=lambda x: len(blocking_pair_masks[x]))
    m3 = greedy_with_order(deg_order)
    if bin(m3).count('1') > best_size:
        best_mask = m3
        best_size = bin(m3).count('1')
        
    # Random orders
    rng = random.Random(42)
    for _ in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        m = greedy_with_order(order)
        size = bin(m).count('1')
        if size > best_size:
            best_mask = m
            best_size = size

    # 2. Kick-and-Refill Local Search
    current_mask = best_mask
    current_size = best_size
    
    # Refill the initial best mask to ensure it's maximal
    current_mask = refill(current_mask)
    current_size = bin(current_mask).count('1')
    if current_size > best_size:
        best_mask = current_mask
        best_size = current_size

    kick_size = 3
    # Increase kick size slightly if we are stuck? Or vary it?
    # Let's try varying kick size between 2 and 5.
    
    for trial in range(5000): # High number, will be cut by deadline
        if time.time() > deadline:
            break
        
        # Random kick
        # Select random elements from current_mask to remove
        # To do this efficiently, get list of elements in current_mask
        # But converting to list every time is O(N). N=128 is small.
        elements_in_mask = [i + 1 for i in range(n) if (current_mask >> i) & 1]
        
        if len(elements_in_mask) < kick_size:
            break
            
        # Shuffle and pick kick_size elements
        rng.shuffle(elements_in_mask)
        kick_elements = elements_in_mask[:kick_size]
        
        # Create new mask by removing kicked elements
        candidate_mask = current_mask
        for x in kick_elements:
            candidate_mask &= ~(1 << (x - 1))
            
        # Refill
        candidate_mask = refill(candidate_mask)
        
        candidate_size = bin(candidate_mask).count('1')
        
        if candidate_size > best_size:
            best_size = candidate_size
            best_mask = candidate_mask
            current_mask = candidate_mask
        elif candidate_size == best_size:
            # Accept with some probability to escape local optima
            if rng.random() < 0.2:
                current_mask = candidate_mask
        # If candidate_size < best_size, reject (keep current_mask as is for next kick? 
        # Or should we kick from best_mask? 
        # Standard ILS kicks from the current solution. If rejected, we stay at current.
        # But if we stay at current and kick again, we might loop.
        # Let's kick from best_mask if we haven't improved recently?
        # For simplicity, kick from current_mask.
        
    # Final check and return
    if best_size == 0:
        return [1]
    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END
