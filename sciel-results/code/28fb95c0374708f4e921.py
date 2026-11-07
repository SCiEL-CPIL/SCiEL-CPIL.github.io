"""Bitmask-based simulated annealing for divisibility-incomparable subset sums."""
import time
import random

# EVOLVE-BLOCK-START

def _precompute():
    """Precompute divisibility masks and subset sum structures."""
    # Max possible subset sum: 30 elements from 1..256, max sum = 256+255+...+227 = 30*241.5 ~ 7245
    # But we need to be safe. Max sum of 30 distinct from 1..256 is 256+255+...+227.
    # 256+255+...+227 = sum of 30 consecutive ending at 256.
    # = 30*(256+227)/2 = 30*241.5 = 7245
    # Let's use MAX_SUM = 8000 to be safe.
    MAX_SUM = 8000
    
    # divisibility_mask[s] = bitmask with bits set for all multiples of s (excluding s) in 1..MAX_SUM
    # We represent the set of subset sums as a Python int bitmask, where bit i is set if sum i is present.
    # divisibility_mask[s] has bits set at positions 2s, 3s, 4s, ... up to MAX_SUM
    divisibility_mask = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM + 1):
        mask = 0
        m = 2 * s
        while m <= MAX_SUM:
            mask |= (1 << m)
            m += s
        divisibility_mask[s] = mask
    
    return MAX_SUM, divisibility_mask

def _check_divisibility(sums_mask, divisibility_mask):
    """Check if no two distinct sums in sums_mask divide each other.
    sums_mask is a Python int where bit s is set if sum s is present.
    Returns True if valid (no divisibility conflicts)."""
    # For each sum s that is present, check if any multiple of s is also present.
    # We need to iterate over set bits in sums_mask.
    mask = sums_mask
    while mask:
        # Get lowest set bit
        low_bit = mask & (-mask)
        s = low_bit.bit_length() - 1
        # Check if any multiple of s is in sums_mask
        if sums_mask & divisibility_mask[s]:
            return False
        mask ^= low_bit
    return True

def _count_set_bits(mask):
    """Count set bits in a Python int."""
    count = 0
    while mask:
        mask &= mask - 1
        count += 1
    return count

def _get_set_bits(mask):
    """Yield indices of set bits in a Python int."""
    while mask:
        low_bit = mask & (-mask)
        yield low_bit.bit_length() - 1
        mask ^= low_bit

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    MAX_SUM, divisibility_mask = _precompute()
    
    def compute_sums_mask(elements):
        """Compute the bitmask of all nonempty subset sums for the given elements.
        Returns (mask, is_valid_unique) where is_valid_unique is False if any duplicate sums exist."""
        n = len(elements)
        if n == 0:
            return 0, True
        # Use iterative construction
        # sums_mask starts with 0 (empty subset)
        # For each element x, new_sums = old_sums | (old_sums << x) but we need to track if collisions occur
        # Actually, let's use a different approach: track the set of sums as a bitmask.
        # Start with mask = 1 (bit 0 set, representing empty sum)
        # For each element x: new_mask = mask | (mask << x)
        # But we need to detect if (mask & (mask << x)) != 0, which would indicate a collision.
        mask = 1  # bit 0 set
        for x in elements:
            shifted = mask << x
            # Check for collision: if any bit in mask is also in shifted, that means a sum was reached twice
            if mask & shifted:
                return 0, False
            mask |= shifted
        # Remove the empty sum (bit 0)
        mask &= ~1
        return mask, True
    
    def try_add(elements, current_mask, x):
        """Try to add x to elements. Returns (new_mask, new_valid) or None if invalid."""
        # New sums = old_sums ∪ (old_sums + x) ∪ {x}
        # In bitmask terms: shifted = current_mask << x
        # New mask = current_mask | shifted | (1 << x)
        # But we need to check for collisions:
        # 1. x should not be in current_mask
        # 2. (current_mask << x) should not intersect current_mask
        # 3. (current_mask << x) should not intersect (1 << x), i.e., x should not be in current_mask (same as 1)
        # 4. Also need to check divisibility
        
        if current_mask & (1 << x):
            return None  # x is already a subset sum
        
        shifted = current_mask << x
        
        # Check collision between shifted and current_mask
        if shifted & current_mask:
            return None
        
        new_mask = current_mask | shifted | (1 << x)
        
        # Check divisibility
        if not _check_divisibility(new_mask, divisibility_mask):
            return None
        
        return new_mask
    
    def greedy_construct(order, max_size=30):
        """Build a valid set by adding elements in the given order."""
        current = []
        current_mask = 0
        
        for x in order:
            if x in current:
                continue
            result = try_add(current, current_mask, x)
            if result is not None:
                current.append(x)
                current_mask = result
                if len(current) >= max_size:
                    break
        
        return current, current_mask
    
    # Phase 1: Greedy construction with multiple orderings
    candidates_all = list(range(1, 257))
    rng = random.Random(42)
    
    # Precompute divisor counts for ordering
    div_count = [0] * 257
    for i in range(1, 257):
        for j in range(1, i + 1):
            if i % j == 0:
                div_count[i] += 1
    
    orderings = []
    # Descending
    orderings.append(sorted(candidates_all, reverse=True))
    # Ascending
    orderings.append(sorted(candidates_all))
    # By divisor count (fewer first)
    orderings.append(sorted(candidates_all, key=lambda x: (div_count[x], x)))
    # By divisor count (more first)
    orderings.append(sorted(candidates_all, key=lambda x: (-div_count[x], x)))
    # Random orderings
    for _ in range(50):
        shuffled = candidates_all[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_set = []
    best_mask = 0
    best_size = 0
    
    for order in orderings:
        if time.time() > deadline:
            break
        elements, mask = greedy_construct(order)
        if len(elements) > best_size:
            best_size = len(elements)
            best_set = elements
            best_mask = mask
    
    # Phase 2: Simulated annealing
    if time.time() < deadline and best_size > 0:
        current_set = list(best_set)
        current_mask = best_mask
        current_size = best_size
        
        # Temperature schedule
        initial_temp = 10.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        temp = initial_temp
        iteration = 0
        max_iterations = 500000
        
        while time.time() < deadline and temp > min_temp and iteration < max_iterations:
            iteration += 1
            
            # Choose a move: insert, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.4 and current_size < 30:
                # Insert move
                candidates = [x for x in candidates_all if x not in current_set]
                if not candidates:
                    continue
                x = rng.choice(candidates)
                result = try_add(current_set, current_mask, x)
                if result is None:
                    continue
                new_set = current_set + [x]
                new_mask = result
                new_size = current_size + 1
                
                # Accept if better or with probability based on temperature
                if new_size > current_size:
                    current_set = new_set
                    current_mask = new_mask
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_set = list(current_set)
                        best_mask = current_mask
                # Don't accept worse moves (size can only stay same or increase with insert)
                
            elif move_type < 0.6 and current_size > 0:
                # Remove move
                i = rng.randrange(current_size)
                removed = current_set[i]
                new_set = current_set[:i] + current_set[i+1:]
                new_mask, valid = compute_sums_mask(new_set)
                if not valid:
                    continue
                new_size = current_size - 1
                
                # Accept removal only with some probability (to escape local optima)
                if rng.random() < 0.3:
                    current_set = new_set
                    current_mask = new_mask
                    current_size = new_size
                    
            else:
                # Swap move
                if current_size == 0:
                    continue
                i = rng.randrange(current_size)
                old_x = current_set[i]
                candidates = [x for x in candidates_all if x not in current_set]
                if not candidates:
                    continue
                new_x = rng.choice(candidates)
                
                # Remove old_x
                remaining = current_set[:i] + current_set[i+1:]
                remaining_mask, valid = compute_sums_mask(remaining)
                if not valid:
                    continue
                
                # Add new_x
                result = try_add(remaining, remaining_mask, new_x)
                if result is None:
                    continue
                
                new_set = remaining + [new_x]
                new_mask = result
                new_size = current_size
                
                # Accept swap with some probability
                if rng.random() < 0.5:
                    current_set = new_set
                    current_mask = new_mask
                    current_size = new_size
            
            temp *= cooling_rate
    
    # Phase 3: Final local search - try to add more elements
    if time.time() < deadline and best_size < 30:
        remaining = [x for x in candidates_all if x not in best_set]
        rng.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            result = try_add(best_set, best_mask, x)
            if result is not None:
                best_set.append(x)
                best_mask = result
                best_size += 1
    
    # Final verification
    if best_set:
        final_mask, valid = compute_sums_mask(best_set)
        if not valid or not _check_divisibility(final_mask, divisibility_mask):
            # Fallback
            best_set = [31, 32, 34]
    
    return list(best_set)

# EVOLVE-BLOCK-END
