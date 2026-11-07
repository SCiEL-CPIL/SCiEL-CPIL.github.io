# EVOLVE-BLOCK-START
import random
import math

def construct():
    n = 12
    m = 40
    
    # Precompute all subsets as bitmasks
    all_masks = list(range(1 << n))
    
    # Precompute bit counts for size
    # Python 3.10+ has int.bit_count()
    # For compatibility, we can use bin(x).count('1') or a lookup table
    # Lookup table for 0..4095 is small and fast
    bit_counts = [0] * (1 << n)
    for i in range(1 << n):
        bit_counts[i] = bin(i).count('1')
        
    def is_comparable(a, b):
        # Strict subset check using bitmasks
        # a is strict subset of b if (a & b) == a and a != b
        # b is strict subset of a if (b & a) == b and a != b
        if a == b:
            return False
        if (a & b) == a:
            return True
        if (b & a) == b:
            return True
        return False

    def get_score(sets):
        """Calculate total comparable pairs."""
        score = 0
        for i in range(m):
            for j in range(i + 1, m):
                if is_comparable(sets[i], sets[j]):
                    score += 1
        return score

    def get_gain_loss(sets, idx_out, mask_in):
        """
        Calculate delta score if we replace sets[idx_out] with mask_in.
        mask_in must not be in sets (or if it is, it's a swap with itself which is 0, but we ensure distinctness).
        """
        gain = 0
        loss = 0
        mask_out = sets[idx_out]
        for i in range(m):
            if i == idx_out:
                continue
            other = sets[i]
            if is_comparable(mask_in, other):
                gain += 1
            if is_comparable(mask_out, other):
                loss += 1
        return gain - loss

    # Initialization: Greedy with Randomness
    # Start with empty set, add sets one by one
    # To avoid bad local minima, try a few random starts or a structured start.
    
    # Structured Start: Take all sets of size 0, 1, and some of size 2
    # This is a good baseline.
    init_sets = []
    used_masks = set()
    
    # Add size 0
    init_sets.append(0)
    used_masks.add(0)
    
    # Add size 1
    for mask in all_masks:
        if bit_counts[mask] == 1:
            init_sets.append(mask)
            used_masks.add(mask)
            
    # Add size 2 until we have m sets
    random.shuffle(all_masks)
    for mask in all_masks:
        if len(init_sets) >= m:
            break
        if bit_counts[mask] == 2 and mask not in used_masks:
            init_sets.append(mask)
            used_masks.add(mask)
            
    # If we didn't reach m (unlikely), fill randomly
    if len(init_sets) < m:
        for mask in all_masks:
            if len(init_sets) >= m:
                break
            if mask not in used_masks:
                init_sets.append(mask)
                used_masks.add(mask)
                
    current_sets = init_sets
    current_score = get_score(current_sets)
    best_sets = current_sets[:]
    best_score = current_score
    
    # Simulated Annealing
    # Parameters
    start_temp = 10.0
    end_temp = 0.01
    cooling = 0.999
    max_iterations = 50000 # Adjust based on time limit
    
    # Precompute indices for fast access?
    # We need to pick a random index to remove and a random mask to add.
    
    # To speed up, we can keep a list of available masks?
    # But checking if a mask is in current_sets is O(M).
    # M is small (40).
    
    for it in range(max_iterations):
        temp = start_temp * (cooling ** it)
        
        # Pick a random index to swap out
        idx_out = random.randint(0, m - 1)
        
        # Pick a random mask to swap in
        # We should pick from all_masks, but avoid those already in current_sets
        # To be efficient, we can just pick a random mask and check if it's in used_masks
        # Or pick from a precomputed list of unused masks?
        # Maintaining a list of unused masks is O(1) to pick, O(M) to update.
        
        # Let's just pick random mask and retry if in used
        # Probability of collision is 40/4096 ~ 1%. So retry is cheap.
        while True:
            mask_in = random.choice(all_masks)
            if mask_in not in used_masks:
                break
        
        # Calculate delta
        delta = get_gain_loss(current_sets, idx_out, mask_in)
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / temp):
            # Perform swap
            old_mask = current_sets[idx_out]
            current_sets[idx_out] = mask_in
            used_masks.remove(old_mask)
            used_masks.add(mask_in)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_sets = current_sets[:]
                
    # Return best found
    # Convert masks back to lists of integers
    result = []
    for mask in best_sets:
        s = []
        for i in range(n):
            if mask & (1 << i):
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
