import time
import random

# EVOLVE-BLOCK-START
def construct():
    n = 12
    m = 40
    deadline = time.time() + 80  # Safety margin
    
    # Helper to check if a is subset of b
    def is_subset(a, b):
        return (a & b) == a
    
    # Helper to calculate score of a list of masks
    def calc_score(masks):
        score = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_subset(masks[i], masks[j]) or is_subset(masks[j], masks[i]):
                    score += 1
        return score
    
    # Helper to calculate delta when replacing mask at index i with new_mask
    def calc_delta(masks, i, new_mask):
        old_mask = masks[i]
        delta = 0
        for j in range(len(masks)):
            if j == i:
                continue
            other = masks[j]
            # Old contribution
            if is_subset(old_mask, other) or is_subset(other, old_mask):
                delta -= 1
            # New contribution
            if is_subset(new_mask, other) or is_subset(other, new_mask):
                delta += 1
        return delta

    # Initialize with random distinct masks
    # To improve start, we can bias towards a chain or dense area
    # Let's start with a random set of masks
    current_masks = []
    used = set()
    while len(current_masks) < m:
        mask = random.getrandbits(n)
        if mask not in used:
            used.add(mask)
            current_masks.append(mask)
            
    current_score = calc_score(current_masks)
    best_masks = current_masks[:]
    best_score = current_score
    
    # Local Search with Annealing
    T = 1.0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        i = random.randint(0, m - 1)
        # Pick a new mask not in the set
        while True:
            new_mask = random.getrandbits(n)
            if new_mask not in used:
                break
        
        delta = calc_delta(current_masks, i, new_mask)
        
        # Accept if better, or with probability exp(delta/T) if worse
        if delta > 0 or (random.random() < pow(2.718, delta / T)):
            # Update
            used.remove(current_masks[i])
            current_masks[i] = new_mask
            used.add(new_mask)
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = current_masks[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Slowly decrease temperature
            T *= 0.999
            
    # One final polish: try swapping pairs? No, single replacement is usually enough for this size.
    # Ensure we return the best found
    
    # Convert masks to lists of ints
    result = []
    for mask in best_masks:
        result.append([b for b in range(n) if (mask >> b) & 1])
        
    return result
# EVOLVE-BLOCK-END
