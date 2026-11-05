import random
import time
import itertools

# Precompute all possible triples on 16 points
POINTS = 16
ALL_TRIPLES = []
TRIPLE_MASKS = []
for combo in itertools.combinations(range(POINTS), 3):
    mask = 0
    for p in combo:
        mask |= (1 << p)
    ALL_TRIPLES.append(list(combo))
    TRIPLE_MASKS.append(mask)

def are_compatible(mask1, mask2):
    # Check if they share at most 1 point
    return (mask1 & mask2).bit_count() <= 1

def is_valid_addition(candidate_mask, current_masks):
    for m in current_masks:
        if not are_compatible(candidate_mask, m):
            return False
    return True

def calc_alpha(triple_indices):
    """
    Calculate the independence number of the hypergraph defined by the given triple indices.
    triple_indices: list of indices into ALL_TRIPLES
    """
    if not triple_indices:
        return POINTS
        
    masks = [TRIPLE_MASKS[i] for i in triple_indices]
    
    max_alpha = 0
    # Iterate through all subsets of points
    for s in range(1 << POINTS):
        # Check if any triple is contained in s
        # If s contains a triple, it's not independent
        is_independent = True
        for m in masks:
            if (s & m) == m:
                is_independent = False
                break
        if is_independent:
            alpha = s.bit_count()
            if alpha > max_alpha:
                max_alpha = alpha
    return max_alpha

def greedy_init():
    """
    Construct a maximal packing using greedy random addition.
    """
    current_indices = []
    current_masks = []
    
    # Shuffle indices to randomize
    indices = list(range(len(ALL_TRIPLES)))
    random.shuffle(indices)
    
    for idx in indices:
        mask = TRIPLE_MASKS[idx]
        if is_valid_addition(mask, current_masks):
            current_indices.append(idx)
            current_masks.append(mask)
            
    return current_indices

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    # Initial solution
    current_indices = greedy_init()
    current_alpha = calc_alpha(current_indices)
    
    best_indices = current_indices.copy()
    best_alpha = current_alpha
    
    # Simulated Annealing parameters
    T = 5.0
    cooling_rate = 0.995
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Generate a move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_indices = current_indices.copy()
        
        if move_type == 'add':
            # Try to add a random triple
            idx = random.randint(0, len(ALL_TRIPLES) - 1)
            mask = TRIPLE_MASKS[idx]
            if is_valid_addition(mask, [TRIPLE_MASKS[i] for i in new_indices]):
                new_indices.append(idx)
            else:
                # If cannot add, do nothing (or skip)
                continue
                
        elif move_type == 'remove':
            if not new_indices:
                continue
            idx_to_remove = random.choice(new_indices)
            new_indices.remove(idx_to_remove)
            
        elif move_type == 'swap':
            if not new_indices:
                continue
            # Remove a random existing triple
            idx_to_remove = random.choice(new_indices)
            new_indices.remove(idx_to_remove)
            # Try to add a random new triple
            found = False
            for _ in range(10): # Try a few times to find a valid replacement
                idx_to_add = random.randint(0, len(ALL_TRIPLES) - 1)
                mask = TRIPLE_MASKS[idx_to_add]
                if is_valid_addition(mask, [TRIPLE_MASKS[i] for i in new_indices]):
                    new_indices.append(idx_to_add)
                    found = True
                    break
            if not found:
                # If we couldn't find a replacement, revert to just removing? 
                # Or skip move. Let's skip move to keep density similar.
                continue

        # Evaluate new solution
        new_alpha = calc_alpha(new_indices)
        
        # Acceptance criterion
        if new_alpha < current_alpha:
            current_indices = new_indices
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_indices = current_indices.copy()
        else:
            # Accept with probability exp(-(new_alpha - current_alpha) / T)
            # Since we minimize, delta = new_alpha - current_alpha >= 0
            delta = new_alpha - current_alpha
            if delta == 0:
                current_indices = new_indices
                current_alpha = new_alpha
            else:
                prob = 2.71828 ** (-delta / T)
                if random.random() < prob:
                    current_indices = new_indices
                    current_alpha = new_alpha
        
        T *= cooling_rate
        
    # Return best found as list of triples
    return [ALL_TRIPLES[i] for i in best_indices]

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it or ensure it's the entry point.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program:
# def construct():
#     return ...
# So I will define construct() as the main entry point.
# EVOLVE-BLOCK-END
