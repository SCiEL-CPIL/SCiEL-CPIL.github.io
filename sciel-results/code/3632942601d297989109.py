import time
import random
import numpy as np
from itertools import combinations

N_POINTS = 16
N_MASKS = 1 << N_POINTS
ALL_MASKS = range(N_MASKS)

# Precompute popcount
POP_COUNT = [0] * N_MASKS
for i in range(1, N_MASKS):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

# Precompute all triples as masks
TRIPLES = []
for a in range(N_POINTS):
    for b in range(a + 1, N_POINTS):
        for c in range(b + 1, N_POINTS):
            TRIPLES.append((1 << a) | (1 << b) | (1 << c))

# Map mask to index
MASK_TO_INDEX = {t: i for i, t in enumerate(TRIPLES)}

# Precompute which points are in each triple
TRIPLE_POINTS = []
for t in TRIPLES:
    pts = []
    for i in range(N_POINTS):
        if t & (1 << i):
            pts.append(i)
    TRIPLE_POINTS.append(pts)

# Precompute supersets for independence number calculation
# For each triple, the masks that contain it
SUPERSETS = [None] * len(TRIPLES)
for i, t in enumerate(TRIPLES):
    comp = ((1 << N_POINTS) - 1) ^ t
    sups = []
    sub = comp
    while True:
        sups.append(t | sub)
        if sub == 0:
            break
        sub = (sub - 1) & comp
    SUPERSETS[i] = sups

# Sort masks by popcount descending for alpha calculation
SORTED_MASKS = sorted(ALL_MASKS, key=lambda x: POP_COUNT[x], reverse=True)

# Precompute conflict matrix: conflict[i][j] = True if triple i and j share >= 2 points
CONFLICT = np.zeros((len(TRIPLES), len(TRIPLES)), dtype=bool)
for i in range(len(TRIPLES)):
    ti = TRIPLES[i]
    for j in range(i + 1, len(TRIPLES)):
        tj = TRIPLES[j]
        if (ti & tj).bit_count() >= 2:
            CONFLICT[i][j] = True
            CONFLICT[j][i] = True

# Precompute point sets for each triple as frozenset for faster intersection? 
# Or just use the bitmasks.

def _calc_alpha_fast(triple_indices):
    """
    Calculate independence number given a list of triple indices.
    Uses a bitset of 'forbidden' masks.
    """
    valid = [True] * N_MASKS
    # Optimization: Only mark supersets of the selected triples
    for idx in triple_indices:
        for s in SUPERSETS[idx]:
            valid[s] = False
    
    # Find largest valid set
    for s in SORTED_MASKS:
        if valid[s]:
            return POP_COUNT[s]
    return 0

def _calc_alpha_mask(triple_masks):
    """Calculate independence number from masks."""
    indices = [MASK_TO_INDEX[t] for t in triple_masks]
    return _calc_alpha_fast(indices)

def _can_add_indices(current_indices_set, new_idx):
    """Check if adding new_idx to current set of indices is valid."""
    if new_idx in current_indices_set:
        return False
    # Check conflicts against all existing
    for i in current_indices_set:
        if CONFLICT[new_idx][i]:
            return False
    return True

def _greedy_construct_indices(seed):
    """Greedy construction returning list of triple indices."""
    _random = random.Random(seed)
    available = list(range(len(TRIPLES)))
    _random.shuffle(available)
    
    selected = []
    selected_set = set()
    
    # Heuristic: prioritize triples that cover "new" pairs or points?
    # For now, simple random greedy.
    
    for idx in available:
        if _can_add_indices(selected_set, idx):
            selected.append(idx)
            selected_set.add(idx)
            
    return selected

def _local_search(start_indices, seed, time_limit=10.0):
    """
    Local search to minimize independence number.
    Moves: Add, Remove, Swap.
    """
    _random = random.Random(seed)
    current_indices = list(start_indices)
    current_set = set(current_indices)
    current_alpha = _calc_alpha_fast(current_indices)
    
    best_indices = current_indices[:]
    best_alpha = current_alpha
    best_set = current_set.copy()
    
    start_time = time.time()
    stall_count = 0
    max_stall = 500
    
    while time.time() - start_time < time_limit:
        move_type = _random.random()
        new_indices = current_indices[:]
        new_set = current_set.copy()
        
        if move_type < 0.3:
            # Add
            # Pick a random candidate
            if len(TRIPLES) > 0:
                cand = _random.choice(TRIPLES)
                cand_idx = MASK_TO_INDEX[cand]
                if cand_idx not in new_set and _can_add_indices(new_set, cand_idx):
                    new_indices.append(cand_idx)
                    new_set.add(cand_idx)
                else:
                    continue
        elif move_type < 0.6:
            # Remove
            if not current_indices:
                continue
            idx_to_remove = _random.choice(current_indices)
            new_indices.remove(idx_to_remove)
            new_set.remove(idx_to_remove)
        else:
            # Swap
            if not current_indices:
                continue
            idx_to_remove = _random.choice(current_indices)
            new_indices.remove(idx_to_remove)
            new_set.remove(idx_to_remove)
            
            # Try to add a compatible triple
            # Find all compatible triples
            # This might be slow if done naively.
            # We can pick a random few candidates.
            found = False
            for _ in range(10):
                cand_idx = _random.choice(TRIPLES)
                if cand_idx not in new_set and _can_add_indices(new_set, cand_idx):
                    new_indices.append(cand_idx)
                    new_set.add(cand_idx)
                    found = True
                    break
            if not found:
                # Revert
                new_indices.append(idx_to_remove)
                new_set.add(idx_to_remove)
                continue

        new_alpha = _calc_alpha_fast(new_indices)
        
        if new_alpha < current_alpha:
            current_indices = new_indices
            current_set = new_set
            current_alpha = new_alpha
            stall_count = 0
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_indices = current_indices[:]
                best_set = current_set.copy()
        elif new_alpha == current_alpha:
            # Accept to explore
            if _random.random() < 0.5:
                current_indices = new_indices
                current_set = new_set
                current_alpha = new_alpha
            stall_count += 1
        else:
            stall_count += 1
            
        if stall_count > max_stall:
            break
            
    return best_indices, best_alpha

def _algebraic_construct():
    """
    Construct a linear triple system based on affine geometry over GF(4) or similar.
    Points are 0..15.
    Let's try to construct a Steiner Triple System if possible, or a partial one.
    STS(16) does not exist (v must be 1 or 3 mod 6). 16 is 4 mod 6.
    So we can't have a full STS.
    However, we can try to construct a large partial linear space.
    
    Let's use the group Z_2 x Z_2 x Z_2 x Z_2.
    Affine lines in AG(4,2) have size 2. That's not triples.
    
    Let's try a different algebraic structure.
    Consider the 16 points as vertices of a 4-dimensional hypercube.
    Or maybe just use a known good construction if we can find one.
    
    Actually, let's just use a strong greedy heuristic based on pair coverage.
    """
    # This function is a placeholder for a specific algebraic design.
    # For now, return an empty list or a basic greedy result.
    return _greedy_construct_indices(0)

def construct():
    deadline = time.time() + 115.0
    
    best_alpha = 16
    best_indices = []
    
    # Phase 1: Multiple Greedy Starts
    # We want to find a good starting point.
    for seed in range(20):
        if time.time() > deadline - 40:
            break
        
        indices = _greedy_construct_indices(seed)
        alpha = _calc_alpha_fast(indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_indices = indices[:]
            
        # Early local search on promising starts
        if alpha <= 8: # If it's already quite good
            improved_indices, improved_alpha = _local_search(indices, seed, time_limit=2.0)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_indices = improved_indices[:]
                
    # Phase 2: Intensive Local Search on the best found
    if best_indices:
        # Run a longer local search
        improved_indices, improved_alpha = _local_search(best_indices, 12345, time_limit=60.0)
        if improved_alpha < best_alpha:
            best_alpha = improved_alpha
            best_indices = improved_indices[:]
            
        # Try a few more restarts of local search from the best state
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            # Perturb the best state slightly
            perturbed = best_indices[:]
            _random = random.Random(seed + 100)
            for _ in range(3):
                if perturbed:
                    idx_to_remove = _random.choice(perturbed)
                    perturbed.remove(idx_to_remove)
                    # Add a random compatible one
                    for _ in range(5):
                        cand_idx = _random.choice(TRIPLES)
                        if cand_idx not in perturbed and _can_add_indices(set(perturbed), cand_idx):
                            perturbed.append(cand_idx)
                            break
            
            p_alpha = _calc_alpha_fast(perturbed)
            if p_alpha <= best_alpha:
                improved_indices, improved_alpha = _local_search(perturbed, seed + 200, time_limit=10.0)
                if improved_alpha < best_alpha:
                    best_alpha = improved_alpha
                    best_indices = improved_indices[:]

    # Convert indices to output format
    result = []
    for idx in best_indices:
        mask = TRIPLES[idx]
        pts = TRIPLE_POINTS[idx]
        result.append(pts)
        
    return result

# Quick sanity check / main guard if run directly
if __name__ == "__main__":
    # Test
    t_start = time.time()
    res = construct()
    t_end = time.time()
    print(f"Time: {t_end - t_start:.2f}s")
    print(f"Number of triples: {len(res)}")
    
    # Verify linearity
    masks = []
    for r in res:
        m = 0
        for p in r:
            m |= (1 << p)
        masks.append(m)
        
    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            if (masks[i] & masks[j]).bit_count() >= 2:
                print(f"Invalid: triples {i} and {j} share more than 1 point")
                
    # Calculate alpha manually for verification
    alpha = _calc_alpha_mask(masks)
    print(f"Independence number: {alpha}")
