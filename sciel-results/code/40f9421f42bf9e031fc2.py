import random
import time
import numpy as np
import itertools

# Precompute popcount for all 16-bit masks
POP_COUNT = np.zeros(1 << 16, dtype=np.int8)
for i in range(1, 1 << 16):
    POP_COUNT[i] = POP_COUNT[i >> 1] + (i & 1)

ALL_MASKS = np.arange(1 << 16, dtype=np.int32)

def compute_alpha(triples):
    """
    Computes the independence number of a linear triple system on 16 points.
    Uses NumPy for vectorized subset checking.
    """
    if not triples:
        return 16
        
    # Precompute masks for triples
    triple_masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        triple_masks.append(m)
    
    if not triple_masks:
        return 16

    triple_masks_np = np.array(triple_masks, dtype=np.int32)
    
    # Check if any triple is contained in each mask
    # Shape: (65536, num_triples)
    # (mask & triple) == triple
    # This might be memory heavy if num_triples is large, but 35 is small.
    # 65536 * 35 * 4 bytes = ~9MB. Acceptable.
    
    intersects = (ALL_MASKS[:, None] & triple_masks_np[None, :]) == triple_masks_np[None, :]
    has_triple = intersects.any(axis=1)
    independent = ~has_triple
    
    if not independent.any():
        return 0 # Should not happen for empty set, but safe
        
    max_alpha = POP_COUNT[independent].max()
    return int(max_alpha)

def generate_all_triples():
    """Generate all possible triples on 16 points."""
    points = range(16)
    triples = []
    for a in points:
        for b in points:
            if b <= a: continue
            for c in points:
                if c <= b: continue
                triples.append((a, b, c))
    return triples

def get_used_pairs(triples):
    """Return a set of frozensets of pairs used in the triples."""
    used_pairs = set()
    for t in triples:
        for i in range(3):
            for j in range(i+1, 3):
                used_pairs.add(frozenset((t[i], t[j])))
    return used_pairs

def is_valid_triple(triple, used_pairs):
    """Check if a triple can be added to the system given the used pairs."""
    a, b, c = triple
    pairs = [frozenset((a, b)), frozenset((a, c)), frozenset((b, c))]
    for p in pairs:
        if p in used_pairs:
            return False
    return True

def greedy_construct(all_triples, seed=None, max_iter=100):
    """
    Randomized greedy construction of a linear triple system.
    Tries to maximize density.
    """
    if seed is not None:
        random.seed(seed)
        
    candidates = list(all_triples)
    random.shuffle(candidates)
    
    used_pairs = set()
    triples = []
    
    for t in candidates:
        if is_valid_triple(t, used_pairs):
            triples.append(t)
            a, b, c = t
            used_pairs.add(frozenset((a, b)))
            used_pairs.add(frozenset((a, c)))
            used_pairs.add(frozenset((b, c)))
            
    return triples

def local_search(triples, all_triples, time_limit):
    """
    Perform local search to minimize independence number.
    Moves: Remove a triple, add a different valid triple.
    """
    start_time = time.time()
    current_triples = list(triples)
    current_alpha = compute_alpha(current_triples)
    best_alpha = current_alpha
    best_triples = list(current_triples)
    
    used_pairs = get_used_pairs(current_triples)
    
    # Precompute valid triples list for faster random selection
    # We will pick a random triple to remove, and a random triple to add.
    # To be efficient, we can just pick random indices from all_triples.
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < time_limit:
        # Pick a random triple to remove
        if not current_triples:
            break
        idx_remove = random.randint(0, len(current_triples) - 1)
        triple_remove = current_triples[idx_remove]
        
        # Remove it
        current_triples.pop(idx_remove)
        a, b, c = triple_remove
        used_pairs.discard(frozenset((a, b)))
        used_pairs.discard(frozenset((a, c)))
        used_pairs.discard(frozenset((b, c)))
        
        # Try to add a random valid triple
        # We can just sample a few random triples from the universe
        added = False
        for _ in range(20): # Try up to 20 random candidates
            new_triple = all_triples[random.randint(0, len(all_triples) - 1)]
            if is_valid_triple(new_triple, used_pairs):
                current_triples.append(new_triple)
                na, nb, nc = new_triple
                used_pairs.add(frozenset((na, nb)))
                used_pairs.add(frozenset((na, nc)))
                used_pairs.add(frozenset((nb, nc)))
                added = True
                break
        
        if not added:
            # Revert
            current_triples.insert(idx_remove, triple_remove)
            used_pairs.add(frozenset((a, b)))
            used_pairs.add(frozenset((a, c)))
            used_pairs.add(frozenset((b, c)))
            continue

        new_alpha = compute_alpha(current_triples)
        
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = list(current_triples)
                if best_alpha <= 4: # Good enough to stop early? Maybe not, 3 is possible?
                    # For 16 points, minimum alpha is likely 4 or 5. 
                    # STS(15) has alpha 4? No, STS(15) covers all pairs. 
                    # Independence number of STS(v) is roughly sqrt(v).
                    # Let's keep searching.
                    pass
        else:
            # Accept worse move with probability exp(-delta/temp)
            delta = new_alpha - current_alpha
            if random.random() < np.exp(-delta / temp):
                current_alpha = new_alpha
            else:
                # Revert
                # Remove new_triple
                current_triples.pop()
                na, nb, nc = new_triple
                used_pairs.discard(frozenset((na, nb)))
                used_pairs.discard(frozenset((na, nc)))
                used_pairs.discard(frozenset((nb, nc)))
                
                # Re-add old triple
                current_triples.insert(idx_remove, triple_remove)
                used_pairs.add(frozenset((a, b)))
                used_pairs.add(frozenset((a, c)))
                used_pairs.add(frozenset((b, c)))
                
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return best_triples, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    all_triples = generate_all_triples()
    
    best_alpha = 16
    best_triples = []
    
    # Run multiple restarts with different seeds
    # Each restart: Greedy init + Local Search
    seed = 0
    while time.time() < deadline:
        # Allocate time for this run
        time_for_run = min(10.0, deadline - time.time())
        if time_for_run < 2.0:
            break
            
        # 1. Greedy Construction
        triples = greedy_construct(all_triples, seed=seed)
        
        # 2. Local Search
        # Give local search the bulk of the time
        ls_time_limit = time.time() + time_for_run - 0.5
        if ls_time_limit < time.time() + 0.5:
             ls_time_limit = time.time() + 0.5
             
        improved_triples, alpha = local_search(triples, all_triples, ls_time_limit)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = improved_triples
            
        seed += 1
        
    return [list(t) for t in best_triples]

