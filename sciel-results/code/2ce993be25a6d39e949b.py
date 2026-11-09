import random
import time
import itertools
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    n = 16
    points = list(range(n))
    
    best_alpha = n
    best_triples = []
    
    # Precompute all possible triples and their masks
    all_triples = list(itertools.combinations(range(n), 3))
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in all_triples]
    
    # Precompute masks for all subsets of size >= 8 (since alpha is likely around 8)
    # We can check sizes from 15 down to 0.
    # To speed up alpha calculation, we precompute lists of masks for each size.
    masks_by_size = {}
    for size in range(16, -1, -1):
        masks_by_size[size] = [sum(1 << p for p in comb) for comb in itertools.combinations(range(n), size)]

    def get_alpha(triples):
        if not triples:
            return n
            
        # Create a set of masks for fast lookup
        t_masks = set()
        for t in triples:
            mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            t_masks.add(mask)
            
        # Check sizes from high to low
        for size in range(n, 0, -1):
            for mask in masks_by_size[size]:
                # Check if any triple is a subset of mask
                # Optimization: if we just need to know if ANY triple is contained
                # We can iterate t_masks and check (t & mask) == t
                # This is O(|triples|) per mask.
                # To speed up, we can break early if found.
                independent = True
                for t in t_masks:
                    if (t & mask) == t:
                        independent = False
                        break
                if independent:
                    return size
        return 0

    def try_random_greedy():
        triples = []
        used_pairs = set()
        
        # Random order of triples
        random.shuffle(all_triples)
        
        for t in all_triples:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            valid = True
            for p in pairs:
                if p in used_pairs:
                    valid = False
                    break
            if valid:
                triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
                    
        return triples

    # Initial greedy solution
    current_triples = try_random_greedy()
    current_alpha = get_alpha(current_triples)
    best_alpha = current_alpha
    best_triples = current_triples[:]
    
    # Simulated Annealing
    # We will try to swap one triple with another to see if we can reduce alpha.
    # Or add/remove triples.
    # Since linearity constraint is hard to maintain with random additions, 
    # swapping is safer if we ensure the new triple doesn't conflict with pairs of the removed triple?
    # Actually, if we remove T1 and add T2, we must ensure T2 doesn't share a pair with any OTHER triple.
    # And T2 can share points with T1 (since T1 is removed), but it must not share pairs with existing ones.
    
    # Let's maintain a set of used pairs for the current solution.
    current_pairs = set()
    for t in current_triples:
        a, b, c = t
        current_pairs.add((a,b))
        current_pairs.add((a,c))
        current_pairs.add((b,c))
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if temperature < 0.001:
            break
            
        # Pick a random triple to remove
        if not current_triples:
            break
            
        idx = random.randint(0, len(current_triples) - 1)
        t_remove = current_triples[idx]
        
        # Remove it from current state
        current_triples.pop(idx)
        
        # Remove its pairs from used_pairs
        a, b, c = t_remove
        current_pairs.discard((a,b))
        current_pairs.discard((a,c))
        current_pairs.discard((b,c))
        
        # Find a valid triple to add
        # To be efficient, we should pick a candidate that might be "good".
        # Random is okay for now.
        # We need a triple T_new such that no pair in T_new is in current_pairs.
        
        # Optimization: Pre-compute for each triple which other triples conflict with it?
        # Or just iterate through all_triples? 560 triples is small.
        
        candidate_found = False
        # Shuffle all_triples to pick a random valid one
        candidates = all_triples[:]
        random.shuffle(candidates)
        
        for t_new in candidates:
            x, y, z = t_new
            pairs_new = [(x,y), (x,z), (y,z)]
            valid = True
            for p in pairs_new:
                if p in current_pairs:
                    valid = False
                    break
            if valid:
                # Accept move
                current_triples.append(t_new)
                for p in pairs_new:
                    current_pairs.add(p)
                candidate_found = True
                break
                
        if not candidate_found:
            # Revert
            current_triples.insert(idx, t_remove)
            current_pairs.add((a,b))
            current_pairs.add((a,c))
            current_pairs.add((b,c))
            temperature *= cooling_rate
            continue
            
        # Evaluate new alpha
        new_alpha = get_alpha(current_triples)
        
        # Metropolis criterion
        if new_alpha < current_alpha:
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
                if best_alpha <= 7:
                    break
        else:
            # Accept with probability exp(-delta / T)
            # delta = new_alpha - current_alpha (positive)
            delta = new_alpha - current_alpha
            if delta == 0:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples.pop()
                    for p in pairs_new:
                        current_pairs.discard(p)
                    current_triples.insert(idx, t_remove)
                    current_pairs.add((a,b))
                    current_pairs.add((a,c))
                    current_pairs.add((b,c))
            else:
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples.pop()
                    for p in pairs_new:
                        current_pairs.discard(p)
                    current_triples.insert(idx, t_remove)
                    current_pairs.add((a,b))
                    current_pairs.add((a,c))
                    current_pairs.add((b,c))
                    
        temperature *= cooling_rate

    return best_triples
# EVOLVE-BLOCK-END
