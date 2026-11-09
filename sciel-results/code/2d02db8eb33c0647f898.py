import random
import itertools
import time

def _independence_number(triples, n=16):
    """Compute the independence number: largest subset with no triple."""
    if not triples:
        return n
    
    # Precompute triple masks
    triple_masks = []
    for t in triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
    
    best = 0
    # Iterate over all masks
    # Optimization: Skip masks that are subsets of already checked? No.
    # Just iterate all 2^16 masks.
    for mask in range(1 << n):
        # Quick check: if popcount <= best, skip
        # Using bin(mask).count('1') is fast enough for 65536 iterations
        bits = bin(mask).count('1')
        if bits <= best:
            continue
        
        # Check if any triple is contained in mask
        is_independent = True
        for tm in triple_masks:
            if (mask & tm) == tm:
                is_independent = False
                break
        
        if is_independent:
            best = bits
            # Early exit if we found a perfect cover? No, we want max independent.
            # If best == n, we can break? No, n is max.
            # If best == n, return n.
    
    return best

def _generate_packing(n=16, seed=0):
    """Generate a linear triple system using randomized greedy."""
    rng = random.Random(seed)
    used_pairs = set()
    triples = []
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(range(n), 3))
    rng.shuffle(all_triples)
    
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        # Check if any pair is already used
        if any(tuple(sorted(p)) in used_pairs for p in pairs):
            continue
        # Add this triple
        for p in pairs:
            used_pairs.add(tuple(sorted(p)))
        triples.append(list(t))
    
    return triples

def construct():
    n = 16
    # Initialize with a good greedy packing
    current_triples = _generate_packing(n, seed=42)
    current_indep = _independence_number(current_triples, n)
    
    # Simulated Annealing
    # We need to define a move: swap a triple with another valid one.
    # Precompute all valid triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # To make moves fast, we need to check validity quickly.
    # A triple T is valid if it doesn't share a pair with any triple in the current set.
    
    start_time = time.time()
    timeout = 100  # Leave 20s buffer for evaluation overhead
    
    best_triples = current_triples
    best_indep = current_indep
    
    # Temperature parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-4
    
    # Precompute pair sets for all triples?
    # triple_pairs[i] = set of 3 pairs for triple i
    triple_pairs = []
    for t in all_triples:
        pairs = {tuple(sorted((t[0], t[1]))), tuple(sorted((t[0], t[2]))), tuple(sorted((t[1], t[2])))}
        triple_pairs.append(pairs)
    
    # Map from pair to list of triples containing it?
    # Or just check against current set.
    
    # Current set of used pairs
    used_pairs = set()
    for t in current_triples:
        used_pairs.add(tuple(sorted((t[0], t[1]))))
        used_pairs.add(tuple(sorted((t[0], t[2]))))
        used_pairs.add(tuple(sorted((t[1], t[2]))))
        
    # Index of triples in current set?
    # We store current_triples as a list.
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Pick a random triple to remove
        idx = random.randint(0, len(current_triples) - 1)
        old_triple = current_triples[idx]
        old_pairs = {tuple(sorted((old_triple[0], old_triple[1]))), 
                     tuple(sorted((old_triple[0], old_triple[2]))), 
                     tuple(sorted((old_triple[1], old_triple[2])))}
        
        # Remove it from used_pairs
        for p in old_pairs:
            used_pairs.discard(p)
            
        # Pick a random new triple from all_triples
        # To be efficient, we can pick from a list of "potentially valid" triples?
        # Or just pick random and check.
        # There are 560 triples.
        # Probability of conflict is high if set is dense.
        # Let's just pick a few random candidates.
        
        new_triple = None
        for _ in range(10): # Try 10 random candidates
            cand_idx = random.randint(0, len(all_triples) - 1)
            cand = all_triples[cand_idx]
            cand_pairs = triple_pairs[cand_idx]
            
            # Check if cand conflicts with current used_pairs
            if not any(p in used_pairs for p in cand_pairs):
                new_triple = cand
                break
        
        if new_triple is None:
            # Restore old triple
            current_triples[idx] = old_triple
            for p in old_pairs:
                used_pairs.add(p)
            T *= cooling_rate
            continue
            
        # Apply move
        current_triples[idx] = list(new_triple)
        for p in triple_pairs[all_triples.index(new_triple)]: # Note: index lookup is slow, precompute mapping?
            # Better: just use the pairs we already have
            pass
        
        # Recalculate used_pairs
        # We removed old_pairs. Now add new_pairs.
        new_pairs = triple_pairs[all_triples.index(new_triple)]
        for p in new_pairs:
            used_pairs.add(p)
            
        # Evaluate new independence number
        new_indep = _independence_number(current_triples, n)
        
        # Acceptance criterion
        delta = new_indep - current_indep
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_indep = new_indep
            if new_indep < best_indep:
                best_indep = new_indep
                best_triples = [list(t) for t in current_triples]
        else:
            # Revert
            current_triples[idx] = old_triple
            for p in new_pairs:
                used_pairs.discard(p)
            for p in old_pairs:
                used_pairs.add(p)
            
        T *= cooling_rate
        
    return best_triples
