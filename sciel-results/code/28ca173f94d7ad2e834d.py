import random
import time
import itertools
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    all_points = list(range(n))
    all_triples = list(itertools.combinations(range(n), 3))
    
    # Precompute pair for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = [(a, b), (a, c), (b, c)]
    
    # Precompute: for each pair, which triples contain it
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)
    
    # Bitmask helper: precompute masks for all 2^n subsets
    # For alpha computation: given a set of triple masks, find largest subset with no triple contained
    # Strategy: for each triple mask T, mark all supersets of T as "covered"
    # Then alpha = largest size of an unmarked mask
    
    # Precompute for each mask, its size (popcount)
    mask_size = [0] * (1 << n)
    for mask in range(1, 1 << n):
        mask_size[mask] = mask_size[mask >> 1] + (mask & 1)
    
    # For fast alpha: we'll use a different approach
    # Given k triples, we want the largest S such that no triple is a subset of S
    # Equivalent: S is an independent set in the 3-uniform hypergraph
    
    # Fast method: enumerate subsets from largest size down, check if any triple is contained
    # But 2^16 = 65536, and for each we check k triples... could be slow
    # Better: for each triple, mark all its supersets. Use a boolean array of size 2^n.
    # Then find max size unmarked.
    
    # Even better: use the fact that we only need to check subsets of size >= some threshold
    # Start from size 16 down, and for each size, check all subsets of that size
    
    # Precompute all masks by size for fast lookup
    masks_by_size = {}
    for size in range(n + 1):
        masks_by_size[size] = [sum(1 << p for p in comb) for comb in itertools.combinations(range(n), size)]
    
    def compute_alpha(triples):
        """Compute the independence number exactly."""
        if not triples:
            return n
        
        # Mark all subsets that contain at least one triple
        # Use a bytearray for speed
        covered = bytearray(1 << n)
        
        for t in triples:
            a, b, c = t
            t_mask = (1 << a) | (1 << b) | (1 << c)
            # Mark all supersets of t_mask
            # Supersets of t_mask are masks m such that (m & t_mask) == t_mask
            # We can iterate over all masks and check, but that's 2^n per triple
            # Better: iterate over the free bits and enumerate
            free_bits = []
            for i in range(n):
                if not (t_mask >> i) & 1:
                    free_bits.append(i)
            k = len(free_bits)
            for sub in range(1 << k):
                extra = 0
                for j in range(k):
                    if (sub >> j) & 1:
                        extra |= (1 << free_bits[j])
                covered[t_mask | extra] = 1
        
        # Find largest unmarked subset
        for size in range(n, -1, -1):
            for mask in masks_by_size[size]:
                if not covered[mask]:
                    return size
        return 0
    
    def greedy_construct(seed_order=None):
        """Randomized greedy construction of a linear triple system."""
        triples = []
        used_pairs = set()
        
        candidates = all_triples[:]
        if seed_order is not None:
            # Use a seeded shuffle
            rng = random.Random(seed_order)
            rng.shuffle(candidates)
        else:
            random.shuffle(candidates)
        
        for t in candidates:
            a, b, c = t
            pairs = triple_pairs[t]
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
    
    def local_search(triples, alpha, time_limit):
        """Local search to improve alpha."""
        current_triples = triples[:]
        current_alpha = alpha
        
        # Build used_pairs
        used_pairs = set()
        for t in current_triples:
            for p in triple_pairs[t]:
                used_pairs.add(p)
        
        best_alpha = current_alpha
        best_triples = current_triples[:]
        
        temperature = 1.0
        cooling_rate = 0.9995
        no_improve_count = 0
        
        while time.time() < time_limit and temperature > 0.001:
            if not current_triples:
                break
            
            # Pick a random triple to remove
            idx = random.randint(0, len(current_triples) - 1)
            t_remove = current_triples[idx]
            
            # Find valid replacement triples (those whose pairs don't conflict with current used_pairs minus t_remove's pairs)
            remove_pairs = triple_pairs[t_remove]
            temp_used = set(used_pairs)
            for p in remove_pairs:
                temp_used.discard(p)
            
            # Find candidates
            valid_candidates = []
            for t in all_triples:
                if t == t_remove:
                    continue
                pairs = triple_pairs[t]
                if all(p not in temp_used for p in pairs):
                    valid_candidates.append(t)
            
            if not valid_candidates:
                temperature *= cooling_rate
                continue
            
            # Pick a candidate (random or heuristic)
            t_new = random.choice(valid_candidates)
            
            # Apply move
            current_triples[idx] = t_new
            for p in remove_pairs:
                used_pairs.discard(p)
            for p in triple_pairs[t_new]:
                used_pairs.add(p)
            
            # Evaluate
            new_alpha = compute_alpha(current_triples)
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                no_improve_count = 0
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
                    if best_alpha <= 6:
                        break
            elif new_alpha == current_alpha:
                # Accept with some probability
                if random.random() < 0.3:
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples[idx] = t_remove
                    for p in triple_pairs[t_new]:
                        used_pairs.discard(p)
                    for p in remove_pairs:
                        used_pairs.add(p)
            else:
                # Worse: accept with probability exp(-delta/T)
                delta = new_alpha - current_alpha
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples[idx] = t_remove
                    for p in triple_pairs[t_new]:
                        used_pairs.discard(p)
                    for p in remove_pairs:
                        used_pairs.add(p)
            
            no_improve_count += 1
            if no_improve_count > 500:
                temperature *= 0.95
                no_improve_count = 0
            
            temperature *= cooling_rate
        
        return best_triples, best_alpha
    
    # Multi-restart strategy
    best_alpha = n
    best_triples = []
    
    # Try several diverse initial constructions
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        seed = num_restarts * 12345 + 67890
        triples = greedy_construct(seed)
        alpha = compute_alpha(triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
        
        # Local search from this starting point
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            ls_triples, ls_alpha = local_search(triples, alpha, time.time() + remaining_time - 2)
            if ls_alpha < best_alpha:
                best_alpha = ls_alpha
                best_triples = ls_triples[:]
        
        if best_alpha <= 6:
            break
    
    return best_triples
