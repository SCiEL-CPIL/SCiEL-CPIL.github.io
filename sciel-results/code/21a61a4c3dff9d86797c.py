import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 115.0
    
    n = 16
    points = list(range(n))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Precompute bitmasks for triples
    triple_masks = []
    triple_pairs = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
        a, b, c = t
        triple_pairs.append([(a,b), (a,c), (b,c)])
        
    # Precompute pair indices for fast lookup?
    # There are 120 pairs. Map pair to index.
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_to_idx[(i,j)] = idx
            idx += 1
    num_pairs = idx
            
    # Function to check independence number
    def get_indep_number(system_masks):
        # system_masks is a list of bitmasks
        # Create a boolean array of size 2^16
        # is_ind[i] = True if set i is independent
        is_ind = [True] * (1 << n)
        
        for mask in system_masks:
            # For all supersets S of mask, is_ind[S] = False
            # Supersets of mask: S = mask | sub, where sub is subset of complement
            # Iterate all subsets of complement?
            # Complement has 16-3=13 bits. 2^13 = 8192 subsets.
            # For each sub in subsets(complement):
            #    S = mask | sub
            #    is_ind[S] = False
            
            # Optimization: Iterate all masks S from 0 to 2^16-1? No, too slow if many triples.
            # Better: Iterate submasks of complement.
            comp = ((1 << n) - 1) ^ mask
            sub = comp
            while True:
                is_ind[mask | sub] = False
                if sub == 0: break
                sub = (sub - 1) & comp
                
        # Find max popcount where is_ind is True
        max_size = 0
        for i in range(1 << n):
            if is_ind[i]:
                pc = bin(i).count('1')
                if pc > max_size:
                    max_size = pc
        return max_size

    best_system = None
    best_indep = n + 1

    # Greedy Construction
    # We will run multiple restarts
    max_restarts = 20000 # Adjust based on speed
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Shuffle triples
        indices = list(range(num_triples))
        random.shuffle(indices)
        
        current_system = []
        current_masks = []
        used_pairs = set() # Set of pair indices
        
        for idx_t in indices:
            t = all_triples[idx_t]
            pairs = triple_pairs[idx_t]
            
            # Check if any pair is used
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
                    
            if not conflict:
                # Add triple
                current_system.append(t)
                current_masks.append(triple_masks[idx_t])
                for p in pairs:
                    used_pairs.add(pair_to_idx[p])
                    
        # Evaluate
        # Heuristic: If number of triples is low, indep is likely high.
        # Max possible triples is 40. If we have < 30, likely bad.
        if len(current_system) < 30:
            continue
            
        indep = get_indep_number(current_masks)
        
        if indep < best_indep:
            best_indep = indep
            best_system = current_system[:]
            
            if best_indep <= 6: # Theoretical lower bound?
                # 5 is possible?
                # If we find 6, it's very good.
                # We can break early if we find 5?
                # Let's aim for 6.
                if best_indep == 5:
                    break

    if best_system is None:
        # Fallback
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_system
