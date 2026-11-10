import random
import time
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 16
    points = list(range(n))
    
    # Precompute all possible triples
    all_triples = list(combinations(points, 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # For each triple, precompute the set of triples that conflict with it (share >=2 points)
    # A triple conflicts with another if they share a pair of points
    # Since we need linearity (share at most 1 point), any two triples sharing 2 or 3 points are invalid together
    
    # Precompute conflict masks using bitsets
    # For each triple i, conflict_mask[i] is a bitset of all triples that conflict with i
    # We can represent this as a list of integers or numpy arrays
    
    # Actually, for the search, we need to check if adding a triple conflicts with existing ones.
    # Let's precompute for each triple, the list of triples that conflict with it.
    conflict_lists = [[] for _ in range(num_triples)]
    
    for i, t1 in enumerate(all_triples):
        for j, t2 in enumerate(all_triples):
            if i == j:
                continue
            # Check if t1 and t2 share at least 2 points
            common = set(t1) & set(t2)
            if len(common) >= 2:
                conflict_lists[i].append(j)
                conflict_lists[j].append(i)
    
    # Convert conflict lists to sets for O(1) lookup
    conflict_sets = [set(lst) for lst in conflict_lists]
    
    # Helper to compute independence number exactly
    def calc_alpha(triples):
        if not triples:
            return n
        
        # Represent triples as bitmasks
        triple_masks = []
        for t in triples:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check all subsets to find max independent set
        max_ind = 0
        # Optimization: iterate subsets in descending order of size?
        # Actually, just check all 2^16 = 65536 subsets
        # For each subset, check if it contains any triple
        
        # Use numpy for vectorized check
        masks = np.array(triple_masks, dtype=np.uint32)
        
        # Generate all subset masks as numpy array
        all_subsets = np.arange(1 << n, dtype=np.uint32)
        
        # For each subset, check if it contains any triple
        # subset contains triple if (subset & triple_mask) == triple_mask
        
        # Broadcasting: all_subsets[:, None] & masks[None, :] == masks[None, :]
        # This creates a (65536, len(triples)) boolean array
        # Check if any triple is contained
        
        contained = np.any((all_subsets[:, None] & masks[None, :]) == masks[None, :], axis=1)
        
        # Independent subsets are those where no triple is contained
        independent = ~contained
        
        # Count bits for independent subsets
        # Use a lookup table or np.unpackbits
        # For speed, precompute popcount for all 2^16 values? 
        # 65536 entries is small.
        if not hasattr(calc_alpha, 'popcount_table'):
            calc_alpha.popcount_table = np.array([bin(i).count('1') for i in range(1 << n)], dtype=np.uint8)
        
        popcounts = calc_alpha.popcount_table[all_subsets]
        
        max_ind = int(np.max(popcounts[independent])) if np.any(independent) else 0
        
        return max_ind
    
    # Simulated Annealing
    def simulated_annealing(initial_triples, initial_used_pairs, start_temp, end_temp, cooling, time_limit):
        current_triples = list(initial_triples)
        current_used_pairs = set(initial_used_pairs)
        current_alpha = calc_alpha(current_triples)
        
        best_triples = list(current_triples)
        best_alpha = current_alpha
        
        temp = start_temp
        t_start = time.time()
        
        while time.time() - t_start < time_limit and temp > end_temp:
            # Pick a random move: remove a triple, add a triple, or swap
            move_type = random.random()
            
            new_triples = list(current_triples)
            new_used_pairs = set(current_used_pairs)
            
            if move_type < 0.3 and len(new_triples) > 0:
                # Remove a random triple
                idx = random.randint(0, len(new_triples) - 1)
                t_remove = new_triples[idx]
                for i in range(3):
                    for j in range(i+1, 3):
                        new_used_pairs.discard(frozenset((t_remove[i], t_remove[j])))
                new_triples.pop(idx)
                
            elif move_type < 0.7:
                # Add a random triple
                # Find triples that can be added
                candidates = []
                for t_idx in range(num_triples):
                    t = all_triples[t_idx]
                    valid = True
                    for i in range(3):
                        for j in range(i+1, 3):
                            p = frozenset((t[i], t[j]))
                            if p in new_used_pairs:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        candidates.append(t_idx)
                
                if candidates:
                    t_idx = random.choice(candidates)
                    t_add = all_triples[t_idx]
                    new_triples.append(list(t_add))
                    for i in range(3):
                        for j in range(i+1, 3):
                            new_used_pairs.add(frozenset((t_add[i], t_add[j])))
                
            else:
                # Swap: remove one, add one
                if len(new_triples) == 0:
                    continue
                
                idx = random.randint(0, len(new_triples) - 1)
                t_remove = new_triples[idx]
                for i in range(3):
                    for j in range(i+1, 3):
                        new_used_pairs.discard(frozenset((t_remove[i], t_remove[j])))
                new_triples.pop(idx)
                
                # Find addable triples
                candidates = []
                for t_idx in range(num_triples):
                    t = all_triples[t_idx]
                    valid = True
                    for i in range(3):
                        for j in range(i+1, 3):
                            p = frozenset((t[i], t[j]))
                            if p in new_used_pairs:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        candidates.append(t_idx)
                
                if candidates:
                    t_idx = random.choice(candidates)
                    t_add = all_triples[t_idx]
                    new_triples.append(list(t_add))
                    for i in range(3):
                        for j in range(i+1, 3):
                            new_used_pairs.add(frozenset((t_add[i], t_add[j])))
            
            new_alpha = calc_alpha(new_triples)
            
            delta = new_alpha - current_alpha
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_triples = new_triples
                current_used_pairs = new_used_pairs
                current_alpha = new_alpha
                
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            
            temp *= cooling
        
        return best_triples, best_alpha
    
    best_triples = []
    best_alpha = n
    
    # Initial solution: greedy random
    triples = []
    used_pairs = set()
    shuffled = all_triples[:]
    random.shuffle(shuffled)
    for t in shuffled:
        valid = True
        for i in range(3):
            for j in range(i+1, 3):
                p = frozenset((t[i], t[j]))
                if p in used_pairs:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            triples.append(list(t))
            for i in range(3):
                for j in range(i+1, 3):
                    used_pairs.add(frozenset((t[i], t[j])))
    
    # Run simulated annealing
    while time.time() < deadline - 10:
        result_triples, result_alpha = simulated_annealing(triples, used_pairs, 1.0, 0.1, 0.95, 30.0)
        
        if result_alpha < best_alpha:
            best_alpha = result_alpha
            best_triples = result_triples[:]
            
            if best_alpha <= 6:
                # Good result, keep searching but with tighter parameters
                pass
        else:
            # Restart with a new random initial solution
            triples = []
            used_pairs = set()
            shuffled = all_triples[:]
            random.shuffle(shuffled)
            for t in shuffled:
                valid = True
                for i in range(3):
                    for j in range(i+1, 3):
                        p = frozenset((t[i], t[j]))
                        if p in used_pairs:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    triples.append(list(t))
                    for i in range(3):
                        for j in range(i+1, 3):
                            used_pairs.add(frozenset((t[i], t[j])))
    
    return best_triples
