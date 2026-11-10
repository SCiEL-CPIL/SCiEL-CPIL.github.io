import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    n = 16
    all_points = list(range(n))
    
    # Precompute all triples as masks
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask = (1 << i) | (1 << j) | (1 << k)
                all_triples.append(mask)
                
    # Precompute masks by size for alpha calculation
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(1 << n):
        k = bin(mask).count('1')
        masks_by_size[k].append(mask)
        
    # Precompute conflicts: for each triple, which other triples conflict?
    # Conflict if intersection size >= 2
    # We can compute this on the fly or precompute.
    # There are 560 triples. 560*560 = 313k checks. Fast.
    conflicts = [[] for _ in range(len(all_triples))]
    for i in range(len(all_triples)):
        for j in range(i + 1, len(all_triples)):
            inter = all_triples[i] & all_triples[j]
            if bin(inter).count('1') >= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    def calc_alpha(triples_indices):
        if not triples_indices:
            return n
        # Convert to masks for fast bitwise ops
        # We pass indices, so we lookup masks
        # Optimization: pass masks directly?
        # Let's assume triples_indices is a list of indices into all_triples
        t_masks = [all_triples[i] for i in triples_indices]
        
        # Check sizes descending
        for k in range(n, 0, -1):
            for s in masks_by_size[k]:
                is_ind = True
                for t in t_masks:
                    if (s & t) == t:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    def is_valid_add(triple_idx, current_triples_set):
        # Check if triple_idx conflicts with any in current_triples_set
        # current_triples_set is a set of indices
        for c in conflicts[triple_idx]:
            if c in current_triples_set:
                return False
        return True

    def greedy_pack():
        current_triples = set()
        available = list(range(len(all_triples)))
        random.shuffle(available)
        
        for idx in available:
            if is_valid_add(idx, current_triples):
                current_triples.add(idx)
        
        return list(current_triples)

    best_alpha = n
    best_triples = None
    
    # Initial greedy attempts
    for _ in range(50):
        triples = greedy_pack()
        alpha = calc_alpha(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
            if alpha <= 5:
                # Good enough to start local search
                break

    # Local Search / Simulated Annealing
    if best_triples is not None:
        current_triples = set(best_triples)
        current_alpha = best_alpha
        
        # Temperature for SA
        T = 1.0
        cooling_rate = 0.99
        
        iterations = 0
        max_iterations = 50000 # Limit per restart
        
        while time.time() < deadline:
            # Try a random move
            move_type = random.random()
            
            new_triples = set(current_triples)
            
            if move_type < 0.5:
                # Add a random valid triple
                # Find all valid adds
                valid_adds = []
                for idx in range(len(all_triples)):
                    if idx not in new_triples and is_valid_add(idx, new_triples):
                        valid_adds.append(idx)
                
                if valid_adds:
                    new_triples.add(random.choice(valid_adds))
                else:
                    # No adds, maybe remove?
                    if new_triples:
                        new_triples.remove(random.choice(list(new_triples)))
            
            elif move_type < 0.8:
                # Remove a random triple
                if new_triples:
                    new_triples.remove(random.choice(list(new_triples)))
            
            else:
                # Swap: remove one, try to add another
                if new_triples:
                    to_remove = random.choice(list(new_triples))
                    new_triples.remove(to_remove)
                    
                    valid_adds = []
                    for idx in range(len(all_triples)):
                        if idx not in new_triples and is_valid_add(idx, new_triples):
                            valid_adds.append(idx)
                            
                    if valid_adds:
                        new_triples.add(random.choice(valid_adds))
            
            if len(new_triples) == len(current_triples) and new_triples == current_triples:
                continue
                
            new_alpha = calc_alpha(list(new_triples))
            
            # Acceptance criterion
            delta = new_alpha - current_alpha
            if delta <= 0:
                current_triples = new_triples
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = list(current_triples)
                    # Reset temperature if we found a new best?
                    T = max(T, 1.0)
            else:
                # Accept with probability
                # Use simple random acceptance for small deltas
                if random.random() < (0.5 ** delta):
                    current_triples = new_triples
                    current_alpha = new_alpha
            
            iterations += 1
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                # Perturb if stuck
                if iterations % 1000 == 0:
                    # Remove 3 random triples to escape local minima
                    for _ in range(3):
                        if current_triples:
                            current_triples.remove(random.choice(list(current_triples)))
                    
    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    # Convert indices back to points
    result = []
    for idx in best_triples:
        mask = all_triples[idx]
        pts = []
        for i in range(n):
            if mask & (1 << i):
                pts.append(i)
        result.append(pts)
        
    return result
