import random
import time
import math
import numpy as np

def construct():
    n = 16
    # Pre-compute all possible triples
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
    
    # Pre-compute pairs for each triple
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = frozenset([(a,b), (a,c), (b,c)])
        
    # Pre-compute popcounts and masks for NumPy evaluation
    popcounts = np.array([bin(i).count('1') for i in range(1 << n)])
    masks = np.arange(1 << n)
    
    def calc_alpha(triples):
        if not triples:
            return n
        triple_masks = np.array([(1<<a) | (1<<b) | (1<<c) for a,b,c in triples])
        # Check independence using broadcasting
        # Shape: (65536, M)
        is_indep = np.all((masks[:, None] & triple_masks[None, :]) != triple_masks[None, :], axis=1)
        if not np.any(is_indep):
            return 0
        max_size = np.max(popcounts[is_indep])
        return int(max_size)

    def generate_random_maximal():
        cur_triples = []
        used_pairs_set = set()
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        for t in shuffled:
            pairs = triple_pairs[t]
            conflict = False
            for p in pairs:
                if p in used_pairs_set:
                    conflict = True
                    break
            if not conflict:
                cur_triples.append(t)
                used_pairs_set.update(pairs)
        return cur_triples

    start_time = time.time()
    time_limit = 110.0 # Leave buffer for safety
    
    # Initial solution
    current_triples = generate_random_maximal()
    current_triples_set = set(current_triples)
    current_alpha = calc_alpha(current_triples)
    
    best_alpha = current_alpha
    best_triples = current_triples[:]
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.995
    min_T = 0.05
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = 0.5 # Reheat to escape local minima
            
        if not current_triples:
            break
            
        # 1. Pick random t_out
        t_out = random.choice(current_triples)
        
        # 2. Compute used pairs for remaining triples
        used_pairs_set = set()
        for t in current_triples:
            if t == t_out: continue
            used_pairs_set.update(triple_pairs[t])
            
        # 3. Find candidates for t_in
        candidates = []
        for t in all_triples:
            if t in current_triples_set:
                continue
            # Check conflict with used_pairs_set
            conflict = False
            for p in triple_pairs[t]:
                if p in used_pairs_set:
                    conflict = True
                    break
            if not conflict:
                candidates.append(t)
                
        if not candidates:
            continue
            
        t_in = random.choice(candidates)
        
        # 4. Apply move
        new_triples_set = current_triples_set.copy()
        new_triples_set.remove(t_out)
        new_triples_set.add(t_in)
        new_triples = list(new_triples_set)
        
        # 5. Evaluate
        new_alpha = calc_alpha(new_triples)
        
        # 6. Acceptance
        delta = current_alpha - new_alpha
        if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
            current_triples = new_triples
            current_triples_set = new_triples_set
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples[:]
                
        T *= cooling
        
    return [list(t) for t in best_triples]
