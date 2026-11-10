import random
import time
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    n = 16
    # Precompute popcount for 16-bit masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    def calc_alpha(triples):
        if not triples:
            return n
        
        # Convert triples to masks
        masks = [0] * len(triples)
        for i, t in enumerate(triples):
            masks[i] = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            
        max_a = 0
        # Iterate all subsets
        # Optimization: Skip if popcount[s] <= max_a
        for s in range(1, 1 << n):
            if popcount[s] <= max_a:
                continue
                
            is_ind = True
            for m in masks:
                if (s & m) == m:
                    is_ind = False
                    break
            if is_ind:
                max_a = popcount[s]
                
        return max_a

    def is_valid_triple(triple, existing_triples):
        a, b, c = triple
        mask_new = (1 << a) | (1 << b) | (1 << c)
        for t in existing_triples:
            mask_old = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
            inter = mask_new & mask_old
            if popcount[inter] > 1:
                return False
        return True

    def generate_initial_packing():
        available_pairs = set((i, j) for i in range(n) for j in range(i+1, n))
        triples = []
        pairs_list = list(available_pairs)
        random.shuffle(pairs_list)
        
        for u, v in pairs_list:
            if (u, v) not in available_pairs:
                continue
            candidates = []
            for w in range(n):
                if w == u or w == v:
                    continue
                p1 = (min(u, w), max(u, w))
                p2 = (min(v, w), max(v, w))
                if p1 in available_pairs and p2 in available_pairs:
                    candidates.append(w)
            
            if candidates:
                w = random.choice(candidates)
                triples.append([u, v, w])
                available_pairs.remove((u, v))
                available_pairs.remove(p1)
                available_pairs.remove(p2)
                
        return triples

    # Initialize
    best_triples = generate_initial_packing()
    best_alpha = calc_alpha(best_triples)
    
    current_triples = best_triples[:]
    current_alpha = best_alpha
    
    T = 1.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if not current_triples:
            break
            
        idx = random.randint(0, len(current_triples) - 1)
        old_triple = current_triples[idx]
        
        # Remove it
        current_triples.pop(idx)
        
        success = False
        # Try to find a valid replacement
        for _ in range(20):
            a, b, c = random.sample(range(n), 3)
            new_triple = [a, b, c]
            if is_valid_triple(new_triple, current_triples):
                current_triples.append(new_triple)
                
                new_alpha = calc_alpha(current_triples)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = current_triples[:]
                    success = True
                    break
                else:
                    delta = new_alpha - current_alpha
                    if delta == 0:
                        if random.random() < 0.5:
                            current_alpha = new_alpha
                            success = True
                            break
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_alpha = new_alpha
                            success = True
                            break
        
        if not success:
            # Revert
            current_triples.insert(idx, old_triple)
            # current_alpha is unchanged (still the value before the move)
            pass

        T *= cooling
        if T < T_min:
            T = T_min

    # Fallback if something went wrong
    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    return best_triples
