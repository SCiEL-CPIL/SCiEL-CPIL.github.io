import random
import time
import math

def construct():
    n = 16
    # Pre-compute all possible triples
    all_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_triples.append((i, j, k))
                
    # Map triple to index
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute masks for all triples
    triple_masks = [(1<<a) | (1<<b) | (1<<c) for a,b,c in all_triples]
    
    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Sort masks by popcount descending
    masks_sorted = sorted(range(1, 1 << n), key=lambda x: popcount[x], reverse=True)
    
    # Precompute supersets for each triple
    # triple_supersets[i] is a list of masks that contain triple i
    triple_supersets = []
    for tm in triple_masks:
        # Find all masks m such that (m & tm) == tm
        # m = tm | s, where s is a subset of (~tm) & ((1<<n)-1)
        complement = (~tm) & ((1 << n) - 1)
        supersets = []
        # Iterate all subsets of complement
        # There are 2^(16-3) = 8192 subsets
        s = complement
        # Standard subset iteration
        while True:
            m = tm | s
            supersets.append(m)
            if s == 0:
                break
            s = (s - 1) & complement
        triple_supersets.append(supersets)
        
    # Initialize cover_count
    cover_count = [0] * (1 << n)
    
    def add_triple(t):
        idx = triple_index[t]
        for m in triple_supersets[idx]:
            cover_count[m] += 1
            
    def remove_triple(t):
        idx = triple_index[t]
        for m in triple_supersets[idx]:
            cover_count[m] -= 1
            
    def get_alpha():
        for m in masks_sorted:
            if cover_count[m] == 0:
                return popcount[m]
        return 0

    # Generate initial random maximal linear triple system
    def generate_initial():
        random.seed(42)
        used_pairs = set()
        triples_set = set()
        
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        for t in shuffled:
            a, b, c = t
            pairs = [(a,b), (a,c), (b,c)]
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                triples_set.add(t)
                used_pairs.update(pairs)
                add_triple(t)
                
        return triples_set

    # Simulated Annealing
    start_time = time.time()
    time_limit = 115
    
    current_triples = generate_initial()
    current_alpha = get_alpha()
    
    best_triples = current_triples.copy()
    best_alpha = current_alpha
    
    T = 1.0
    cooling = 0.99
    
    # Precompute pair sets for each triple
    triple_pairs = {}
    for t in all_triples:
        a,b,c = t
        triple_pairs[t] = [(a,b), (a,c), (b,c)]
        
    def get_available(triples_set):
        used_pairs = set()
        for t in triples_set:
            used_pairs.update(triple_pairs[t])
        
        available = []
        for t in all_triples:
            if t in triples_set:
                continue
            pairs = triple_pairs[t]
            if all(p not in used_pairs for p in pairs):
                available.append(t)
        return available

    available = get_available(current_triples)
    
    while time.time() - start_time < time_limit:
        move_type = random.random()
        
        # We need to track the effect on cover_count
        # We will perform the move, check alpha, and revert if rejected
        
        # Save state? No, just revert manually.
        
        t_add = None
        t_rem = None
        
        if move_type < 0.4 and available:
            t_add = random.choice(available)
        elif move_type < 0.7 and current_triples:
            t_rem = random.choice(list(current_triples))
        else:
            if current_triples:
                t_rem = random.choice(list(current_triples))
            if available:
                t_add = random.choice(available)
            else:
                if not t_rem:
                    continue

        # Apply move
        if t_add:
            add_triple(t_add)
            current_triples.add(t_add)
        if t_rem:
            remove_triple(t_rem)
            current_triples.discard(t_rem)
            
        new_alpha = get_alpha()
        delta = new_alpha - current_alpha
        
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = current_triples.copy()
            # Update available
            available = get_available(current_triples)
        else:
            # Revert
            if t_add:
                remove_triple(t_add)
                current_triples.discard(t_add)
            if t_rem:
                add_triple(t_rem)
                current_triples.add(t_rem)
            # available list is stale, but we recompute on accept.
            # If we reject, available might be wrong if we added/removed.
            # But we only use available to pick the next move.
            # If we rejected, the state is back to previous.
            # The previous available list was correct for the previous state.
            # So we don't need to update it on reject.
            
        T *= cooling
        if T < 0.01:
            T = 0.01

    return [list(t) for t in best_triples]
