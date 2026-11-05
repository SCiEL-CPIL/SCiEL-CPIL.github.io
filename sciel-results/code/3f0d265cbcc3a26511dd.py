import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    triple_pairs = [ [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])] for t in all_triples ]
    triple_masks = [ (1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples ]
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    max_mask = 1 << 16
    full_mask = max_mask - 1
    
    # Popcount and subsets
    popcount = [0] * max_mask
    subsets_by_popcount = [[] for _ in range(17)]
    for i in range(max_mask):
        c = 0
        x = i
        while x:
            x &= x - 1
            c += 1
        popcount[i] = c
        subsets_by_popcount[c].append(i)
        
    # Supersets
    triple_supersets = []
    for m in triple_masks:
        comp = full_mask ^ m
        supersets = []
        sub = comp
        while True:
            supersets.append(m | sub)
            if sub == 0: break
            sub = (sub - 1) & comp
        triple_supersets.append(supersets)

    # State
    used_pairs = set()
    current_triple_indices = []  # Store indices, not tuples
    bad_count = [0] * max_mask
    current_alpha = 16
    
    best_alpha = 16
    best_triples = []
    
    def get_alpha():
        for k in range(16, -1, -1):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return 0

    def add_triple(idx):
        current_triple_indices.append(idx)
        for p in triple_pairs[idx]:
            used_pairs.add(p)
        for s in triple_supersets[idx]:
            bad_count[s] += 1
            
    def remove_triple(idx):
        # Remove from list by index value
        for i, idx_in_list in enumerate(current_triple_indices):
            if idx_in_list == idx:
                current_triple_indices.pop(i)
                break
        for p in triple_pairs[idx]:
            used_pairs.discard(p)
        for s in triple_supersets[idx]:
            bad_count[s] -= 1

    def update_alpha_after_add(old_alpha):
        # Check if old_alpha is still valid
        for s in subsets_by_popcount[old_alpha]:
            if bad_count[s] == 0:
                return old_alpha
        
        for k in range(old_alpha - 1, -1, -1):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return 0

    def update_alpha_after_remove(old_alpha):
        for k in range(old_alpha + 1, 17):
            for s in subsets_by_popcount[k]:
                if bad_count[s] == 0:
                    return k
        return old_alpha

    def is_valid_add(idx):
        for p in triple_pairs[idx]:
            if p in used_pairs:
                return False
        return True

    def init_sparse():
        nonlocal used_pairs, current_triple_indices, bad_count, current_alpha
        used_pairs.clear()
        current_triple_indices.clear()
        for i in range(max_mask):
            bad_count[i] = 0
            
        # Start with a sparse random selection
        indices = list(range(n_triples))
        random.shuffle(indices)
        # Add only about 20-30% of valid triples initially
        max_adds = n_triples // 3
        added = 0
        for idx in indices:
            if added >= max_adds:
                break
            if is_valid_add(idx):
                add_triple(idx)
                added += 1
        current_alpha = get_alpha()
        
    # Initial
    init_sparse()
    best_alpha = current_alpha
    best_triples = [all_triples[idx] for idx in current_triple_indices]
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    steps_since_improvement = 0
    max_stuck = 2000
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = []
            for idx in range(n_triples):
                if is_valid_add(idx):
                    candidates.append(idx)
            if not candidates:
                continue
                
            idx = random.choice(candidates)
            old_alpha = current_alpha
            
            add_triple(idx)
            new_alpha = update_alpha_after_add(old_alpha)
            delta = new_alpha - old_alpha
            current_alpha = new_alpha
            
            if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [all_triples[i] for i in current_triple_indices]
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                remove_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        elif move_type < 0.7:
            # Remove
            if not current_triple_indices:
                continue
            rem_pos = random.randrange(len(current_triple_indices))
            idx = current_triple_indices[rem_pos]
            
            old_alpha = current_alpha
            remove_triple(idx)
            new_alpha = update_alpha_after_remove(old_alpha)
            delta = new_alpha - old_alpha
            current_alpha = new_alpha
            
            if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [all_triples[i] for i in current_triple_indices]
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                add_triple(idx)
                current_alpha = old_alpha
                steps_since_improvement += 1

        else:
            # Swap
            if not current_triple_indices:
                continue
            rem_pos = random.randrange(len(current_triple_indices))
            idx_rem = current_triple_indices[rem_pos]
            
            # Temporarily remove
            remove_triple(idx_rem)
            
            # Find valid add
            candidates = []
            for idx_add in range(n_triples):
                if idx_add == idx_rem:
                    continue
                if is_valid_add(idx_add):
                    candidates.append(idx_add)
                    
            if not candidates:
                add_triple(idx_rem)
                continue
                
            idx_add = random.choice(candidates)
            
            # Calculate alpha after removal (state without idx_rem)
            alpha_after_remove = update_alpha_after_remove(current_alpha)
            
            # Add new triple
            add_triple(idx_add)
            new_alpha = update_alpha_after_add(alpha_after_remove)
            delta = new_alpha - current_alpha
            current_alpha = new_alpha
            
            if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [all_triples[i] for i in current_triple_indices]
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                # Undo: remove idx_add, re-add idx_rem
                remove_triple(idx_add)
                add_triple(idx_rem)
                current_alpha = alpha_after_remove  # Wait, this is wrong. We need the original alpha.
                # Actually, let's recalculate
                current_alpha = get_alpha()
                steps_since_improvement += 1

        T *= cooling
        if T < T_min: T = T_min
        
        if steps_since_improvement > max_stuck:
            init_sparse()
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_triples = [all_triples[i] for i in current_triple_indices]
            steps_since_improvement = 0
            T = 1.0

    return best_triples
