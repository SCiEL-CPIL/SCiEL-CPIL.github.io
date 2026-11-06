import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Map pair -> triples containing it
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)
            
    # Precompute bitmask for each triple
    triple_masks = {}
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[t] = mask

    def is_independent(mask, t_masks):
        """Check if the set represented by mask is independent (contains no triple)."""
        for tm in t_masks:
            if (mask & tm) == tm:
                return False
        return True

    def get_alpha_fast(triples):
        """
        Compute the independence number using hitting set formulation.
        alpha = 16 - min_hitting_set_size
        Returns (alpha, witness_set).
        """
        if not triples:
            return 16, set(points)
            
        t_masks = [triple_masks[t] for t in triples]
        full_mask = (1 << 16) - 1
        
        # Find minimum hitting set C such that every triple intersects C
        # Then alpha = 16 - |C|, and witness = all points not in C
        for c_size in range(1, 17):
            s_size = 16 - c_size
            
            if s_size >= 8:
                # Enumerate hitting sets of size c_size
                for C in itertools.combinations(points, c_size):
                    c_mask = 0
                    for p in C:
                        c_mask |= (1 << p)
                    hits_all = True
                    for tm in t_masks:
                        if (c_mask & tm) == 0:
                            hits_all = False
                            break
                    if hits_all:
                        s_mask = full_mask ^ c_mask
                        witness = set()
                        for p in points:
                            if s_mask & (1 << p):
                                witness.add(p)
                        return s_size, witness
            else:
                # Enumerate independent sets of size s_size
                for S in itertools.combinations(points, s_size):
                    s_mask = 0
                    for p in S:
                        s_mask |= (1 << p)
                    if is_independent(s_mask, t_masks):
                        return s_size, set(S)
        
        return 1, set([0])

    # Strategy:
    # 1. Generate random greedy packings.
    # 2. Local search with guidance from the witness set.
    
    best_alpha = 16
    best_system = []
    
    # Initialize with random greedy packings
    n_restarts = 30
    for _ in range(n_restarts):
        if time.time() - start_time > deadline - 30:
            break
        random.shuffle(all_triples)
        current = []
        used_pairs = set()
        for t in all_triples:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        alpha, _ = get_alpha_fast(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]
            
        if best_alpha <= 5:
            break
    
    current_system = best_system[:]
    current_alpha = best_alpha
    current_witness = set()
    
    # Update witness for current best
    if current_system:
        current_alpha, current_witness = get_alpha_fast(current_system)
    else:
        current_witness = set(points)

    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    while time.time() - start_time < deadline:
        # Get current witness if it changed or periodically
        # To save time, we might not recalculate witness every single step, 
        # but if alpha drops, we must.
        
        move_type = random.choice(['add', 'swap', 'drop'])
        
        if move_type == 'add':
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                # Bias towards candidates that intersect the witness set more
                witness_set = current_witness
                scored = []
                for t in candidates:
                    inter = len(set(t) & witness_set)
                    scored.append((inter, t))
                
                # Pick from top-scoring candidates with some randomness
                scored.sort(key=lambda x: -x[0])
                top_n = min(20, len(scored))
                weights = [scored[i][0] + 1 for i in range(top_n)]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen_idx = 0
                for i in range(top_n):
                    cum += weights[i]
                    if r <= cum:
                        chosen_idx = i
                        break
                
                new_t = scored[chosen_idx][1]
                new_system = current_system + [new_t]
                new_alpha, new_witness = get_alpha_fast(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < 0.3:
                        current_system = new_system
                        current_witness = new_witness

        elif move_type == 'swap':
            if not current_system:
                continue
                
            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]
            
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs) and t != t_remove:
                    candidates.append(t)
            
            if candidates:
                witness_set = current_witness
                scored = []
                for t in candidates:
                    inter = len(set(t) & witness_set)
                    scored.append((inter, t))
                
                scored.sort(key=lambda x: -x[0])
                top_n = min(20, len(scored))
                weights = [scored[i][0] + 1 for i in range(top_n)]
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                chosen_idx = 0
                for i in range(top_n):
                    cum += weights[i]
                    if r <= cum:
                        chosen_idx = i
                        break
                
                new_t = scored[chosen_idx][1]
                new_system = current_system[:]
                new_system[idx] = new_t
                new_alpha, new_witness = get_alpha_fast(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < 0.3:
                        current_system = new_system
                        current_witness = new_witness

        elif move_type == 'drop':
            if not current_system:
                continue
            
            idx = random.randint(0, len(current_system) - 1)
            new_system = current_system[:]
            new_system.pop(idx)
            new_alpha, new_witness = get_alpha_fast(new_system)
            
            if new_alpha <= current_alpha:
                if random.random() < 0.2:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness

        if best_alpha <= 5:
            break
    
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
