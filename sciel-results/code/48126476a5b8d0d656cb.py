import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pair masks for each triple
    # A triple is (a,b,c), pairs are (a,b), (a,c), (b,c)
    # We need to check if adding a triple conflicts with existing ones
    # Two triples conflict if they share a pair (i.e., 2 or more points)
    
    # For each triple, store the set of pairs it contains
    triple_pairs = {}
    for t in all_triples:
        a, b, c = t
        triple_pairs[t] = ((a, b), (a, c), (b, c))
    
    # For each pair, store which triples contain it
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)
    
    def get_alpha(triple_list):
        """
        Compute the independence number: largest set of points containing no entire triple.
        Uses bitmask-based approach for speed.
        """
        if not triple_list:
            return 16
        
        triple_masks = []
        for t in triple_list:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check from largest size down
        for k in range(16, 0, -1):
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for t_mask in triple_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 1
    
    def get_alpha_fast(triple_set):
        """
        Faster alpha computation using precomputed triple masks.
        triple_set is a set of triple tuples.
        """
        if not triple_set:
            return 16
        
        triple_masks = []
        for t in triple_set:
            mask = 0
            for p in t:
                mask |= (1 << p)
            triple_masks.append(mask)
        
        # Check from largest size down
        for k in range(16, 0, -1):
            for subset in itertools.combinations(range(16), k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_independent = True
                for t_mask in triple_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 1
    
    # Greedy construction: build a linear triple system
    # Start with all triples, remove conflicts
    # Use a greedy approach: randomly order triples, add if no conflict
    
    random.seed(42)
    
    best_alpha = 16
    best_system = []
    
    # Try multiple random greedy constructions
    for attempt in range(200):
        if time.time() > deadline - 10:
            break
        
        # Randomly shuffle triples
        shuffled = all_triples[:]
        random.shuffle(shuffled)
        
        # Greedy: add triple if it doesn't share a pair with any existing triple
        current_system = []
        used_pairs = set()
        
        for t in shuffled:
            pairs = triple_pairs[t]
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                current_system.append(t)
                for p in pairs:
                    used_pairs.add(p)
        
        current_alpha = get_alpha_fast(set(current_system))
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = current_system[:]
        
        if best_alpha <= 4:
            break
    
    # Local search: try to improve by swapping triples
    if best_alpha > 4 and time.time() < deadline - 20:
        current_system = best_system[:]
        current_alpha = best_alpha
        used_pairs = set()
        for t in current_system:
            for p in triple_pairs[t]:
                used_pairs.add(p)
        
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5 and no_improve_count < max_no_improve:
            # Pick a random triple from current system
            idx = random.randint(0, len(current_system) - 1)
            removed = current_system[idx]
            
            # Remove its pairs from used_pairs
            for p in triple_pairs[removed]:
                used_pairs.discard(p)
            
            # Try to find a replacement triple
            improved = False
            for _ in range(100):
                candidate = random.choice(all_triples)
                if candidate == removed:
                    continue
                pairs = triple_pairs[candidate]
                conflict = False
                for p in pairs:
                    if p in used_pairs:
                        conflict = True
                        break
                if not conflict:
                    # Try adding this candidate
                    current_system[idx] = candidate
                    for p in pairs:
                        used_pairs.add(p)
                    
                    new_alpha = get_alpha_fast(set(current_system))
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        improved = True
                        no_improve_count = 0
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_system = current_system[:]
                    else:
                        # Revert
                        current_system[idx] = removed
                        for p in pairs:
                            used_pairs.discard(p)
                    break
            
            if not improved:
                # Revert the removal
                current_system[idx] = removed
                for p in triple_pairs[removed]:
                    used_pairs.add(p)
                no_improve_count += 1
    
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
