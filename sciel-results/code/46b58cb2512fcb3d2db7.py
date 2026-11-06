import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    
    # Precompute disjointness masks or just use set membership in the check
    
    def is_valid_addition(t, present_set):
        """
        Returns True if adding t to present_set does NOT create a K(2,2,2).
        Checks all valid (disjoint) K(2,2,2) configurations containing t.
        """
        a, b, c = t
        # To speed up, we can precompute valid partners?
        # For now, inline loops with disjointness checks.
        
        # We need to find a', b', c' such that:
        # 1. {a, a'}, {b, b'}, {c, c'} are disjoint pairs.
        #    This implies a' != a, b' != b, c' != c.
        #    And a' not in {b, c}, b' not in {a, c, a'}, c' not in {a, b, a', b'}.
        #    Actually, disjointness of pairs means:
        #    a' != b, a' != c
        #    b' != a, b' != c, b' != a'
        #    c' != a, c' != b, c' != a', c' != b'
        
        for a2 in range(n):
            if a2 == a or a2 == b or a2 == c:
                continue
            for b2 in range(n):
                if b2 == b or b2 == a or b2 == c or b2 == a2:
                    continue
                for c2 in range(n):
                    if c2 == c or c2 == a or c2 == b or c2 == a2 or c2 == b2:
                        continue
                    
                    # Check the other 7 transversals
                    if (a, b, c2) in present_set:
                        if (a, b2, c) in present_set:
                            if (a, b2, c2) in present_set:
                                if (a2, b, c) in present_set:
                                    if (a2, b, c2) in present_set:
                                        if (a2, b2, c) in present_set:
                                            if (a2, b2, c2) in present_set:
                                                return False
        return True

    # 1. Greedy Initialization
    random.shuffle(all_triples)
    current_set = set()
    current_list = []
    
    for t in all_triples:
        if is_valid_addition(t, current_set):
            current_set.add(t)
            current_list.append(t)
            
    best_set = set(current_set)
    best_list = list(current_list)
    best_size = len(best_list)
    
    # 2. Simulated Annealing / Local Search
    # We try to add more triples. If stuck, we try swaps.
    # Since "Add" strictly increases size, we prioritize it.
    # We maintain a list of "missing" triples to pick from.
    
    missing_triples = [t for t in all_triples if t not in current_set]
    random.shuffle(missing_triples)
    
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    while time.time() < deadline:
        if not missing_triples:
            break
            
        # Pick a candidate to add
        # To be efficient, we can try a few random candidates
        added = False
        # Try up to 10 random adds per step
        for _ in range(10):
            if not missing_triples:
                break
            idx = random.randint(0, len(missing_triples) - 1)
            t_add = missing_triples[idx]
            
            if is_valid_addition(t_add, current_set):
                # Accept
                current_set.add(t_add)
                current_list.append(t_add)
                missing_triples.pop(idx)
                added = True
                
                if len(current_list) > best_size:
                    best_size = len(current_list)
                    best_set = set(current_set)
                    best_list = list(current_list)
                break
        
        if not added:
            # Try a swap: remove one, add one
            # This is more expensive. Only do it if we have time.
            if time.time() > deadline - 5:
                break
                
            if current_list and missing_triples:
                # Pick random remove
                idx_rm = random.randint(0, len(current_list) - 1)
                t_rm = current_list[idx_rm]
                
                # Pick random add
                idx_add = random.randint(0, len(missing_triples) - 1)
                t_add = missing_triples[idx_add]
                
                # Check if swap is valid
                # Temporarily remove t_rm
                current_set.remove(t_rm)
                
                if is_valid_addition(t_add, current_set):
                    # Accept swap
                    current_set.add(t_add)
                    current_list[idx_rm] = t_add
                    missing_triples[idx_add] = t_rm
                    # Note: size remains same, so no improvement in best_size
                    # But it might enable future adds
                    
                    # Swap in missing list
                    missing_triples[idx_add] = t_rm
                else:
                    # Revert
                    current_set.add(t_rm)
                    
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Restart? Or just continue with low T
            # Given the structure, we might just continue

    return [list(t) for t in best_list]
