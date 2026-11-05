import random
import time
from itertools import combinations

def _get_max_independent_set(triples):
    """
    Returns (alpha, set_S) where S is a max independent set.
    """
    n = 16
    if not triples:
        return n, set(range(n))
        
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
    points = list(range(n))
    
    for k in range(n, 0, -1):
        for combo in combinations(points, k):
            mask = 0
            for p in combo:
                mask |= (1 << p)
            
            contains_triple = False
            for tm in triple_masks:
                if (mask & tm) == tm:
                    contains_triple = True
                    break
            
            if not contains_triple:
                return k, set(combo)
                
    return 0, set()

def _is_valid_addition(triples, new_triple):
    """Check if new_triple can be added to maintain linearity."""
    a, b, c = new_triple
    set_new = {a, b, c}
    for t in triples:
        overlap = len(set_new & set(t))
        if overlap >= 2:
            return False
    return True

def construct():
    random.seed(42)
    n = 16
    start_time = time.time()
    time_limit = 115
    
    # 1. Greedy Construction
    triples = []
    all_triples = list(combinations(range(n), 3))
    random.shuffle(all_triples)
    
    for t in all_triples:
        if _is_valid_addition(triples, t):
            triples.append(t)
            
    current_alpha, current_set = _get_max_independent_set(triples)
    best_alpha = current_alpha
    best_triples = triples[:]
    
    # 2. Local Search with Targeted Moves
    temp = 1.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Find current max independent set
        current_alpha, current_set = _get_max_independent_set(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            temp = 1.0 # Reset temperature on improvement
            
        # If alpha is 1, we are done (perfect covering? No, alpha=1 means no 2 points are free? 
        # Actually alpha=1 means every pair is in a triple? No, alpha is size of set with NO triple.
        # If alpha=1, any set of size 2 contains a triple? Impossible for 3-uniform.
        # Min alpha for 3-uniform on 16 points?
        # If we have an STS(15) + 1 point, alpha is likely 4-6.
        
        # Targeted Move: Find a triple contained in current_set
        # We want to add a triple T such that T is subset of current_set.
        # This will break current_set's independence.
        
        # Generate all triples from current_set
        if len(current_set) >= 3:
            set_list = list(current_set)
            candidate_triples = list(combinations(set_list, 3))
            random.shuffle(candidate_triples)
            
            move_made = False
            for t in candidate_triples:
                # Check if t is already in triples
                if t in triples:
                    continue
                    
                # Check if t is valid to add
                if _is_valid_addition(triples, t):
                    # Add t
                    triples.append(t)
                    
                    # To keep the number of triples constant (optional, but good for stability),
                    # we might remove a random triple. 
                    # However, adding a triple strictly decreases or keeps alpha same.
                    # It never increases alpha.
                    # So we should just add it!
                    # But we might hit the max packing limit.
                    # If we hit max packing, we can't add more.
                    # So we should swap: remove a random triple, add t.
                    
                    # Let's try adding first. If it works, great.
                    # If we are at max packing, we need to swap.
                    
                    # For now, let's just add it. If we can't add more valid triples, 
                    # we will start swapping.
                    
                    new_alpha, new_set = _get_max_independent_set(triples)
                    if new_alpha < current_alpha:
                        current_alpha = new_alpha
                        move_made = True
                        break
                    else:
                        # Alpha didn't decrease? 
                        # Adding a triple inside S should decrease alpha or keep it same.
                        # If it keeps it same, S is still independent? 
                        # If T subset S, and T is added, S is no longer independent.
                        # So there must be a new max independent set S' with size <= alpha.
                        # It could be that alpha remains the same but the set changes.
                        # In that case, we didn't improve the score.
                        # We should undo this move if it doesn't improve score, 
                        # unless we accept worse states for SA.
                        
                        # Let's undo for now to keep best_triples pure.
                        triples.pop()
                        # Try next candidate
                        continue
                else:
                    # t is not valid. 
                    # We could try to remove a triple that conflicts with t, then add t.
                    # This is complex.
                    pass
            
            if move_made:
                temp = 1.0
                continue
                
        # If no targeted move worked, do a random swap
        if triples:
            idx = random.randint(0, len(triples) - 1)
            removed = triples.pop(idx)
            
            # Find a random valid triple
            new_t = None
            for _ in range(50):
                t = random.sample(range(n), 3)
                t.sort()
                if _is_valid_addition(triples, t):
                    new_t = t
                    break
                    
            if new_t:
                triples.append(new_t)
                new_alpha, _ = _get_max_independent_set(triples)
                
                if new_alpha < current_alpha:
                    current_alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_triples = triples[:]
                        temp = 1.0
                else:
                    # Reject
                    triples.pop()
                    triples.insert(idx, removed)
            else:
                triples.insert(idx, removed)
                
        temp *= cooling

    return best_triples
