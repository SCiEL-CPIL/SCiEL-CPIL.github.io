import time
import itertools
import random
import sys

def _has_independent_set_size(triples_masks, size, n=16):
    """
    Returns True if there exists an independent set of the given size.
    triples_masks: list of integer masks representing triples.
    """
    for combo in itertools.combinations(range(n), size):
        s_mask = 0
        for v in combo:
            s_mask |= (1 << v)
        
        is_independent = True
        for t_mask in triples_masks:
            if (s_mask & t_mask) == t_mask:
                is_independent = False
                break
        
        if is_independent:
            return True
            
    return False

def _independence_number(triples):
    if not triples:
        return 16
        
    masks = []
    for t in triples:
        m = 0
        for p in t:
            m |= (1 << p)
        masks.append(m)
        
    for r in range(16, 0, -1):
        if _has_independent_set_size(masks, r):
            return r
    return 0

def _greedy_construct(seed):
    random.seed(seed)
    V = list(range(16))
    all_triples = list(itertools.combinations(V, 3))
    random.shuffle(all_triples)
    
    used_pairs = set()
    triples = []
    
    for t in all_triples:
        ok = True
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair in used_pairs:
                ok = False
                break
        if ok:
            triples.append(list(t))
            for a, b in itertools.combinations(t, 2):
                used_pairs.add((min(a, b), max(a, b)))
                
    return triples

def _is_linear(triples):
    used_pairs = set()
    for t in triples:
        for a, b in itertools.combinations(t, 2):
            pair = (min(a, b), max(a, b))
            if pair in used_pairs:
                return False
            used_pairs.add(pair)
    return True

def _csp_construct(seed, deadline):
    """
    CSP-based backtracking solver with forward checking.
    Variables: points 0..15
    Constraints: no three points form a hyperedge (triple)
    Goal: construct a linear triple system with small independence number
    """
    random.seed(seed)
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(range(16), 3))
    
    # State: used_pairs set, list of triples
    used_pairs = set()
    triples = []
    
    # Order triples by some heuristic - maybe by how "central" they are
    # For now, just shuffle
    shuffled_triples = all_triples[:]
    random.shuffle(shuffled_triples)
    
    # Backtracking with forward checking
    # We'll try to build a system that covers all pairs (or as many as possible)
    # while maintaining linearity
    
    def backtrack(remaining_triples, depth=0):
        if time.time() > deadline:
            return None
        if not remaining_triples:
            return triples[:]
        
        # Forward checking: for each remaining triple, check if it can be added
        valid_triples = []
        for t in remaining_triples:
            can_add = True
            for a, b in itertools.combinations(t, 2):
                pair = (min(a, b), max(a, b))
                if pair in used_pairs:
                    can_add = False
                    break
            if can_add:
                valid_triples.append(t)
        
        if not valid_triples:
            return triples[:] if triples else None
        
        # Choose the next triple to add
        # Heuristic: choose the one that constrains the most future choices
        # For simplicity, pick randomly from valid
        idx = random.randint(0, len(valid_triples) - 1)
        chosen = valid_triples[idx]
        
        # Add the chosen triple
        triples.append(chosen)
        for a, b in itertools.combinations(chosen, 2):
            used_pairs.add((min(a, b), max(a, b)))
        
        # Recurse
        remaining = [t for t in remaining_triples if t != chosen]
        result = backtrack(remaining, depth + 1)
        
        # Backtrack
        triples.pop()
        for a, b in itertools.combinations(chosen, 2):
            used_pairs.discard((min(a, b), max(a, b)))
        
        return result
    
    # Try to build a complete system
    result = backtrack(shuffled_triples)
    return result if result else triples

def construct():
    start = time.time()
    DEADLINE = 115.0
    
    best_alpha = 16
    best_sys = []
    
    # Initial best with greedy
    sys_candidate = _greedy_construct(0)
    alpha = _independence_number(sys_candidate)
    best_alpha = alpha
    best_sys = sys_candidate
    
    if best_alpha == 5:
        return best_sys
        
    seed = 1
    while time.time() - start < DEADLINE:
        # Try CSP-based construction
        sys_candidate = _csp_construct(seed, DEADLINE - 5)
        
        if sys_candidate and _is_linear(sys_candidate):
            current_alpha = _independence_number(sys_candidate)
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_sys = sys_candidate
                if best_alpha == 5:
                    return best_sys
            
            # Local search on the CSP result
            improved = True
            while improved and (time.time() - start < DEADLINE - 5):
                improved = False
                
                # Try to add a triple that reduces independence number
                used_pairs = set()
                for t in sys_candidate:
                    for a, b in itertools.combinations(t, 2):
                        pair = (min(a, b), max(a, b))
                        used_pairs.add(pair)
                
                # Find all valid triples that can be added
                valid_adds = []
                for t in itertools.combinations(range(16), 3):
                    t = tuple(t)
                    if t in sys_candidate:
                        continue
                    can_add = True
                    for a, b in itertools.combinations(t, 2):
                        pair = (min(a, b), max(a, b))
                        if pair in used_pairs:
                            can_add = False
                            break
                    if can_add:
                        valid_adds.append(t)
                
                # Try each valid addition
                random.shuffle(valid_adds)
                for t in valid_adds[:20]:  # Try up to 20 random additions
                    if time.time() - start > DEADLINE - 5:
                        break
                    
                    new_sys = sys_candidate[:] + [list(t)]
                    new_alpha = _independence_number(new_sys)
                    
                    if new_alpha < current_alpha:
                        sys_candidate = new_sys
                        current_alpha = new_alpha
                        improved = True
                        
                        if current_alpha < best_alpha:
                            best_alpha = current_alpha
                            best_sys = sys_candidate
                            if best_alpha == 5:
                                return best_sys
                        break
        
        # Also try greedy with different seeds
        sys_candidate = _greedy_construct(seed)
        current_alpha = _independence_number(sys_candidate)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_sys = sys_candidate
            if best_alpha == 5:
                return best_sys
                
        seed += 1
        
    return best_sys
