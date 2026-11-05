import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit
    
    # 1. Generate an STS(15)
    # Points 0..14
    # We need 35 triples.
    # Backtracking search for STS(15)
    
    def generate_sts_15():
        points = list(range(15))
        pairs_to_cover = set(frozenset(p) for p in itertools.combinations(points, 2))
        triples = []
        
        def backtrack(remaining_pairs):
            if not remaining_pairs:
                return True
            if time.time() > deadline - 5.0:
                return False
                
            # Pick a pair to cover
            # Heuristic: pick a pair with fewest available triples? 
            # For simplicity, pick the first one.
            pair = next(iter(remaining_pairs))
            
            # Generate all possible triples containing this pair
            a, b = tuple(pair)
            candidates = []
            for c in range(15):
                if c == a or c == b:
                    continue
                t = frozenset([a, b, c])
                # Check if t is compatible with existing triples
                # t must not share 2 points with any existing triple
                compatible = True
                for existing in triples:
                    if len(t & existing) > 1:
                        compatible = False
                        break
                if compatible:
                    # Check if t covers only pairs in remaining_pairs?
                    # No, t covers 3 pairs. All 3 must be in remaining_pairs.
                    p1 = frozenset([a, b])
                    p2 = frozenset([a, c])
                    p3 = frozenset([b, c])
                    if p1 in remaining_pairs and p2 in remaining_pairs and p3 in remaining_pairs:
                        candidates.append(t)
                        
            random.shuffle(candidates)
            
            for t in candidates:
                triples.append(t)
                new_remaining = remaining_pairs - {frozenset([a, b]), frozenset([a, list(t)[0] if list(t)[0]!=a else list(t)[1]]), frozenset([b, list(t)[0] if list(t)[0]!=b else list(t)[1]])}
                # Correctly remove the 3 pairs of t
                p1, p2, p3 = [frozenset(p) for p in itertools.combinations(list(t), 2)]
                new_remaining = remaining_pairs - {p1, p2, p3}
                
                if backtrack(new_remaining):
                    return True
                
                triples.pop()
            
            return False
        
        if backtrack(pairs_to_cover):
            return [list(t) for t in triples]
        return None

    sts_15 = generate_sts_15()
    if not sts_15:
        # Fallback: return disjoint triples
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

    best_system = None
    best_alpha = 16
    
    # 2. Try to modify STS(15) to add point 15 and reduce alpha
    # We will try removing k triples and adding k triples with 15
    for k in range(1, 8):
        if time.time() > deadline:
            break
            
        # Randomly select k triples to remove
        # We need to ensure the added triples are linear
        for _ in range(100):  # 100 attempts per k
            if time.time() > deadline:
                break
                
            removed_indices = random.sample(range(len(sts_15)), k)
            removed_triples = [sts_15[i] for i in removed_indices]
            
            new_triples = [t for i, t in enumerate(sts_15) if i not in removed_indices]
            
            # For each removed triple, add a triple with 15
            # We have 3 choices per removed triple
            # Let's try a few random combinations
            for attempt in range(10):
                if time.time() > deadline:
                    break
                    
                candidate_triples = list(new_triples)
                valid = True
                for rt in removed_triples:
                    # Pick a random pair from rt to form new triple with 15
                    pair = random.sample(rt, 2)
                    new_t = [15, pair[0], pair[1]]
                    candidate_triples.append(new_t)
                
                # Check linearity
                if not is_linear(candidate_triples):
                    continue
                    
                # Check independence number
                # We want to know if alpha <= 8
                # This is equivalent to: no independent set of size 9
                alpha_9_exists = has_independent_set_size_9(candidate_triples)
                
                if not alpha_9_exists:
                    # alpha <= 8
                    return candidate_triples
                else:
                    # Check if alpha is better than best found
                    # We don't know exact alpha, but we know it's >= 9
                    # We can try to find if alpha <= 9? 
                    # Actually, if we find one with alpha 9, it's better than 11.
                    # But we want the best.
                    # Let's just keep the first valid one if we can't find alpha 8 quickly?
                    # No, we should search for the best.
                    # However, checking exact alpha is expensive.
                    # We can estimate by finding max independent set size.
                    # For now, if we find alpha 9, we store it.
                    if best_system is None:
                        best_system = candidate_triples
                        best_alpha = 9 # Assuming it's 9, might be higher

    if best_system:
        return best_system
        
    # Fallback
    return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]

def is_linear(triples):
    pair_count = {}
    for t in triples:
        for p in itertools.combinations(t, 2):
            key = frozenset(p)
            pair_count[key] = pair_count.get(key, 0) + 1
            if pair_count[key] > 1:
                return False
    return True

def has_independent_set_size_9(triples):
    # Check if there is a subset of 9 points containing no triple
    triples_set = set(frozenset(t) for t in triples)
    points = range(16)
    
    for subset in itertools.combinations(points, 9):
        subset_f = frozenset(subset)
        independent = True
        for t in triples_set:
            if t.issubset(subset_f):
                independent = False
                break
        if independent:
            return True
    return False
