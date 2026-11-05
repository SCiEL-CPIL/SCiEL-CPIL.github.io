import random
import time

def _independence_number(triples):
    """Compute the largest subset of {0..15} containing no triple."""
    n = 16
    # For each subset, check if it contains any triple
    # Represent subset as bitmask
    # A subset contains triple [a,b,c] if mask has bits a, b, c set
    # We want max popcount(mask) such that no triple is subset of mask
    
    # Precompute triple masks
    triple_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
    
    best = 0
    # Iterate all masks
    for mask in range(1 << n):
        # Check if mask contains any triple
        contains_triple = False
        for tm in triple_masks:
            if (mask & tm) == tm:
                contains_triple = True
                break
        if not contains_triple:
            pop = bin(mask).count('1')
            if pop > best:
                best = pop
    return best

def _is_valid_addition(triples, new_triple):
    """Check if new_triple can be added to maintain linearity."""
    a, b, c = new_triple
    for t in triples:
        overlap = len(set([a, b, c]) & set(t))
        if overlap >= 2:
            return False
    return True

def _random_valid_triple(existing_triples, n=16):
    """Try to find a random triple that can be added."""
    points = list(range(n))
    for _ in range(100):
        triple = random.sample(points, 3)
        triple.sort()
        if _is_valid_addition(existing_triples, triple):
            return triple
    return None

def construct():
    random.seed(42)
    n = 16
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer
    
    # Start with a random packing
    triples = []
    for _ in range(200):
        t = _random_valid_triple(triples, n)
        if t is not None:
            triples.append(t)
    
    current_score = _independence_number(triples)
    best_triples = triples[:]
    best_score = current_score
    
    # Simulated annealing
    temp = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.5 and len(triples) < 50:
            # Add a triple
            t = _random_valid_triple(triples, n)
            if t is not None:
                triples.append(t)
                new_score = _independence_number(triples)
                if new_score <= current_score or random.random() < (1.0 / (1.0 + temp)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = triples[:]
                else:
                    triples.pop()
        elif move_type < 0.8 and triples:
            # Remove a triple
            idx = random.randint(0, len(triples) - 1)
            removed = triples.pop(idx)
            new_score = _independence_number(triples)
            if new_score <= current_score or random.random() < (1.0 / (1.0 + temp)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_triples = triples[:]
            else:
                triples.insert(idx, removed)
        else:
            # Swap: remove one, try to add another
            if triples:
                idx = random.randint(0, len(triples) - 1)
                removed = triples.pop(idx)
                t = _random_valid_triple(triples, n)
                if t is not None:
                    triples.append(t)
                    new_score = _independence_number(triples)
                    if new_score <= current_score or random.random() < (1.0 / (1.0 + temp)):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_triples = triples[:]
                    else:
                        triples.pop()
                        triples.insert(idx, removed)
                else:
                    triples.insert(idx, removed)
        
        temp *= cooling
    
    return best_triples
