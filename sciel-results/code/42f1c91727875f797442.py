import random
import time
import math

_POP_COUNTS = None
_MASKS_BY_POPCOUNT = None

def _init_popcounts():
    global _POP_COUNTS, _MASKS_BY_POPCOUNT
    if _POP_COUNTS is None:
        _POP_COUNTS = [0] * (1 << 16)
        _MASKS_BY_POPCOUNT = [[] for _ in range(17)]
        for m in range(1 << 16):
            # Use bit_count for speed (Python 3.10+)
            # Fallback for older versions
            try:
                p = m.bit_count()
            except AttributeError:
                p = bin(m).count('1')
            _POP_COUNTS[m] = p
            _MASKS_BY_POPCOUNT[p].append(m)

def _independence_number(triples):
    _init_popcounts()
    if not triples:
        return 16
    
    triple_masks = []
    for a, b, c in triples:
        triple_masks.append((1 << a) | (1 << b) | (1 << c))
    
    # Check from largest popcount down
    for pop in range(16, -1, -1):
        for mask in _MASKS_BY_POPCOUNT[pop]:
            # Check if mask contains any triple
            # Early exit if found
            independent = True
            for tm in triple_masks:
                if (mask & tm) == tm:
                    independent = False
                    break
            if independent:
                return pop
    return 0

def _is_valid_addition(triples, new_triple):
    a, b, c = new_triple
    m = (1 << a) | (1 << b) | (1 << c)
    for t in triples:
        tm = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        # Check overlap > 1
        if (m & tm).bit_count() > 1:
            return False
    return True

def _random_valid_triple(existing_triples, n=16):
    points = list(range(n))
    for _ in range(50): # Reduced attempts per call to speed up failure
        triple = random.sample(points, 3)
        triple.sort()
        if _is_valid_addition(existing_triples, triple):
            return triple
    return None

def _greedy_start(n=16, seeds=10):
    best_triples = []
    best_score = 100
    for seed in range(seeds):
        random.seed(seed)
        t_list = []
        # Try to build a large packing
        for _ in range(200):
            t = _random_valid_triple(t_list, n)
            if t is not None:
                t_list.append(t)
            else:
                # If we can't add more, break? Or try a few more?
                # _random_valid_triple already tries 50 times.
                # If it fails, it's likely stuck or near max.
                # We can break to save time.
                break
        
        score = _independence_number(t_list)
        if score < best_score:
            best_score = score
            best_triples = t_list[:]
    return best_triples

def construct():
    n = 16
    start_time = time.time()
    time_limit = 110
    
    # 1. Get a good initial solution via greedy
    current_triples = _greedy_start(n, seeds=10)
    current_score = _independence_number(current_triples)
    best_triples = current_triples[:]
    best_score = current_score
    
    # 2. Simulated Annealing
    # We will run SA for a duration, then restart if no improvement for a while
    # Or just run SA until time limit.
    # Given the speedup in _independence_number, we can do more iterations.
    
    temp = 5.0
    cooling = 0.9999
    last_improvement_time = start_time
    no_improve_threshold = 10.0 # seconds
    
    while time.time() - start_time < time_limit:
        # Move
        move_type = random.random()
        
        if move_type < 0.7 and len(current_triples) < 60:
            # Add
            t = _random_valid_triple(current_triples, n)
            if t is not None:
                current_triples.append(t)
                new_score = _independence_number(current_triples)
                if new_score <= current_score or random.random() < math.exp(-(new_score - current_score) / max(temp, 0.01)):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_triples = current_triples[:]
                        last_improvement_time = time.time()
                else:
                    current_triples.pop()
        elif move_type < 0.9 and current_triples:
            # Remove
            idx = random.randint(0, len(current_triples) - 1)
            removed = current_triples.pop(idx)
            new_score = _independence_number(current_triples)
            if new_score <= current_score or random.random() < math.exp(-(new_score - current_score) / max(temp, 0.01)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_triples = current_triples[:]
                    last_improvement_time = time.time()
            else:
                current_triples.insert(idx, removed)
        else:
            # Swap
            if current_triples:
                idx = random.randint(0, len(current_triples) - 1)
                removed = current_triples.pop(idx)
                t = _random_valid_triple(current_triples, n)
                if t is not None:
                    current_triples.append(t)
                    new_score = _independence_number(current_triples)
                    if new_score <= current_score or random.random() < math.exp(-(new_score - current_score) / max(temp, 0.01)):
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_triples = current_triples[:]
                            last_improvement_time = time.time()
                    else:
                        current_triples.pop()
                        current_triples.insert(idx, removed)
                else:
                    current_triples.insert(idx, removed)
        
        temp *= cooling
        
        # Restart logic: if stuck, perturb or restart
        if time.time() - last_improvement_time > no_improve_threshold:
            # Perturb: randomly remove some triples and re-greedy?
            # Or just reset temperature and try again?
            # Let's try a "kick": remove 5 random triples
            for _ in range(5):
                if current_triples:
                    idx = random.randint(0, len(current_triples) - 1)
                    current_triples.pop(idx)
            # Re-evaluate
            current_score = _independence_number(current_triples)
            temp = 5.0 # Reset temp
            last_improvement_time = time.time()
            
    return best_triples
