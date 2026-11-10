"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_valid(seq):
    products = set()
    n = len(seq)
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= seq[j]
            if prod in products:
                return False
            products.add(prod)
    return True

def _greedy_construction():
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _randomized_greedy(seed):
    random.seed(seed)
    candidates = list(range(2, 129))
    random.shuffle(candidates)
    values, products = [], set()
    for x in candidates:
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values

def _improve_by_insertion(seq, deadline):
    seq = list(seq)
    used = set(seq)
    missing = [x for x in range(2, 129) if x not in used]
    
    # We will try to insert numbers.
    # To avoid getting stuck in a local minimum where a specific choice of insertion
    # blocks others, we can try randomizing the order of 'missing' in each pass.
    
    while time.time() < deadline:
        improved = False
        # Randomize order of missing to explore different paths
        random.shuffle(missing)
        
        for m in missing:
            # Randomize positions too?
            positions = list(range(len(seq) + 1))
            random.shuffle(positions)
            
            for i in positions:
                new_seq = seq[:i] + [m] + seq[i:]
                if _is_valid(new_seq):
                    seq = new_seq
                    used.add(m)
                    missing.remove(m)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # If no improvement in a full randomized pass, we are likely stuck.
            # But maybe a different random order would find it?
            # Unlikely if the set of valid insertions is empty.
            # However, if the set is non-empty but we just didn't pick the "right" one
            # to maximize future potential, randomizing helps.
            # But if we did a full pass and found NO valid insertion, then no insertion is valid.
            # So we break.
            break
            
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Generate initial candidates
    candidates = []
    candidates.append(_greedy_construction())
    
    # Generate randomized candidates
    for i in range(100):
        if time.time() > deadline:
            break
        candidates.append(_randomized_greedy(i))
        
    # Sort by length descending
    candidates.sort(key=len, reverse=True)
    
    # Improve the top few candidates
    best_seq = candidates[0]
    best_len = len(best_seq)
    
    # Try to improve the top 5 distinct length sequences?
    # Or just the top 10.
    seen_lengths = set()
    for seq in candidates[:10]:
        if time.time() > deadline:
            break
        # Skip if we already processed a sequence of this length? 
        # No, different sequences of same length might improve differently.
        # But it's expensive. Let's just do the top 5.
        
        improved_seq = _improve_by_insertion(seq, deadline)
        if len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(best_seq)
            
    return best_seq
# EVOLVE-BLOCK-END
