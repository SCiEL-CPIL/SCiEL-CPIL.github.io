import time
import random
import math
from itertools import combinations

def score_set(U):
    """Compute the score for a set U."""
    if not U or 0 not in U:
        return -1
    U = set(U)
    if len(U) < 2:
        return -1
    max_u = max(U)
    if max_u == 0:
        return -1
    # Compute sumset and difference set
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return -1
    return 1.0 + math.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Try some known good sets
    candidates = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 4],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 16],
    ]
    for c in candidates:
        s = score_set(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
    
    # Local search
    random.seed(42)
    current_set = list(best_set)
    current_score = best_score
    
    max_iter = 50000
    for i in range(max_iter):
        if time.time() > deadline:
            break
        
        # Random mutation
        new_set = set(current_set)
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and len(new_set) < 12:
            # Add a new element
            max_elem = max(new_set)
            new_elem = random.randint(1, max_elem * 2 + 10)
            if new_elem not in new_set:
                new_set.add(new_elem)
        elif mutation_type == 'remove' and len(new_set) > 2:
            # Remove a random non-zero element
            elems = [x for x in new_set if x != 0]
            if elems:
                new_set.remove(random.choice(elems))
        elif mutation_type == 'replace' and len(new_set) > 2:
            # Replace a non-zero element
            elems = [x for x in new_set if x != 0]
            if elems:
                old = random.choice(elems)
                new_set.remove(old)
                max_elem = max(new_set) if new_set else 1
                new_elem = random.randint(1, max_elem * 2 + 10)
                if new_elem not in new_set:
                    new_set.add(new_elem)
        
        if len(new_set) < 2:
            continue
        
        new_score = score_set(list(new_set))
        if new_score > current_score:
            current_set = list(new_set)
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = list(new_set)
    
    return best_set

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
