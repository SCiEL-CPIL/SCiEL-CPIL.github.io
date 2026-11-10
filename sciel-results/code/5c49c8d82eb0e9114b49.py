"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    N = 1000
    
    def greedy_init():
        chosen = set()
        counts = {}
        for x in range(1, N + 1):
            valid = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            if valid:
                if counts.get(2 * x, 0) >= 2:
                    valid = False
            
            if valid:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen.add(x)
        return chosen, counts

    def local_search(chosen, counts, steps=100000):
        # chosen: set, counts: dict
        # We modify these in place
        
        def is_add_valid(x):
            # x not in chosen
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    return False
            if counts.get(2 * x, 0) >= 2:
                return False
            return True

        def do_add(x):
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            chosen.add(x)

        def do_remove(x):
            for y in chosen:
                if y == x:
                    continue
                s = x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            counts[2 * x] -= 1
            if counts[2 * x] == 0:
                del counts[2 * x]
            chosen.remove(x)

        list_chosen = list(chosen)
        
        for _ in range(steps):
            # Try to add a random element
            b = random.randint(1, N)
            if b not in chosen:
                if is_add_valid(b):
                    do_add(b)
                    list_chosen.append(b)
                    continue
            
            # Try to swap
            if len(chosen) > 0:
                a = random.choice(list_chosen)
                b = random.randint(1, N)
                if b not in chosen:
                    # Remove a
                    do_remove(a)
                    list_chosen.remove(a)
                    
                    if is_add_valid(b):
                        do_add(b)
                        list_chosen.append(b)
                    else:
                        # Revert
                        do_add(a)
                        list_chosen.append(a)

        return chosen

    best_size = 0
    best_result = []
    
    # Strategy 1: Greedy + Local Search
    init_set, init_counts = greedy_init()
    ls_set = set(init_set)
    ls_counts = dict(init_counts)
    
    ls_result = local_search(ls_set, ls_counts, steps=100000)
    
    if len(ls_result) > best_size:
        best_size = len(ls_result)
        best_result = list(ls_result)
        
    # Strategy 2: Random Start + Local Search
    random.seed(123)
    sparse_set = set()
    sparse_counts = {}
    
    # Use a helper to add to sparse set
    def sparse_add(x):
        for y in sparse_set:
            s = x + y
            if sparse_counts.get(s, 0) >= 2:
                return False
        if sparse_counts.get(2 * x, 0) >= 2:
            return False
        for y in sparse_set:
            s = x + y
            sparse_counts[s] = sparse_counts.get(s, 0) + 1
        sparse_counts[2 * x] = sparse_counts.get(2 * x, 0) + 1
        sparse_set.add(x)
        return True

    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if len(sparse_set) >= 50:
            break
        sparse_add(x)
        
    ls_result2 = local_search(sparse_set, sparse_counts, steps=100000)
    
    if len(ls_result2) > best_size:
        best_size = len(ls_result2)
        best_result = list(ls_result2)

    return best_result

# EVOLVE-BLOCK-END
