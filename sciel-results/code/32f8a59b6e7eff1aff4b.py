def construct():
    import random
    from math import gcd
    from itertools import combinations
    
    # Precompute gcd for all pairs
    n = 128
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)
    
    # Check if a set is valid (no bad triples)
    def is_valid(subset):
        if len(subset) < 3:
            return True
        subset_list = sorted(subset)
        for i in range(len(subset_list)):
            for j in range(i + 1, len(subset_list)):
                for k in range(j + 1, len(subset_list)):
                    a, b, c = subset_list[i], subset_list[j], subset_list[k]
                    if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                        return False
        return True
    
    # Count bad triples in a set
    def count_bad_triples(subset):
        if len(subset) < 3:
            return 0
        subset_list = sorted(subset)
        count = 0
        for i in range(len(subset_list)):
            for j in range(i + 1, len(subset_list)):
                for k in range(j + 1, len(subset_list)):
                    a, b, c = subset_list[i], subset_list[j], subset_list[k]
                    if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                        count += 1
        return count
    
    # Simulated annealing
    random.seed(42)
    
    # Start with a random valid set
    current = set()
    candidates = list(range(1, n + 1))
    random.shuffle(candidates)
    for x in candidates:
        test_set = current | {x}
        if is_valid(test_set):
            current = test_set
    
    best = current.copy()
    best_size = len(current)
    
    # Annealing parameters
    temp = 10.0
    cooling = 0.999
    min_temp = 0.01
    max_iter = 5000
    
    iteration = 0
    while temp > min_temp and iteration < max_iter:
        iteration += 1
        # Propose a move: add, remove, or swap an element
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            # Try adding a random element not in current
            not_in_current = [x for x in range(1, n + 1) if x not in current]
            if not not_in_current:
                continue
            x = random.choice(not_in_current)
            new_set = current | {x}
            delta_bad = count_bad_triples(new_set) - count_bad_triples(current)
            delta_size = len(new_set) - len(current)
        elif move_type == 'remove':
            # Try removing a random element from current
            if not current:
                continue
            x = random.choice(list(current))
            new_set = current - {x}
            delta_bad = count_bad_triples(new_set) - count_bad_triples(current)
            delta_size = len(new_set) - len(current)
        else:  # swap
            # Try swapping: remove one, add another
            if not current:
                continue
            x = random.choice(list(current))
            not_in_current = [y for y in range(1, n + 1) if y not in current]
            if not not_in_current:
                continue
            y = random.choice(not_in_current)
            new_set = (current - {x}) | {y}
            delta_bad = count_bad_triples(new_set) - count_bad_triples(current)
            delta_size = len(new_set) - len(current)
        
        # Acceptance criterion: prefer valid sets with larger size
        # Score = -1000 * bad_triples + size
        old_score = -1000 * count_bad_triples(current) + len(current)
        new_score = -1000 * count_bad_triples(new_set) + len(new_set)
        delta_score = new_score - old_score
        
        if delta_score >= 0:
            current = new_set
            if len(current) > best_size:
                best = current.copy()
                best_size = len(current)
        else:
            # Accept with probability exp(delta_score / temp)
            import math
            if random.random() < math.exp(delta_score / temp):
                current = new_set
        
        temp *= cooling
    
    return list(best)
