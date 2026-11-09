import time
from math import gcd
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    n = 128
    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def is_valid_triple(a, b, c):
        """Check if triple (a,b,c) is valid (no equal GCDs)."""
        ga = gcd_table[a][b]
        gb = gcd_table[b][c]
        gc = gcd_table[c][a]
        return not (ga == gb == gc)
    
    def can_add(x, selected_set):
        """Check if x can be added without creating a bad triple."""
        selected_list = list(selected_set)
        for i in range(len(selected_list)):
            a = selected_list[i]
            ga = gcd_table[x][a]
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                if ga == gcd_table[x][b] == gcd_table[a][b]:
                    return False
        return True
    
    def count_bad_triples(selected_set):
        """Count number of bad triples in the set."""
        selected_list = list(selected_set)
        count = 0
        for i in range(len(selected_list)):
            a = selected_list[i]
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                ga = gcd_table[a][b]
                for k in range(j + 1, len(selected_list)):
                    c = selected_list[k]
                    if ga == gcd_table[b][c] == gcd_table[c][a]:
                        count += 1
        return count
    
    def get_bad_triples(selected_set):
        """Return list of bad triples."""
        selected_list = list(selected_set)
        bad = []
        for i in range(len(selected_list)):
            a = selected_list[i]
            for j in range(i + 1, len(selected_list)):
                b = selected_list[j]
                ga = gcd_table[a][b]
                for k in range(j + 1, len(selected_list)):
                    c = selected_list[k]
                    if ga == gcd_table[b][c] == gcd_table[c][a]:
                        bad.append((a, b, c))
        return bad
    
    # Phase 1: Greedy construction with multiple orderings
    rng = random.Random(42)
    best_set = set()
    best_size = 0
    
    # Try different orderings
    orderings = [
        list(range(1, 129)),  # natural
        list(range(128, 0, -1)),  # reverse
        sorted(range(1, 129), key=lambda x: -x.bit_count()),  # by bit count
    ]
    
    for _ in range(50):
        ordering = list(range(1, 129))
        rng.shuffle(ordering)
        orderings.append(ordering)
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        selected = set()
        for x in ordering:
            if can_add(x, selected):
                selected.add(x)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected.copy()
    
    # Phase 2: Local search with swaps
    current_set = best_set.copy()
    current_size = len(current_set)
    all_numbers = set(range(1, 129))
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try to add a random element not in set
        not_in_set = all_numbers - current_set
        if not_in_set:
            candidates = list(not_in_set)
            rng.shuffle(candidates)
            for x in candidates[:20]:  # Try up to 20 candidates
                test_set = current_set | {x}
                if count_bad_triples(test_set) == 0:
                    current_set = test_set
                    current_size += 1
                    break
        
        # Try swap: remove one element, add another
        if len(current_set) > 0:
            in_set = list(current_set)
            rng.shuffle(in_set)
            not_in_set = all_numbers - current_set
            if not_in_set:
                out_candidates = list(not_in_set)
                rng.shuffle(out_candidates)
                
                for x_out in in_set[:10]:
                    for x_in in out_candidates[:20]:
                        test_set = (current_set - {x_out}) | {x_in}
                        if len(test_set) == len(current_set) and count_bad_triples(test_set) == 0:
                            # Only accept if it helps (same size but maybe better structure)
                            # For now, skip same-size swaps unless we can then add more
                            pass
    
    # Phase 3: Try to add elements one by one in various orders
    for order in [list(range(1, 129)), list(range(128, 0, -1))]:
        for x in order:
            if time.time() > deadline:
                break
            if x not in current_set:
                if can_add(x, current_set):
                    current_set.add(x)
    
    # Phase 4: More aggressive local search
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding
        not_in_set = all_numbers - current_set
        if not_in_set:
            candidates = list(not_in_set)
            rng.shuffle(candidates)
            for x in candidates[:30]:
                test_set = current_set | {x}
                if count_bad_triples(test_set) == 0:
                    current_set = test_set
                    current_size += 1
                    improved = True
                    break
        
        # Try swap: remove x_out, add x_in, then try to add more
        if not improved and len(current_set) > 0:
            in_set = list(current_set)
            rng.shuffle(in_set)
            not_in_set = all_numbers - current_set
            if not_in_set:
                out_candidates = list(not_in_set)
                rng.shuffle(out_candidates)
                
                for x_out in in_set[:15]:
                    for x_in in out_candidates[:15]:
                        if x_in in current_set:
                            continue
                        test_set = (current_set - {x_out}) | {x_in}
                        if count_bad_triples(test_set) == 0:
                            # After swap, try to add more
                            temp_set = test_set.copy()
                            temp_size = len(temp_set)
                            remaining = all_numbers - temp_set
                            for x_add in list(remaining):
                                if can_add(x_add, temp_set):
                                    temp_set.add(x_add)
                            if len(temp_set) > len(current_set):
                                current_set = temp_set
                                current_size = len(current_set)
                                improved = True
                                break
                    if improved:
                        break
    
    return sorted(current_set)
