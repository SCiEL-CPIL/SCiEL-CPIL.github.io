import time
import random
from math import gcd

def construct():
    # Strict deadline to avoid timeout
    deadline = time.time() + 55
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def is_bad_triple(a, b, c):
        return gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]

    def is_compatible(x, selected):
        """Check if x can be added to selected without creating a bad triple."""
        # selected is a set
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            for j in range(i + 1, n):
                b = sel_list[j]
                if is_bad_triple(a, b, x):
                    return False
        return True

    def is_compatible_fast(x, selected_list):
        """Faster compatibility check using list instead of set."""
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            for j in range(i + 1, n):
                b = selected_list[j]
                if is_bad_triple(a, b, x):
                    return False
        return True

    # 1. Greedy Construction: Prioritize numbers with fewer conflicts
    # Calculate degree: number of bad triples each number is part of
    degrees = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad_triple(a, b, c):
                    degrees[a] += 1
                    degrees[b] += 1
                    degrees[c] += 1
    
    # Sort numbers by degree (ascending) to pick "least problematic" first
    sorted_numbers = sorted(numbers, key=lambda x: degrees[x])
    
    selected = set()
    for x in sorted_numbers:
        if time.time() > deadline - 15:
            break
        if is_compatible(x, selected):
            selected.add(x)
    
    # Convert to list for faster access
    selected_list = list(selected)
    
    # 2. Local Search: Try to improve by adding or swapping
    last_improvement = time.time()
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline - 5:
        improved = False
        
        # Phase A: Try to add any compatible number
        for x in numbers:
            if x not in selected:
                if is_compatible_fast(x, selected_list):
                    selected.add(x)
                    selected_list.append(x)
                    improved = True
                    last_improvement = time.time()
                    no_improve_count = 0
                    break # Restart loop after addition
        
        if improved:
            continue
        
        no_improve_count += 1
        
        # Phase B: Try swaps (remove one, add one)
        # Systematically try all pairs (y in selected, x not in selected)
        # but limit based on time budget
        found_swap = False
        
        # Get current time to check budget
        if time.time() > deadline - 8:
            break
            
        # Try all y in selected, but sample x values if needed
        # For efficiency, we'll try a subset of x values
        unselected = [x for x in numbers if x not in selected]
        
        # Shuffle to randomize order
        random.shuffle(unselected)
        
        for y in selected_list:
            if time.time() > deadline - 5:
                break
            if found_swap:
                break
                
            # Temporarily remove y
            temp_list = [z for z in selected_list if z != y]
            
            for x in unselected:
                if x == y:
                    continue
                if is_compatible_fast(x, temp_list):
                    # Accept swap
                    selected.discard(y)
                    selected.add(x)
                    selected_list = temp_list + [x]
                    improved = True
                    found_swap = True
                    last_improvement = time.time()
                    no_improve_count = 0
                    break
        
        if not improved:
            # No improvement this round
            if no_improve_count > 5:
                # Try a more aggressive restart: remove a random element and rebuild
                if time.time() < deadline - 10:
                    # Remove 1-2 random elements and try to rebuild
                    if len(selected) > 10:
                        for _ in range(min(2, len(selected))):
                            y = random.choice(list(selected))
                            selected.discard(y)
                            selected_list = list(selected)
                        # Try to add back compatible elements
                        for x in sorted_numbers:
                            if x not in selected and is_compatible_fast(x, selected_list):
                                selected.add(x)
                                selected_list.append(x)
                                if time.time() > deadline - 5:
                                    break
                        no_improve_count = 0
                    else:
                        break
                else:
                    break

    return list(selected)
