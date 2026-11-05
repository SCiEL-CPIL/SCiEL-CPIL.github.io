import time
import random
from math import gcd
from itertools import combinations

def construct():
    # Strict deadline to avoid timeout
    deadline = time.time() + 50
    
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
        # For each pair (a, b) in selected, check if gcd(a,b) == gcd(a,x) == gcd(b,x)
        for a in selected:
            ga_x = gcd_table[a][x]
            for b in selected:
                if a >= b:
                    continue
                if gcd_table[a][b] == ga_x == gcd_table[b][x]:
                    return False
        return True
    
    def conflict_count(x, selected):
        """Count how many bad triples x would form with selected."""
        count = 0
        for a in selected:
            ga_x = gcd_table[a][x]
            for b in selected:
                if a >= b:
                    continue
                if gcd_table[a][b] == ga_x == gcd_table[b][x]:
                    count += 1
        return count

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
        if time.time() > deadline - 10:
            break
        if is_compatible(x, selected):
            selected.add(x)
            
    # 2. Local Search: Try to improve by adding or swapping
    # We will run this for a few seconds or until no improvement
    last_improvement = time.time()
    improved = True
    best_selected = set(selected)
    
    # Precompute for each x, the set of y's that conflict with x (i.e., form a bad triple with some pair)
    # Actually, let's just do efficient swaps
    
    while time.time() < deadline - 5:
        improved = False
        
        # Phase A: Try to add any compatible number
        for x in numbers:
            if x not in selected:
                if is_compatible(x, selected):
                    selected.add(x)
                    improved = True
                    last_improvement = time.time()
                    if len(selected) > len(best_selected):
                        best_selected = set(selected)
                    break # Restart loop after addition
        
        if improved:
            continue
        
        # Phase B: Try swaps (remove one, add one)
        # For each y in selected, compute which x's would be compatible if y were removed
        # We want to find a swap that doesn't decrease size (same size swap) or ideally we're already at max addable
        
        # Better approach: For each y in selected, try to find an x not in selected 
        # such that x is compatible with (selected - {y})
        # To make this faster, precompute for each x, which y's it conflicts with
        
        # Actually, let's try a smarter swap: find y such that removing y allows adding x
        # where x has lower "degree" or better properties
        
        found_swap = False
        selected_list = list(selected)
        random.shuffle(selected_list)
        
        for y in selected_list:
            if time.time() > deadline - 5:
                break
            if len(selected) == 0:
                break
            
            temp_selected = selected - {y}
            
            # Find best x to add: prefer x that conflicts least with temp_selected
            best_x = None
            best_conflicts = float('inf')
            
            for x in numbers:
                if x == y or x in selected:
                    continue
                # Quick check: is x compatible with temp_selected?
                if is_compatible(x, temp_selected):
                    # This is a valid swap
                    best_x = x
                    break
                else:
                    # Count conflicts to find "almost compatible" x
                    conflicts = conflict_count(x, temp_selected)
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        best_x = x
            
            if best_x is not None and best_x not in selected:
                # Check if it's actually compatible
                if is_compatible(best_x, temp_selected):
                    selected = temp_selected | {best_x}
                    improved = True
                    found_swap = True
                    last_improvement = time.time()
                    if len(selected) > len(best_selected):
                        best_selected = set(selected)
                    break
        
        if not improved:
            # Try 2-out, 2-in moves? This is more expensive but can escape deeper local optima
            if time.time() < deadline - 10:
                # Sample a few 2-out moves
                for _ in range(5):
                    if time.time() > deadline - 10:
                        break
                    if len(selected) < 2:
                        break
                    
                    y1, y2 = random.sample(list(selected), 2)
                    temp_selected = selected - {y1, y2}
                    
                    # Try to add 2 numbers
                    added = []
                    for x in numbers:
                        if x in selected or x == y1 or x == y2:
                            continue
                        if is_compatible(x, temp_selected | set(added)):
                            added.append(x)
                            if len(added) == 2:
                                break
                    
                    if len(added) == 2:
                        selected = temp_selected | set(added)
                        improved = True
                        last_improvement = time.time()
                        if len(selected) > len(best_selected):
                            best_selected = set(selected)
                        break
        
        if not improved and time.time() - last_improvement > 15:
            # Stagnation: accept best found so far
            break
    
    # Use best found
    if len(best_selected) > len(selected):
        selected = best_selected

    return list(selected) if selected else [1]
