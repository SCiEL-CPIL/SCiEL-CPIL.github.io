"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def _count_red_k4(colors, n):
    """Count the number of red K4^3 subhypergraphs."""
    count = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # Check all 4 triples of the 4 vertices
        triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        all_red = True
        for t in triples:
            idx = _triple_index(t, n)
            if colors[idx] != 0:
                all_red = False
                break
        if all_red:
            count += 1
    return count

def _count_blue_k5(colors, n):
    """Count the number of blue K5^3 subhypergraphs."""
    count = 0
    for quint in combinations(range(n), 5):
        a, b, c, d, e = quint
        # Check all 10 triples of the 5 vertices
        triples = [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), 
                   (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]
        all_blue = True
        for t in triples:
            idx = _triple_index(t, n)
            if colors[idx] != 1:
                all_blue = False
                break
        if all_blue:
            count += 1
    return count

def _triple_index(triple, n):
    """Compute the lexicographic index of a triple (a,b,c) with a<b<c."""
    a, b, c = triple
    # Number of triples with first vertex < a
    count = 0
    for i in range(a):
        count += (n - i - 1) * (n - i - 2) // 2
    # Number of triples with first vertex = a and second vertex < b
    for j in range(a + 1, b):
        count += (n - j - 1)
    # Number of triples with first vertex = a, second vertex = b, and third vertex < c
    count += (c - b - 1)
    return count

def _validate(colors, n):
    """Check if the coloring is valid (no red K4^3, no blue K5^3)."""
    if _count_red_k4(colors, n) > 0:
        return False
    if _count_blue_k5(colors, n) > 0:
        return False
    return True

def _local_search(colors, n, deadline):
    """Perform local search to eliminate violations."""
    colors = list(colors)
    start_time = time.time()
    
    while time.time() < deadline - 5:
        # Count violations
        red_violations = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            red_count = sum(1 for t in triples if colors[_triple_index(t, n)] == 0)
            if red_count == 4:
                red_violations.append(quad)
        
        blue_violations = []
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            triples = [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), 
                       (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]
            blue_count = sum(1 for t in triples if colors[_triple_index(t, n)] == 1)
            if blue_count == 10:
                blue_violations.append(quint)
        
        if not red_violations and not blue_violations:
            return colors, True
        
        # Pick a random violation and try to fix it
        if red_violations and (not blue_violations or random.random() < 0.5):
            quad = random.choice(red_violations)
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            # Try flipping one of the red triples to blue
            for t in triples:
                idx = _triple_index(t, n)
                if colors[idx] == 0:
                    colors[idx] = 1
                    # Check if this helps
                    if _count_red_k4(colors, n) < len(red_violations):
                        break
                    else:
                        colors[idx] = 0
        elif blue_violations:
            quint = random.choice(blue_violations)
            a, b, c, d, e = quint
            triples = [(a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), 
                       (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)]
            # Try flipping one of the blue triples to red
            for t in triples:
                idx = _triple_index(t, n)
                if colors[idx] == 1:
                    colors[idx] = 0
                    # Check if this helps
                    if _count_blue_k5(colors, n) < len(blue_violations):
                        break
                    else:
                        colors[idx] = 1
    
    return colors, _validate(colors, n)

def construct():
    """Construct a valid coloring maximizing n."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    best_n = 0
    best_colors = None
    
    # Try increasing n from a reasonable starting point
    for n in range(10, 35):
        if time.time() > deadline - 10:
            break
        
        # Initialize with a heuristic coloring
        colors = [0 if (a + b + c) % 3 == 0 else 1 
                  for a, b, c in combinations(range(n), 3)]
        
        # Check if already valid
        if _validate(colors, n):
            best_n = n
            best_colors = colors
            continue
        
        # Try local search
        improved_colors, valid = _local_search(colors, n, deadline)
        
        if valid:
            best_n = n
            best_colors = improved_colors
        
        # If we found a valid coloring, try to extend
        if best_n >= n and n < 34:
            # Try adding one more vertex
            n2 = n + 1
            if time.time() > deadline - 10:
                break
            
            # Extend the coloring
            new_colors = list(improved_colors)
            # Color new triples involving vertex n
            for a in range(n):
                for b in range(a + 1, n):
                    idx = _triple_index((a, b, n), n2)
                    # Use a heuristic: color based on sum mod 3
                    new_colors.append(0 if (a + b + n) % 3 == 0 else 1)
            
            # Try to fix violations with local search
            fixed_colors, valid2 = _local_search(new_colors, n2, deadline)
            if valid2:
                best_n = n2
                best_colors = fixed_colors
    
    if best_colors is None:
        # Fallback to the original simple construction
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 
                       for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
