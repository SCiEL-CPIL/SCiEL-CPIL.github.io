"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations
from math import comb

def construct():
    """Construct a valid 2-coloring of triples maximizing n.
    
    Returns dict with 'n' and 'colors' (list of 0/1 for each triple in lex order).
    """
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 120s limit (evaluator kills at 150s)
    
    # Precompute triple indices for various n values
    # For n vertices, triples (a,b,c) with a<b<c are in lex order
    # Index = C(c,3) + C(b,2) + a  (combinatorial number system)
    
    def triple_index(a, b, c):
        """Compute lexicographic index of triple (a,b,c), a<b<c."""
        return comb(c, 3) + comb(b, 2) + a
    
    def generate_triples(n):
        """Generate all triples for n vertices in lex order."""
        return list(combinations(range(n), 3))
    
    def make_index_map(n):
        """Create a mapping from (a,b,c) to index for fast lookup."""
        triples = generate_triples(n)
        return {t: i for i, t in enumerate(triples)}, triples
    
    def check_validity(colors, n, triples):
        """Check if coloring is valid: no red K4^3, no blue K5^3."""
        # Precompute for each 4-set whether all 4 triples are red
        # For each 5-set whether all 10 triples are blue
        
        # Build a set of red triples and blue triples for fast lookup
        red_set = set()
        blue_set = set()
        for i, t in enumerate(triples):
            if colors[i] == 0:
                red_set.add(t)
            else:
                blue_set.add(t)
        
        # Check red K4^3: any 4 vertices where all 4 triples are red
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (a,b,c) in red_set and (a,b,d) in red_set and (a,c,d) in red_set and (b,c,d) in red_set:
                return False
        
        # Check blue K5^3: any 5 vertices where all 10 triples are blue
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            all_blue = True
            for t in combinations(quint, 3):
                if t not in blue_set:
                    all_blue = False
                    break
            if all_blue:
                return False
        
        return True
    
    def count_violations(colors, n, triples):
        """Count red K4^3 and blue K5^3 violations."""
        red_set = set()
        blue_set = set()
        for i, t in enumerate(triples):
            if colors[i] == 0:
                red_set.add(t)
            else:
                blue_set.add(t)
        
        red_count = 0
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (a,b,c) in red_set and (a,b,d) in red_set and (a,c,d) in red_set and (b,c,d) in red_set:
                red_count += 1
        
        blue_count = 0
        for quint in combinations(range(n), 5):
            a, b, c, d, e = quint
            all_blue = True
            for t in combinations(quint, 3):
                if t not in blue_set:
                    all_blue = False
                    break
            if all_blue:
                blue_count += 1
        
        return red_count, blue_count
    
    def local_search(colors, n, triples, deadline, max_iterations=5000):
        """Perform local search to eliminate violations."""
        colors = list(colors)
        idx_map, triples = make_index_map(n)
        
        # Precompute for each 4-set the list of triple indices
        quad_triples = {}
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            quad_triples[quad] = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
        
        # Precompute for each 5-set the list of triple indices
        quint_triples = {}
        for quint in combinations(range(n), 5):
            quint_triples[quint] = list(combinations(quint, 3))
        
        iteration = 0
        while time.time() < deadline - 3 and iteration < max_iterations:
            iteration += 1
            
            # Find violations
            red_violations = []
            for quad, trs in quad_triples.items():
                if all(colors[idx_map[t]] == 0 for t in trs):
                    red_violations.append(quad)
            
            blue_violations = []
            for quint, trs in quint_triples.items():
                if all(colors[idx_map[t]] == 1 for t in trs):
                    blue_violations.append(quint)
            
            if not red_violations and not blue_violations:
                return colors, True
            
            # Pick a violation to fix
            if red_violations and (not blue_violations or random.random() < 0.5):
                quad = random.choice(red_violations)
                trs = quad_triples[quad]
                # Try flipping each red triple to blue
                for t in trs:
                    idx = idx_map[t]
                    if colors[idx] == 0:
                        old_color = colors[idx]
                        colors[idx] = 1
                        # Quick check: did we reduce red violations?
                        # Just accept if it doesn't create new blue violations immediately
                        # (full check is expensive, so we do a lightweight check)
                        new_red, new_blue = count_violations(colors, n, triples)
                        if new_red < len(red_violations) or (new_red == len(red_violations) and new_blue < len(blue_violations)):
                            break
                        else:
                            colors[idx] = old_color
            elif blue_violations:
                quint = random.choice(blue_violations)
                trs = quint_triples[quint]
                # Try flipping each blue triple to red
                for t in trs:
                    idx = idx_map[t]
                    if colors[idx] == 1:
                        old_color = colors[idx]
                        colors[idx] = 0
                        new_red, new_blue = count_violations(colors, n, triples)
                        if new_blue < len(blue_violations) or (new_blue == len(blue_violations) and new_red < len(red_violations)):
                            break
                        else:
                            colors[idx] = old_color
        
        # Final validation
        valid = check_validity(colors, n, triples)
        return colors, valid
    
    def heuristic_coloring(n, seed=0):
        """Generate a heuristic initial coloring."""
        random.seed(seed)
        triples = generate_triples(n)
        colors = []
        for a, b, c in triples:
            # Use a mix of modular arithmetic and randomness
            r = (a + 2*b + 3*c) % 5
            if r < 2:
                colors.append(0)  # red
            else:
                colors.append(1)  # blue
        return colors, triples
    
    def try_construct(n, deadline, max_restarts=5):
        """Try to find a valid coloring for given n."""
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Generate initial coloring
            seed = restart * 12345 + n
            colors, triples = heuristic_coloring(n, seed)
            
            # Check if already valid
            if check_validity(colors, n, triples):
                return colors, True
            
            # Try local search
            colors, valid = local_search(colors, n, triples, deadline)
            if valid:
                return colors, True
        
        return None, False
    
    # Try to find the maximum n with a valid coloring
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Start from smaller n and try to increase
    # The known construction has n=34, but let's try to find valid colorings
    # We'll try n from 10 up to 35, but prioritize higher n
    
    # First, try a few moderate n values to see if local search works
    for n in [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 35]:
        if time.time() > deadline - 10:
            break
        
        colors, valid = try_construct(n, deadline, max_restarts=3)
        if valid:
            best_n = n
            best_colors = colors
    
    # If we found a valid coloring, try to extend it
    if best_n > 9 and best_n < 35:
        n2 = best_n + 1
        if time.time() < deadline - 15:
            # Extend the coloring
            old_colors = list(best_colors)
            old_n = best_n
            new_triples = generate_triples(n2)
            new_colors = [0] * len(new_triples)
            
            # Copy old colors
            old_idx_map, old_triples = make_index_map(old_n)
            new_idx_map, _ = make_index_map(n2)
            
            for t, idx in old_idx_map.items():
                new_colors[new_idx_map[t]] = old_colors[idx]
            
            # Color new triples involving vertex n2-1
            for a in range(old_n):
                for b in range(a + 1, old_n):
                    t = (a, b, n2 - 1)
                    idx = new_idx_map[t]
                    # Use heuristic
                    r = (a + 2*b + 3*(n2-1)) % 5
                    new_colors[idx] = 0 if r < 2 else 1
            
            # Try local search on the extended coloring
            fixed_colors, valid = local_search(new_colors, n2, new_triples, deadline, max_iterations=3000)
            if valid:
                best_n = n2
                best_colors = fixed_colors
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
