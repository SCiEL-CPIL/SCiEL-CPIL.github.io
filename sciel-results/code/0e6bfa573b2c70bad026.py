"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    """
    Construct a 2-coloring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Uses a deterministic algebraic construction based on modular arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Try increasing n values, starting from known good constructions
    # The published construction has n=34, so we aim for that or better
    
    # Algebraic construction: use finite field / modular arithmetic
    # For a prime p, color triple (a,b,c) red if (a+b+c) mod p == 0, else blue
    # This avoids monochromatic K4^(3) in red when p >= 4
    
    # Better construction: use multiple modular conditions
    # Color based on the value of a^2 + b^2 + c^2 mod m or similar
    
    # Known approach: use n = p where p is prime, and color by (a+b+c) mod p
    # But we need to avoid blue K5^(3) as well
    
    # Let's try a construction based on the cyclic group Z_n
    # For n = 2m+1 (odd), color (a,b,c) red if {a,b,c} has some property
    
    # Simpler: use the fact that for n <= 34, we can use a specific algebraic rule
    # Color red if (a+b+c) % 2 == 0 AND (a*b + b*c + c*a) % 3 == 0
    
    # Let's try a systematic approach with different algebraic rules
    
    best_n = 0
    best_colors = []
    
    # Try various algebraic constructions
    constructions = []
    
    # Construction 1: (a+b+c) mod k == 0 for red
    for k in range(2, 20):
        constructions.append(('mod_sum', k))
    
    # Construction 2: (a^2+b^2+c^2) mod k == 0 for red
    for k in range(2, 20):
        constructions.append(('mod_sum_sq', k))
    
    # Construction 3: (a*b+b*c+c*a) mod k == 0 for red
    for k in range(2, 20):
        constructions.append(('mod_pairwise', k))
    
    # Construction 4: Combination of conditions
    for k1 in range(2, 10):
        for k2 in range(2, 10):
            constructions.append(('combo', k1, k2))
    
    # Try n from 34 down to 10, and for each try different constructions
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
        
        for constr in constructions:
            if time.time() > deadline - 5:
                break
            
            colors = []
            valid = True
            
            for a, b, c in itertools.combinations(range(n), 3):
                if constr[0] == 'mod_sum':
                    color = 0 if (a + b + c) % constr[1] == 0 else 1
                elif constr[0] == 'mod_sum_sq':
                    color = 0 if (a*a + b*b + c*c) % constr[1] == 0 else 1
                elif constr[0] == 'mod_pairwise':
                    color = 0 if (a*b + b*c + c*a) % constr[1] == 0 else 1
                elif constr[0] == 'combo':
                    k1, k2 = constr[1], constr[2]
                    color = 0 if ((a + b + c) % k1 == 0 and (a*b + b*c + c*a) % k2 == 0) else 1
                else:
                    continue
                
                colors.append(color)
            
            # Check validity: no red K4^(3) and no blue K5^(3)
            if not _check_validity(n, colors):
                continue
            
            # This construction works for n
            if n > best_n:
                best_n = n
                best_colors = colors
            
            # If we found n=34, that's the target
            if n >= 34:
                return {'n': n, 'colors': colors}
        
        # If we've tried all constructions for this n and none work, try smaller n
        if best_n >= n:
            continue
    
    # If no algebraic construction worked, fall back to a simple valid construction
    if best_n == 0:
        n = 9
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
        return {'n': n, 'colors': colors}
    
    return {'n': best_n, 'colors': best_colors}

def _check_validity(n, colors):
    """Check if coloring has no red K4^(3) and no blue K5^(3)."""
    import itertools
    
    # Map triple to color index
    triple_to_color = {}
    idx = 0
    for a, b, c in itertools.combinations(range(n), 3):
        triple_to_color[(a, b, c)] = colors[idx]
        idx += 1
    
    # Check for red K4^(3): 4 vertices with all 4 triples red
    for quad in itertools.combinations(range(n), 4):
        all_red = True
        for triple in itertools.combinations(quad, 3):
            if triple_to_color.get(triple, 1) != 0:
                all_red = False
                break
        if all_red:
            return False
    
    # Check for blue K5^(3): 5 vertices with all 10 triples blue
    for quint in itertools.combinations(range(n), 5):
        all_blue = True
        for triple in itertools.combinations(quint, 3):
            if triple_to_color.get(triple, 0) != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True
# EVOLVE-BLOCK-END
