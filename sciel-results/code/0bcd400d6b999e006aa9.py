"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Start with the seed construction
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(9), 3)]
    
    def count_violations(colors, n):
        """Count red K4^3 and blue K5^3 violations."""
        red_count = 0
        blue_count = 0
        
        # Check for red K4^3: all 4 triples red among any 4 vertices
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            # Get the 4 triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            # Need to find their indices in lexicographic order
            # (a,b,c) -> index
            # We'll compute indices directly
            idx_abc = (a * (n-1) * (n-2)) // 2 + (b - a - 1) * (n - b - 1) + (c - b - 1)
            # Actually, let's use a simpler approach: precompute a lookup
            # For now, let's just use combinations and check
            pass
        
        # Simpler approach: use a helper to get triple index
        def triple_index(a, b, c, n):
            """Get the lexicographic index of triple (a,b,c) in combinations of range(n), 3."""
            # Number of triples starting with 0,1,...,a-1
            count = 0
            for x in range(a):
                count += (n - x - 1) * (n - x - 2) // 2
            # Number of triples starting with a, second element < b
            for y in range(a + 1, b):
                count += (n - y - 1)
            # Number of triples starting with (a,b), third element < c
            count += (c - b - 1)
            return count
        
        # Check red K4^3
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            triples = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            all_red = True
            for t in triples:
                idx = triple_index(t[0], t[1], t[2], n)
                if colors[idx] != 0:
                    all_red = False
                    break
            if all_red:
                red_count += 1
        
        # Check blue K5^3
        for quint in itertools.combinations(range(n), 5):
            all_blue = True
            for t in itertools.combinations(quint, 3):
                idx = triple_index(t[0], t[1], t[2], n)
                if colors[idx] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_count += 1
        
        return red_count + blue_count
    
    # Try to extend the construction
    for target_n in range(10, 20):
        if time.time() > deadline - 5:
            break
        
        # Try to find a valid coloring for target_n
        # Start from the seed pattern and extend
        base_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(target_n), 3)]
        
        # Try some random perturbations
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            colors = base_colors[:]
            
            # Randomly flip some colors
            num_flips = random.randint(0, min(20, len(colors) // 10))
            for _ in range(num_flips):
                idx = random.randint(0, len(colors) - 1)
                colors[idx] = 1 - colors[idx]
            
            violations = count_violations(colors, target_n)
            if violations == 0:
                best_n = target_n
                best_colors = colors
                break
        
        if best_n >= target_n:
            break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
