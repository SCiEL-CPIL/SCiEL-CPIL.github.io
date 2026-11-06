import math
import random
import time

def _is_disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    best_pairs = []
    
    # Moduli 2 to 128
    moduli = list(range(2, 129))
    
    # We will run many random iterations
    # Each iteration shuffles moduli and greedily adds them
    # We can also try different residue selection strategies
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        random.shuffle(moduli)
        
        current_pairs = []
        
        for m in moduli:
            # Check if we can add modulus m
            # We need to find an r in [0, m) such that for all (r_i, m_i) in current_pairs:
            # (r - r_i) % gcd(m, m_i) != 0
            
            # Optimization: Precompute forbidden residues for m
            # forbidden[r] = True if r is not disjoint with some existing pair
            forbidden = [False] * m
            
            for r_i, m_i in current_pairs:
                g = math.gcd(m, m_i)
                # r must not be congruent to r_i mod g
                # r_i mod g is the start of the forbidden class
                start = r_i % g
                for r in range(start, m, g):
                    forbidden[r] = True
            
            # Find an available r
            available = [r for r in range(m) if not forbidden[r]]
            
            if available:
                # To diversify, we can pick a random available r
                # Or the first one. Random might help escape local optima in the residue space
                r = random.choice(available)
                current_pairs.append([r, m])
        
        if len(current_pairs) > len(best_pairs):
            best_pairs = current_pairs
            
        # If we found a very large set, we might stop early? 
        # Max possible is 127. If we get close, maybe stop?
        # But let's run until deadline to be safe.
        
    return best_pairs
