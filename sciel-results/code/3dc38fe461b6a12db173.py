import time
import random
import numpy as np
from itertools import combinations
from math import comb

def construct():
    """Construct a valid 2-coloring of triples maximizing n."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def try_find_coloring(n, time_limit):
        """
        Try to find a valid coloring for n vertices using local search.
        """
        if time.time() > deadline:
            return None
            
        # Precompute triples
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute for each triple, the 4-sets it belongs to
        quads = list(combinations(range(n), 4))
        num_quads = len(quads)
        
        tri_to_quads = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quads):
            a, b, c, d = q
            t1 = (a, b, c)
            t2 = (a, b, d)
            t3 = (a, c, d)
            t4 = (b, c, d)
            tri_to_quads[triple_to_idx[t1]].append(q_idx)
            tri_to_quads[triple_to_idx[t2]].append(q_idx)
            tri_to_quads[triple_to_idx[t3]].append(q_idx)
            tri_to_quads[triple_to_idx[t4]].append(q_idx)

        # Precompute for each triple, the 5-sets it belongs to
        quints = list(combinations(range(n), 5))
        num_quints = len(quints)
        
        tri_to_quints = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quints):
            for t in combinations(q, 3):
                tri_to_quints[triple_to_idx[t]].append(q_idx)

        # We will try multiple restarts
        max_restarts = 50 # Adjust based on time
        
        while time.time() < deadline - 2:
            # Random initialization
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Calculate initial red counts for quads
            red_counts = [0] * num_quads
            for t_idx in range(num_triples):
                if colors[t_idx] == 0: # Red
                    for q_idx in tri_to_quads[t_idx]:
                        red_counts[q_idx] += 1
            
            # Calculate initial blue counts for quints
            blue_counts = [0] * num_quints
            for t_idx in range(num_triples):
                if colors[t_idx] == 1: # Blue
                    for q_idx in tri_to_quints[t_idx]:
                        blue_counts[q_idx] += 1
            
            # Count violations
            red_violations = sum(1 for c in red_counts if c == 4)
            blue_violations = sum(1 for c in blue_counts if c == 10)
            
            if red_violations == 0 and blue_violations == 0:
                return colors
            
            # Local Search
            no_improve_count = 0
            max_no_improve = 10000 # Adjust
            
            while time.time() < deadline - 2 and no_improve_count < max_no_improve:
                if red_violations == 0 and blue_violations == 0:
                    return colors
                
                # Pick a random violation
                # Weighted choice: pick red or blue violation based on count
                total_violations = red_violations + blue_violations
                pick_red = (red_violations > 0 and (blue_violations == 0 or random.random() < red_violations / total_violations))
                
                if pick_red:
                    # Find a random red violation
                    while True:
                        q_idx = random.randint(0, num_quads - 1)
                        if red_counts[q_idx] == 4:
                            break
                    quad = quads[q_idx]
                    a, b, c, d = quad
                    candidates = [
                        triple_to_idx[(a,b,c)],
                        triple_to_idx[(a,b,d)],
                        triple_to_idx[(a,c,d)],
                        triple_to_idx[(b,c,d)]
                    ]
                    t_idx = random.choice(candidates)
                    
                    # Flip Red (0) to Blue (1)
                    # Check if this creates new blue violations
                    # It reduces red counts, so red violations only decrease.
                    # It increases blue counts. Check for 5-sets with blue count 9.
                    new_blue_violations_created = 0
                    for q_i in tri_to_quints[t_idx]:
                        if blue_counts[q_i] == 9:
                            new_blue_violations_created += 1
                    
                    # We resolve at least 1 red violation (the one we picked).
                    # Actually, we might resolve more if other quads in tri_to_quads[t_idx] had count 4.
                    # But for the decision, knowing we resolve >= 1 and create 'new_blue_violations_created' is enough.
                    # If new_blue_violations_created < 1, it's a strict improvement.
                    # If == 1, it's a lateral move.
                    # If > 1, it's a worsening move.
                    
                    if new_blue_violations_created == 0:
                        # Strict improvement
                        colors[t_idx] = 1
                        for q_i in tri_to_quads[t_idx]:
                            red_counts[q_i] -= 1
                            if red_counts[q_i] == 3: # Was 4
                                red_violations -= 1
                        for q_i in tri_to_quints[t_idx]:
                            blue_counts[q_i] += 1
                        no_improve_count = 0
                    elif new_blue_violations_created == 1 and random.random() < 0.3:
                        # Lateral move, accept with some probability
                        colors[t_idx] = 1
                        for q_i in tri_to_quads[t_idx]:
                            red_counts[q_i] -= 1
                            if red_counts[q_i] == 3: # Was 4
                                red_violations -= 1
                        for q_i in tri_to_quints[t_idx]:
                            blue_counts[q_i] += 1
                            if blue_counts[q_i] == 10: # Was 9
                                blue_violations += 1
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    # Find a random blue violation
                    while True:
                        q_idx = random.randint(0, num_quints - 1)
                        if blue_counts[q_idx] == 10:
                            break
                    quint = quints[q_idx]
                    t_candidates = list(combinations(quint, 3))
                    t = random.choice(t_candidates)
                    t_idx = triple_to_idx[t]
                    
                    # Flip Blue (1) to Red (0)
                    # Check if this creates new red violations
                    new_red_violations_created = 0
                    for q_i in tri_to_quads[t_idx]:
                        if red_counts[q_i] == 3:
                            new_red_violations_created += 1
                    
                    if new_red_violations_created == 0:
                        # Strict improvement
                        colors[t_idx] = 0
                        for q_i in tri_to_quints[t_idx]:
                            blue_counts[q_i] -= 1
                            if blue_counts[q_i] == 9: # Was 10
                                blue_violations -= 1
                        for q_i in tri_to_quads[t_idx]:
                            red_counts[q_i] += 1
                        no_improve_count = 0
                    elif new_red_violations_created == 1 and random.random() < 0.3:
                        # Lateral move
                        colors[t_idx] = 0
                        for q_i in tri_to_quints[t_idx]:
                            blue_counts[q_i] -= 1
                            if blue_counts[q_i] == 9: # Was 10
                                blue_violations -= 1
                        for q_i in tri_to_quads[t_idx]:
                            red_counts[q_i] += 1
                            if red_counts[q_i] == 4: # Was 3
                                red_violations += 1
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                        
            # Restart if stuck
            
        return None

    # Try to find a valid coloring for decreasing n
    # The known solution is n=34.
    for n in [34, 33, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 9]:
        if time.time() > deadline - 5:
            break
            
        found = try_find_coloring(n, deadline)
        if found:
            return {'n': n, 'colors': found}
            
    # Fallback to n=9
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
