import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Helper to check if a specific coloring is valid
    def is_valid(n, colors):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4
        for comb4 in combinations(range(n), 4):
            a, b, c, d = comb4
            if (colors[triple_to_idx[(a,b,c)]] == 0 and
                colors[triple_to_idx[(a,b,d)]] == 0 and
                colors[triple_to_idx[(a,c,d)]] == 0 and
                colors[triple_to_idx[(b,c,d)]] == 0):
                return False
        
        # Check Blue K5
        for comb5 in combinations(range(n), 5):
            a, b, c, d, e = comb5
            ts = [
                (a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e),
                (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)
            ]
            is_blue = True
            for t in ts:
                if colors[triple_to_idx[t]] != 1:
                    is_blue = False
                    break
            if is_blue:
                return False
        return True

    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 5.0:
            break
            
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Pre-compute K4 and K5 structures for fast delta calculation
        # k4s_containing[i] = list of (k4_index, [other_triple_indices])
        # Actually, we just need to iterate d for K4 and (d,e) for K5 on the fly, 
        # but we need to access colors fast.
        
        # Let's pre-compute the neighbors for each triple index
        # For K4: for each triple i, list of d such that (a,b,c,d) is a K4
        # For K5: for each triple i, list of (d,e) such that (a,b,c,d,e) is a K5
        
        # This pre-computation might be heavy, but it's done once per n.
        # Alternatively, calculate on the fly. On-the-fly is O(n^2) per flip.
        # With 5984 triples, and potentially many flips, pre-computing might be better.
        
        # Let's try a simpler approach first: Random restarts with a few local search steps.
        # If that fails, we fall back to smaller n.
        
        found = False
        max_restarts = 20
        for r in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            # Initial violation count
            # We can use a fast check. If valid, return.
            if is_valid(n, colors):
                return {'n': n, 'colors': colors}
            
            # Local Search
            improved = True
            steps = 0
            max_steps = 1000 # Limit steps per restart to avoid getting stuck
            while improved and steps < max_steps and time.time() < deadline - 2.0:
                improved = False
                indices = list(range(num_triples))
                random.shuffle(indices)
                
                for idx in indices:
                    if time.time() > deadline - 2.0:
                        break
                        
                    a, b, c = triples[idx]
                    old_color = colors[idx]
                    new_color = 1 - old_color
                    
                    # Calculate delta violations
                    delta = 0
                    
                    # Check K4s containing (a,b,c)
                    # A K4 is red if all 4 triples are 0.
                    # If we flip (a,b,c), the K4 status changes only if the other 3 are 0.
                    # If old_color == 0 and others are 0: K4 was red, now not. delta -= 1
                    # If old_color == 1 and others are 0: K4 was not red, now red. delta += 1
                    
                    for d in range(n):
                        if d == a or d == b or d == c:
                            continue
                        # Other triples in K4 {a,b,c,d}
                        i2 = triple_to_idx[(a,b,d)]
                        i3 = triple_to_idx[(a,c,d)]
                        i4 = triple_to_idx[(b,c,d)]
                        
                        if colors[i2] == 0 and colors[i3] == 0 and colors[i4] == 0:
                            if old_color == 0:
                                delta -= 1 # Removed a red K4
                            else:
                                delta += 1 # Created a red K4
                    
                    # Check K5s containing (a,b,c)
                    # A K5 is blue if all 10 triples are 1.
                    # If we flip (a,b,c), the K5 status changes only if the other 9 are 1.
                    # If old_color == 1 and others are 1: K5 was blue, now not. delta -= 1
                    # If old_color == 0 and others are 1: K5 was not blue, now blue. delta += 1
                    
                    others = [v for v in range(n) if v != a and v != b and v != c]
                    for i in range(len(others)):
                        for j in range(i+1, len(others)):
                            d = others[i]
                            e = others[j]
                            # Other 9 triples in K5 {a,b,c,d,e}
                            # (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e)
                            # (b,c,d), (b,c,e), (b,d,e), (c,d,e)
                            
                            # Check if all 9 are 1
                            is_blue_other = (
                                colors[triple_to_idx[(a,b,d)]] == 1 and
                                colors[triple_to_idx[(a,b,e)]] == 1 and
                                colors[triple_to_idx[(a,c,d)]] == 1 and
                                colors[triple_to_idx[(a,c,e)]] == 1 and
                                colors[triple_to_idx[(a,d,e)]] == 1 and
                                colors[triple_to_idx[(b,c,d)]] == 1 and
                                colors[triple_to_idx[(b,c,e)]] == 1 and
                                colors[triple_to_idx[(b,d,e)]] == 1 and
                                colors[triple_to_idx[(c,d,e)]] == 1
                            )
                            
                            if is_blue_other:
                                if old_color == 1:
                                    delta -= 1 # Removed a blue K5
                                else:
                                    delta += 1 # Created a blue K5
                                    
                    if delta < 0:
                        colors[idx] = new_color
                        improved = True
                        # We can't easily track total violations incrementally without a global counter 
                        # and updating it, but since we check validity at the end of the pass or restart,
                        # we just need to ensure we are making progress.
                        # Actually, we should check if total violations became 0.
                        # But we don't have the total count here.
                        # We can just run is_valid() periodically or at the end of the pass.
                        
                    steps += 1
                    if steps > 2000: # Hard cap per restart to prevent infinite loops
                        break
            
            # After local search, check if valid
            if is_valid(n, colors):
                return {'n': n, 'colors': colors}

    # If we didn't find a solution for n > 9, fall back to the algebraic construction for n=9
    # Or try to find a small n solution
    # The seed program returns n=9.
    # Let's try to find a valid coloring for n=10, 11... just in case.
    # But n=9 is guaranteed by the seed.
    
    # Use the seed's algebraic construction as a fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
