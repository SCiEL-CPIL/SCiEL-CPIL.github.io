import random
from itertools import combinations

def construct():
    n = 34
    triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute 4-sets and 5-sets for conflict checking?
    # No, we can just check on the fly.
    
    random.seed(42)
    
    # Initialize
    colors = [random.randint(0, 1) for _ in triples]
    
    # Helper to check if a specific 4-set is a red K4
    def is_red_k4(v4):
        t1 = triple_to_idx[(v4[0], v4[1], v4[2])]
        t2 = triple_to_idx[(v4[0], v4[1], v4[3])]
        t3 = triple_to_idx[(v4[0], v4[2], v4[3])]
        t4 = triple_to_idx[(v4[1], v4[2], v4[3])]
        return colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0

    # Helper to check if a specific 5-set is a blue K5
    def is_blue_k5(v5):
        for t3 in combinations(v5, 3):
            idx = triple_to_idx[t3]
            if colors[idx] != 1:
                return False
        return True

    # Repair loop
    max_iters = 100000
    for _ in range(max_iters):
        # Check for any conflict
        conflict_found = False
        
        # Check a random subset of 4-sets and 5-sets to find a conflict
        # To be thorough, we should check all, but that's slow.
        # However, if we fix one, we might create another.
        # Let's just pick a random 4-set and a random 5-set.
        
        # Pick random 4-set
        v4 = tuple(sorted(random.sample(range(n), 4)))
        if is_red_k4(v4):
            # Fix: flip one of the 4 triples to blue
            t1 = triple_to_idx[(v4[0], v4[1], v4[2])]
            t2 = triple_to_idx[(v4[0], v4[1], v4[3])]
            t3 = triple_to_idx[(v4[0], v4[2], v4[3])]
            t4 = triple_to_idx[(v4[1], v4[2], v4[3])]
            idx_to_flip = random.choice([t1, t2, t3, t4])
            colors[idx_to_flip] = 1
            continue # Restart check
        
        # Pick random 5-set
        v5 = tuple(sorted(random.sample(range(n), 5)))
        if is_blue_k5(v5):
            # Fix: flip one of the 10 triples to red
            triples_in_v5 = [triple_to_idx[t3] for t3 in combinations(v5, 3)]
            idx_to_flip = random.choice(triples_in_v5)
            colors[idx_to_flip] = 0
            continue # Restart check
            
        # If we didn't find a conflict in these random samples, 
        # we might still have one. 
        # To be sure, we need a full scan occasionally or a counter.
        # Given the small n, a full scan is O(300k). 
        # If we do a full scan every 1000 steps, it's manageable.
        
        if _ % 1000 == 0:
            # Full scan
            has_conflict = False
            for v4 in combinations(range(n), 4):
                if is_red_k4(v4):
                    has_conflict = True
                    # Fix it
                    t1 = triple_to_idx[(v4[0], v4[1], v4[2])]
                    t2 = triple_to_idx[(v4[0], v4[1], v4[3])]
                    t3 = triple_to_idx[(v4[0], v4[2], v4[3])]
                    t4 = triple_to_idx[(v4[1], v4[2], v4[3])]
                    idx_to_flip = random.choice([t1, t2, t3, t4])
                    colors[idx_to_flip] = 1
                    break
            if not has_conflict:
                for v5 in combinations(range(n), 5):
                    if is_blue_k5(v5):
                        has_conflict = True
                        triples_in_v5 = [triple_to_idx[t3] for t3 in combinations(v5, 3)]
                        idx_to_flip = random.choice(triples_in_v5)
                        colors[idx_to_flip] = 0
                        break
            
            if not has_conflict:
                return {'n': n, 'colors': colors}
                
    # If we exit the loop without finding a solution, return n=9
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
