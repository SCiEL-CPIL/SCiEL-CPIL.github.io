"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    n = 128
    best_set = set()
    
    # Run multiple random greedy trials
    for _ in range(100):
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            can_add = True
            # Check if adding x creates a violation with existing pairs in S
            # Violation triple (x, a, b) requires:
            # a, b in S, a != b
            # a+b in S
            # x+a in S
            # x+b in S
            
            # Iterate a in S
            for a in S:
                if not can_add:
                    break
                # Check if x+a is in S
                if (x + a) in S:
                    # Now check for b
                    for b in S:
                        if b == a:
                            continue
                        # We can assume b > a to avoid duplicates, but set iteration is fine
                        # Check a+b in S and x+b in S
                        if (a + b) in S and (x + b) in S:
                            can_add = False
                            break
            if can_add:
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = S.copy()
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
