"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

N = 128

def construct():
    # Use a branch-and-bound style search with greedy initialization
    # and local improvement
    
    # Start with a good initial solution
    # Option 1: all numbers > 64 (size 64)
    # Option 2: all odd numbers (size 64)
    # Let's try to find something better
    
    # Greedy: try to build a large set
    # We'll use a randomized restart approach with greedy + local search
    
    import random
    random.seed(42)
    
    best_size = 0
    best_set = set()
    
    # Try multiple random orderings
    for attempt in range(200):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        S = set()
        for x in candidates:
            # Check if adding x creates a forbidden configuration
            valid = True
            for a, b in itertools.combinations(S, 2):
                # Check if {a, b, x} with all pair sums in S
                # Pair sums: a+b, a+x, b+x
                if a + b in S and a + x in S and b + x in S:
                    valid = False
                    break
                # Also need to check triples where x is one of the "sums"
                # Actually the forbidden config is: three distinct members a,b,c 
                # such that a+b, a+c, b+c are also in S
                # So when adding x, we need to check if there exist a,b in S 
                # such that {a,b,x} are the three distinct members and 
                # a+b, a+x, b+x are all in S
                # OR if x is one of the pair sums: e.g., x = a+b for some a,b in S,
                # and then we need a+c, b+c in S for some c... this is complex
                
            # Simpler: check all triples in S ∪ {x}
            S_new = S | {x}
            valid = True
            for a, b, c in itertools.combinations(S_new, 3):
                if a + b in S_new and a + c in S_new and b + c in S_new:
                    valid = False
                    break
            if valid:
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Try local improvement on the best found
    # Try to add elements that were excluded
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            S_new = best_set | {x}
            valid = True
            for a, b, c in itertools.combinations(S_new, 3):
                if a + b in S_new and a + c in S_new and b + c in S_new:
                    valid = False
                    break
            if valid:
                best_set.add(x)
                changed = True
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
