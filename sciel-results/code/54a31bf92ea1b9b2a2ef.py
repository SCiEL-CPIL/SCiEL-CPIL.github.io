import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s kill
    
    N = 180
    # Generate all forbidden triples: distinct a,b,c in [1,180] with 1/a = 1/b + 1/c
    # a = bc/(b+c), need b+c | bc
    # WLOG assume a < b < c
    forbidden = []  # list of (a, b, c) with a < b < c
    for b in range(1, N+1):
        for c in range(b+1, N+1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    forbidden.append((a, b, c))
    
    # Build conflict maps: for each element, list of triples it's in
    elem_triples = {i: [] for i in range(1, N+1)}
    for t in forbidden:
        for x in t:
            elem_triples[x].append(t)
    
    # Initial solution: [91..180]
    S = set(range(91, 181))
    best_S = set(S)
    best_size = len(S)
    
    def is_valid_add(S, x):
        """Check if adding x to S creates any forbidden triple."""
        for t in elem_triples[x]:
            a, b, c = t
            others = [y for y in t if y != x]
            if all(y in S for y in others):
                return False
        return True
        
    def count_conflicts(S, x):
        """Return list of conflicting triples if x is added to S."""
        conflicts = []
        for t in elem_triples[x]:
            a, b, c = t
            others = [y for y in t if y != x]
            if all(y in S for y in others):
                conflicts.append(t)
        return conflicts

    # Greedy: try to add elements from [1..90] in random order
    random.seed(42)
    
    # Try multiple random orders
    for trial in range(500):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        S = set(range(91, 181))
        candidates = list(range(1, 91))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in S:
                continue
            if is_valid_add(S, x):
                S.add(x)
        
        # Local search: try swaps with FAST validation
        improved = True
        # Limit iterations to prevent infinite loops in a single trial
        for _ in range(10): 
            if time.time() - start_time > DEADLINE - 5:
                break
            improved = False
            excluded = [x for x in range(1, 91) if x not in S]
            random.shuffle(excluded)
            for x in excluded:
                if time.time() - start_time > DEADLINE - 5:
                    break
                conflicts = count_conflicts(S, x)
                if not conflicts:
                    S.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    continue
                # Try to resolve: for each conflict triple, remove one of the other two elements
                # and see if we can add x
                resolved = False
                for t in conflicts:
                    a, b, c = t
                    others = [y for y in t if y != x]
                    # Try removing each other element
                    for rem in others:
                        # Optimization: Only check triples involving x in the new set S2
                        # S2 = (S - {rem}) + {x}
                        # A triple is invalid in S2 only if it contains x and all its other members are in S2.
                        # Since S was valid, only triples containing x can be newly formed.
                        valid = True
                        for t2 in elem_triples[x]:
                            # Check if all elements of t2 are in S2
                            # t2 contains x. Let others2 be the other two.
                            # They must be in S (since x is not in S yet, and we are simulating)
                            # AND not equal to rem (since rem is removed)
                            others2 = [y for y in t2 if y != x]
                            # If any other is rem, it's not in S2.
                            # If any other is not in S, it's not in S2.
                            if all(y in S and y != rem for y in others2):
                                valid = False
                                break
                        if valid:
                            S.discard(rem)
                            S.add(x)
                            improved = True
                            resolved = True
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                            break
                    if resolved:
                        break
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Final polish: try all single-element additions
    if time.time() - start_time < DEADLINE - 2:
        for x in range(1, 91):
            if x not in best_S:
                if is_valid_add(best_S, x):
                    best_S.add(x)
                    best_size = len(best_S)
    
    return sorted(best_S)
