import random
from itertools import combinations
from math import lcm

# Precompute bad triples
# conflicts[x] is a set of tuples (a, b) such that {a, b, x} is a bad triple
# We assume a < b for the pair to avoid duplicates in the set, but for checking we just need existence.
conflicts = {i: set() for i in range(1, 129)}

def is_bad_triple(a, b, c):
    if a == b or b == c or a == c:
        return False
    l_ab = lcm(a, b)
    l_ac = lcm(a, c)
    l_bc = lcm(b, c)
    return l_ab == l_ac == l_bc

# Precompute conflicts
for a in range(1, 129):
    for b in range(a + 1, 129):
        for c in range(b + 1, 129):
            if is_bad_triple(a, b, c):
                # Add (b, c) to conflicts[a]
                conflicts[a].add((b, c))
                # Add (a, c) to conflicts[b]
                conflicts[b].add((a, c))
                # Add (a, b) to conflicts[c]
                conflicts[c].add((a, b))

def can_add(S, x):
    """Check if x can be added to S without creating a bad triple."""
    for a, b in conflicts[x]:
        if a in S and b in S:
            return False
    return True

def construct():
    best_set = set()
    best_size = 0
    
    # Number of restarts. 60s limit.
    # Each restart: Greedy O(N^2 * D) + Local Search.
    # N=128. D is small.
    # Let's try 500 restarts.
    
    for _ in range(500):
        S = set()
        # Random order for greedy addition
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S, x):
                S.add(x)
        
        # Local Search: Try to add any remaining element
        # This handles the case where order mattered
        improved = True
        while improved:
            improved = False
            remaining = [x for x in range(1, 129) if x not in S]
            random.shuffle(remaining) # Randomize check order to break ties/symmetries
            for x in remaining:
                if can_add(S, x):
                    S.add(x)
                    improved = True
        
        # Local Search: Swaps
        # Try to swap u in S with v not in S to potentially allow future additions
        # This is a random walk. We do a limited number of steps.
        for _ in range(20): # 20 swap attempts
            if len(S) == 0: 
                break
            u = random.choice(list(S))
            # Pick v not in S
            # To make it fast, we can just pick a random number and check if it's in S, or maintain a list
            # Since N is small, picking from range(1,129) and checking membership is O(1)
            v = random.randint(1, 128)
            if v in S:
                continue
            
            # Check if S' = S \ {u} U {v} is valid
            # This is equivalent to: 
            # 1. v is compatible with S \ {u}
            # 2. No bad triples created by v with pairs in S \ {u}
            # Actually, can_add(S \ {u}, v) is the check.
            
            # We need to check if v conflicts with any pair in S\{u}.
            # conflicts[v] contains pairs (a,b). We need to ensure no (a,b) in conflicts[v]
            # has both a,b in S\{u}.
            
            valid_swap = True
            for a, b in conflicts[v]:
                # If both a and b are in S, and neither is u, then conflict exists.
                # If one is u, it's removed, so no conflict.
                if a in S and b in S:
                    if a != u and b != u:
                        valid_swap = False
                        break
                    # If a==u or b==u, the pair (a,b) is not fully in S\{u}, so it's fine.
                    # Wait, if a==u, then a is not in S\{u}. So the pair is not present.
                    # So we only fail if a != u AND b != u.
            
            if valid_swap:
                S.remove(u)
                S.add(v)
                
                # After swap, try to add more
                improved = True
                while improved:
                    improved = False
                    remaining = [x for x in range(1, 129) if x not in S]
                    random.shuffle(remaining)
                    for x in remaining:
                        if can_add(S, x):
                            S.add(x)
                            improved = True
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
        # Early stop if we found a very good solution
        if best_size >= 115: 
             break

    return list(best_set)
