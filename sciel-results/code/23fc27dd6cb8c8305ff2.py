import itertools

def construct():
    # Define cores
    C1 = [0, 1, 2, 3]
    C2 = [4, 5, 6, 7]
    C3 = [8, 9, 10]
    
    sets = []
    
    # Group 1: Subsets of C1
    for r in range(len(C1) + 1):
        for combo in itertools.combinations(C1, r):
            sets.append(list(combo))
            
    # Group 2: C1 union Subsets of C2
    for r in range(len(C2) + 1):
        for combo in itertools.combinations(C2, r):
            s = C1 + list(combo)
            sets.append(s)
            
    # Group 3: C1 union C2 union Subsets of C3
    for r in range(len(C3) + 1):
        for combo in itertools.combinations(C3, r):
            s = C1 + C2 + list(combo)
            sets.append(s)
            
    return sets
