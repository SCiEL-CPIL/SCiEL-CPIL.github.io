import itertools

def construct():
    # Modular inverses of 1..10 modulo 101
    p = 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, p)
        
    # We want to assign the smallest possible subset to each residue.
    # Iterate by subset size k from 1 to 10.
    # For each size, generate all combinations.
    # If a residue is not yet covered, assign the current combination to it.
    
    assigned = {}
    # The denominators are 1..10
    denominators = list(range(1, 11))
    
    for k in range(1, 11):
        # Generate all combinations of size k
        for comb in itertools.combinations(denominators, k):
            # Calculate sum of inverses modulo 101
            s = sum(inv[x] for x in comb) % p
            if s not in assigned:
                assigned[s] = list(comb)
                
        # If we have covered all 101 residues, we can stop early
        if len(assigned) == 101:
            break
            
    # Construct the result list of 101 lists
    # Row r contains the list for residue r
    result = []
    for r in range(101):
        result.append(assigned[r])
        
    return result
