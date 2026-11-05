import math
import time

def construct():
    deadline = time.time() + 115
    
    # Precompute gcds and conflicts structure if needed, 
    # but for 181 variables, direct check is fast enough.
    
    moduli = list(range(20, 201))
    n = len(moduli)
    
    # We want to assign residues r[i] for moduli[i]
    # such that r[i] != r[j] % gcd(moduli[i], moduli[j])
    
    # Backtracking with Forward Checking
    # Order: We process moduli in increasing order.
    # This is often good because smaller moduli have smaller domains (relatively) 
    # and constrain larger ones.
    
    assignment = [None] * n
    # domains[i] is a list of valid residues for moduli[i] given current assignment
    # Initially all residues 0..m-1
    
    # To speed up, we can pre-compute gcds
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    def is_valid(i, val):
        """Check if assigning val to moduli[i] is consistent with previous assignments."""
        for j in range(i):
            if assignment[j] is not None:
                g = gcds[i][j]
                if val % g == assignment[j] % g:
                    return False
        return True
        
    def backtrack(i):
        if time.time() > deadline:
            return False
        if i == n:
            return True
            
        m = moduli[i]
        # Try residues in a specific order? 
        # Randomizing might help escape dead ends if any, 
        # but for this problem, 0 is often fine.
        # Let's try 0, 1, 2...
        
        for r in range(m):
            if is_valid(i, r):
                assignment[i] = r
                if backtrack(i + 1):
                    return True
                assignment[i] = None
                
        return False

    # Start backtracking
    if backtrack(0):
        # Found a valid assignment for all moduli
        # This is the optimal solution (sum of all 1/m)
        return [[moduli[i], assignment[i]] for i in range(n)]
    else:
        # If backtracking failed (should not happen if full solution exists),
        # fall back to a greedy heuristic to get a partial solution
        selected = []
        for i, m in enumerate(moduli):
            for r in range(m):
                if is_valid_for_selected(selected, m, r):
                    selected.append((m, r))
                    break
        return [[m, r] for m, r in selected]

def is_valid_for_selected(selected, m, r):
    for m2, r2 in selected:
        g = math.gcd(m, m2)
        if r % g == r2 % g:
            return False
    return True
