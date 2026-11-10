import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 192
    
    # Precompute: for each x, which triples (a,b,c) with a<b<c from 1..n have (a+b+c)%x==0
    # We'll use this for fast conflict checking during backtracking
    
    # For each element x, precompute for each pair (b,c) the residue (b+c)%x
    # This allows O(1) lookup when checking if adding x conflicts with existing set
    
    # Better approach: precompute conflict triples
    # A triple (x, a, b, c) is "conflicting" if x divides a+b+c
    # When building a set, if we include x, we cannot include all of {a,b,c}
    # But this is a 4-clique constraint, not a graph edge
    
    # Alternative: use the can_add function but optimize it significantly
    
    # Precompute for each x, and for each possible "current set size", 
    # we can't precompute easily. Instead, let's optimize can_add.
    
    # Key optimization: for the second check (any a in S divides x+b+c),
    # we can precompute for each a, the set of residues (b+c)%a for all pairs.
    # But S changes, so we need to maintain this incrementally.
    
    # Let's use a different strategy: backtracking with strong pruning
    # and a good ordering heuristic
    
    def can_add_optimized(S, x):
        """
        Checks if x can be added to set S without violating the condition.
        """
        # Check 1: x does not divide sum of any 3 elements in S
        if len(S) >= 3:
            # For each triple in S, check if sum % x == 0
            # Optimization: precompute prefix sums? No, triples are arbitrary.
            # Just iterate. S is small (<=48).
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    for k in range(j + 1, len(S)):
                        if (S[i] + S[j] + S[k]) % x == 0:
                            return False
        
        # Check 2: no a in S divides x + b + c for any b,c in S\{a}, b!=c
        for a in S:
            if a == 1:
                if len(S) >= 3:
                    return False
                continue
            
            # Need: no b,c in S\{a}, b!=c, such that (x+b+c) % a == 0
            # i.e., (b+c) % a != (-x) % a
            target = (-x) % a
            
            # Collect residues of (b+c)%a for all pairs b,c in S\{a}
            S_minus_a = [v for v in S if v != a]
            if len(S_minus_a) < 2:
                continue
            
            # Check if any pair sums to target mod a
            # Use a set of seen residues
            seen_residues = set()
            for i in range(len(S_minus_a)):
                b = S_minus_a[i]
                for j in range(i + 1, len(S_minus_a)):
                    c = S_minus_a[j]
                    seen_residues.add((b + c) % a)
            
            if target in seen_residues:
                return False
        
        return True
    
    best_set = []
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
    
    # Phase 1: Greedy to get a good initial solution
    S_greedy = []
    for x in range(n, 0, -1):
        if can_add_optimized(S_greedy, x):
            S_greedy.append(x)
    update_best(S_greedy)
    
    # Phase 2: Backtracking with branch and bound
    # Order candidates by a heuristic: elements that are "harder" to place first
    # Heuristic: smaller numbers are more likely to divide sums, so they're more constrained
    # Try ascending order for backtracking
    
    def backtrack(current_set, remaining, idx):
        nonlocal best_size, best_set
        nonlocal deadline, start_time
        
        # Time check
        if time.time() > deadline:
            return
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Upper bound: current size + remaining candidates
        if len(current_set) + len(remaining) <= best_size:
            return
        
        # If no more candidates
        if idx >= len(remaining):
            return
        
        # Branch: try adding remaining[idx]
        x = remaining[idx]
        if can_add_optimized(current_set, x):
            # Include x
            backtrack(current_set + [x], remaining, idx + 1)
            if time.time() > deadline:
                return
        
        # Exclude x
        backtrack(current_set, remaining, idx + 1)
    
    # Use a good ordering: sort by "constraint degree"
    # Elements that divide more sums are more constrained
    # Precompute for each x, how many triples (a,b,c) have (a+b+c)%x==0
    constraint_degree = {}
    for x in range(1, n + 1):
        count = 0
        if x == 1:
            # 1 divides everything
            count = 10**9
        else:
            for a in range(1, n + 1):
                for b in range(a + 1, n + 1):
                    for c in range(b + 1, n + 1):
                        if (a + b + c) % x == 0:
                            count += 1
        constraint_degree[x] = count
    
    # Sort candidates by constraint degree (most constrained first)
    candidates = sorted(range(1, n + 1), key=lambda x: -constraint_degree[x])
    
    backtrack([], candidates, 0)
    
    return best_set
