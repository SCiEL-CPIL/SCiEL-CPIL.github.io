import random
import time

def construct():
    n = 128
    # Start with a random or structured initial solution
    # Odds are a good start (size 64, 0 violations)
    S = set(range(1, n + 1, 2))
    
    # Define violation check
    def has_violation(S):
        # Check if there exist distinct a, b, c in S
        # such that a+b, a+c, b+c are in S
        S_list = list(S)
        # Optimization: Only check pairs (a,b) where a+b is in S
        # If a+b not in S, no violation can involve a,b as the pair summing to the first term
        # Wait, the condition is symmetric. 
        # If a+b in S, a+c in S, b+c in S.
        # So at least one pair sum must be in S.
        # Iterate all pairs (a,b) with a < b
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                s_ab = a + b
                if s_ab not in S:
                    continue
                # We have a+b in S.
                # Now find c in S, c != a, c != b
                # such that a+c in S and b+c in S
                # Pre-compute neighbors? 
                # For fixed a, b:
                # c must be in S
                # a+c in S => c in S-a
                # b+c in S => c in S-b
                # So c in S \cap (S-a) \cap (S-b) \ {a,b}
                # S-a = {x-a for x in S if x-a > 0}
                # This is getting complex to do on the fly.
                
                # Simpler: iterate c
                for c in S_list:
                    if c == a or c == b:
                        continue
                    if (a + c) in S and (b + c) in S:
                        return True
        return False

    # This is O(N^3) worst case, but breaks early.
    # With N=64, N^3 = 250k. Fast.
    
    # Simulated Annealing
    # We want to maximize size.
    # Start with S (size 64).
    # Try to add evens.
    
    # Better: Random restarts or local search from odds.
    # Let's try a greedy local search:
    # 1. Start with odds.
    # 2. For each even e not in S:
    #    Try adding e.
    #    If valid, add.
    #    If invalid, try to remove one element from S to make it valid?
    #    This is tricky.
    
    # Let's use the SA approach properly.
    # State: Set S.
    # Objective: Maximize |S| subject to No Violations.
    # We can define a score: |S| - M * Violations.
    # Start with S = Odds. Violations = 0. Score = 64.
    # Moves: Add a random number, Remove a random number.
    
    def count_violations(S):
        count = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if a + b in S:
                    for c in S_list:
                        if c != a and c != b:
                            if a + c in S and b + c in S:
                                count += 1
        return count

    # Initial S
    S = set(range(1, n + 1, 2))
    current_size = len(S)
    
    # Candidates to add
    candidates = [x for x in range(1, n + 1) if x not in S]
    
    # Local Search: Try to add each candidate, if conflict, try to resolve by removing one element?
    # Actually, let's just do a simple Greedy with Backtracking for a few steps.
    
    # Alternative: Just use the existing greedy but smarter order?
    # The current program adds evens in descending order.
    # Maybe ascending is better? Or random?
    
    # Let's try Randomized Greedy with multiple restarts.
    best_S = set(S)
    
    for _ in range(50): # 50 restarts
        S_curr = set(range(1, n + 1, 2))
        candidates = [x for x in range(1, n + 1) if x not in S_curr]
        random.shuffle(candidates)
        
        for e in candidates:
            S_test = S_curr | {e}
            if not has_violation(S_test):
                S_curr = S_test
                if len(S_curr) > len(best_S):
                    best_S = S_curr.copy()
                    
    return sorted(best_S)
