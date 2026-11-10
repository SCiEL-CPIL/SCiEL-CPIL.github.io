"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for 90s total, safety margin
    
    # Precompute GCDs for all pairs in [1, 128]
    N = 128
    g = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g[i][j] = gcd(i, j)
            g[j][i] = g[i][j]
    
    # Initial candidate set: all numbers
    # To improve performance, we can start with a heuristic subset.
    # Numbers with more small prime factors are more likely to have GCDs > 1.
    # But let's start with all and prune.
    
    # A better initial guess: Start with numbers divisible by 2, 3, or 5?
    # Or just all numbers and remove conflicts.
    
    # Let's try a greedy construction in multiple random orders and keep the best.
    # Or a local search.
    
    # Given N=128, a local search removing elements from violating triples is fast.
    
    def count_violations(S):
        """Returns list of violating triples"""
        S_list = list(S)
        violations = []
        # To speed up, we can check triples involving each pair?
        # Or just iterate all triples. C(128,3) ~ 340k. 
        # 340k * 3 gcd lookups is fast.
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                gab = g[a][b]
                for k in range(j + 1, len(S_list)):
                    c = S_list[k]
                    if gab == g[a][c] == g[b][c]:
                        violations.append((a, b, c))
        return violations

    def solve_local_search(initial_set):
        S = set(initial_set)
        # Iterate until no violations
        while True:
            if time.time() > deadline:
                return S
            violations = count_violations(S)
            if not violations:
                return S
            
            # Find the element involved in the most violations
            conflict_count = {}
            for a, b, c in violations:
                conflict_count[a] = conflict_count.get(a, 0) + 1
                conflict_count[b] = conflict_count.get(b, 0) + 1
                conflict_count[c] = conflict_count.get(c, 0) + 1
            
            # Remove the element with the max conflict count
            # Tie-breaker: remove the larger number? Or smaller?
            # Let's remove the one with max conflict, then max value.
            max_conflict = -1
            candidate_to_remove = None
            for elem in S:
                c = conflict_count.get(elem, 0)
                if c > max_conflict:
                    max_conflict = c
                    candidate_to_remove = elem
                elif c == max_conflict:
                    # Tie-breaker: prefer removing numbers that are "less central" or just arbitrary
                    # Let's stick to first found or random.
                    pass
            
            S.remove(candidate_to_remove)
            
    # Try multiple starting points to escape local minima
    best_set = set()
    
    # Strategy 1: Start with all numbers
    best_set = solve_local_search(range(1, 129))
    
    # Strategy 2: Start with even numbers
    if time.time() < deadline:
        s2 = solve_local_search(range(2, 129, 2))
        if len(s2) > len(best_set):
            best_set = s2
            
    # Strategy 3: Start with numbers divisible by 2 or 3
    if time.time() < deadline:
        s3 = solve_local_search([x for x in range(1, 129) if x % 2 == 0 or x % 3 == 0])
        if len(s3) > len(best_set):
            best_set = s3
            
    # Strategy 4: Greedy addition
    # Try to add numbers one by one, checking validity.
    # Order by number of divisors?
    # We can try a few random orders.
    import random
    for _ in range(10):
        if time.time() > deadline:
            break
        random.seed(_ + 123)
        candidates = list(range(1, 129))
        random.shuffle(candidates)
        S = set()
        for x in candidates:
            # Check if adding x violates condition with any pair in S
            valid = True
            # Optimization: Only check pairs where GCDs might be equal to gcd(x, a)
            # But simple check is fine for N=128
            # We need to ensure no a,b in S such that gcd(a,b)=gcd(a,x)=gcd(b,x)
            # Let's just check all pairs in S.
            # To speed up, we can break early.
            for a, b in combinations(list(S), 2):
                if g[a][b] == g[a][x] == g[b][x]:
                    valid = False
                    break
            if valid:
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = S

    # Final polish: try to add back removed numbers?
    # The local search above only removes. 
    # A "swap" local search could improve it, but removal is robust.
    
    return list(best_set)
# EVOLVE-BLOCK-END
