"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # 1. Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    def get_sum(s):
        return sum(s)
    
    # 2. Initial Solution: Greedy by value
    # We start with 1. Then we try to add numbers in descending order of value.
    # If a number is compatible with all currently selected numbers, we add it.
    # This is a simple heuristic to get a valid baseline.
    
    # A better initial solution might be to prioritize numbers with fewer conflicts or higher value.
    # Let's try sorting by value descending, but maybe we can do better.
    # Actually, since we want to maximize sum, larger numbers are better.
    # But large numbers often have more factors (e.g., 256=2^8, 240=2^4*3*5).
    # Primes are very good candidates because they only conflict with their multiples.
    
    # Let's try a greedy approach that picks the largest compatible number available.
    # Candidates: 1..N
    # Current set: empty
    # While candidates exist:
    #   Pick candidate c with max value such that c is compatible with all in current set.
    #   Add c to current set, remove c and its incompatible neighbors from candidates.
    
    # This is the standard greedy for Maximum Weight Independent Set.
    # It's fast but might not be optimal.
    
    # Let's implement this greedy first to get a strong baseline.
    
    # We represent the set of available candidates.
    # To make it efficient, we can just iterate through sorted candidates.
    
    # However, the "pick max compatible" approach requires checking compatibility with the current set.
    # Since the set grows, this is O(N^2) roughly, which is fine for N=256.
    
    # Let's refine:
    # candidates = list(range(1, N+1))
    # current_set = []
    # while candidates:
    #    best = None
    #    best_val = -1
    #    for c in candidates:
    #        if all(is_compatible(c, s) for s in current_set):
    #            if c > best_val:
    #                best_val = c
    #                best = c
    #    if best is None: break
    #    current_set.append(best)
    #    # Remove best and all incompatible from candidates
    #    candidates = [c for c in candidates if c != best and is_compatible(c, best)]
    
    # This greedy is O(N^3) in worst case (N iterations, N candidates, N set size check).
    # 256^3 is ~16M, which is fine in Python if optimized, but might be slow.
    # Let's optimize the compatibility check using bitsets or precomputed masks?
    # Or just use the prime factor sets intersection which is fast.
    
    # Let's try a different greedy: Sort by value descending. Iterate. If compatible with current set, add.
    # This is O(N^2) and simpler.
    
    # sorted_candidates = sorted(range(1, N+1), reverse=True)
    # current_set = []
    # for c in sorted_candidates:
    #     if all(is_compatible(c, s) for s in current_set):
    #         current_set.append(c)
    
    # This is very fast. Let's use this as the baseline.
    
    # But wait, the "pick max compatible from remaining" is generally better than "iterate sorted and pick if fits".
    # Example: 10, 6, 5.
    # Sorted: 10, 6, 5.
    # 10: add. Set={10}.
    # 6: gcd(10,6)=2, no.
    # 5: gcd(10,5)=5, no.
    # Result: {10}. Sum=10.
    # 
    # Pick max compatible:
    # Candidates: 10, 6, 5.
    # Max is 10. Compatible with empty. Add 10.
    # Remove 10 and incompatible (6, 5). Candidates empty.
    # Result: {10}.
    
    # Example: 6, 10, 15.
    # Sorted: 10, 6, 15.
    # 10: add. Set={10}.
    # 6: no.
    # 15: gcd(10,15)=5, no.
    # Result: {10}. Sum=10.
    
    # Pick max compatible:
    # Candidates: 10, 6, 15.
    # Max 10. Add. Remove 10, 6, 15 (10 conflicts with 6,15).
    # Result: {10}.
    
    # Example: 6, 35, 10.
    # Sorted: 35, 10, 6.
    # 35: add. Set={35}.
    # 10: gcd(35,10)=5, no.
    # 6: gcd(35,6)=1, yes. Add. Set={35, 6}.
    # Result: {35, 6}. Sum=41.
    
    # Pick max compatible:
    # Candidates: 35, 10, 6.
    # Max 35. Add. Remove 35, 10 (conflict). Candidates: 6.
    # Max 6. Add.
    # Result: {35, 6}.
    
    # It seems similar. But consider: 100, 99, 98.
    # 100=2^2*5^2. 99=9*11. 98=2*7^2.
    # 100 conflicts with 98 (gcd 2). 99 conflicts with neither? gcd(100,99)=1, gcd(99,98)=1.
    # Sorted: 100, 99, 98.
    # 100: add.
    # 99: yes. Add. Set={100, 99}.
    # 98: gcd(100,98)=2, no.
    # Result: {100, 99}. Sum=199.
    
    # Pick max compatible:
    # Max 100. Add. Remove 100, 98. Candidates: 99.
    # Max 99. Add.
    # Result: {100, 99}.
    
    # Okay, let's implement the "Pick max compatible" greedy as it's theoretically slightly more robust for MWIS.
    
    # Actually, for speed, the "iterate sorted" is O(N^2) and "pick max" is O(N^3) if not careful.
    # With N=256, O(N^3) is 16M ops. In Python, 16M simple ops is ~1-2 seconds.
    # The compatibility check `is_compatible` involves set intersection.
    # Let's stick to the simpler "iterate sorted" first, and if time permits, try local search.
    
    # Let's use the simple greedy:
    # Sort by value descending.
    # If compatible with all selected, select.
    
    selected = []
    for c in range(N, 0, -1):
        if all(is_compatible(c, s) for s in selected):
            selected.append(c)
    
    best_sum = sum(selected)
    best_set = selected[:]
    
    # 3. Local Search / Hill Climbing
    # Try to swap elements or add/remove to improve sum.
    # Since we want to maximize sum, and we started with large numbers,
    # we might have missed a combination of slightly smaller numbers that fits better.
    
    # A common move:
    # 1. Try to add any number not in set.
    # 2. Try to replace a number in set with a larger number not in set (if compatible).
    # 3. Try to replace two numbers with three? (Complex)
    
    # Let's try a simple randomized local search.
    # We have a budget.
    
    # Precompute compatibility matrix for faster checks?
    # comp[a][b] = 1 if compatible, 0 otherwise.
    # 256x256 matrix is 65k entries. Fast to build.
    
    comp = [[False] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if is_compatible(a, b):
                comp[a][b] = True
                comp[b][a] = True
        comp[a][a] = True # Self compatible
    
    # Helper to check if x can be added to set S
    def can_add(x, S):
        for s in S:
            if not comp[x][s]:
                return False
        return True
    
    # Helper to check if x can replace y in set S
    # x must be compatible with all s in S \ {y}
    def can_replace(x, y, S):
        for s in S:
            if s != y and not comp[x][s]:
                return False
        return True
    
    # Local Search
    improved = True
    while improved:
        improved = False
        # Try adding any number
        for x in range(N, 0, -1):
            if x not in selected:
                if can_add(x, selected):
                    selected.append(x)
                    improved = True
                    # Since we added a positive number, sum increased.
                    # We can break and restart the loop? Or continue?
                    # Breaking and restarting ensures we find the best addition.
                    # But we might add multiple.
                    # Let's just mark improved and break to re-evaluate.
                    break
        
        if improved:
            continue
            
        # Try replacing
        # For each y in selected, try to find x > y such that x can replace y.
        # To be efficient, iterate x in descending order.
        for x in range(N, 0, -1):
            if x in selected:
                continue
            for y in selected:
                if x > y:
                    if can_replace(x, y, selected):
                        # Replace y with x
                        selected.remove(y)
                        selected.append(x)
                        improved = True
                        break
            if improved:
                break
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Update best
    if sum(selected) > best_sum:
        best_sum = sum(selected)
        best_set = selected[:]
    
    # If we still have time, try randomized restarts or different greedy strategies.
    # But the above should be quite good.
    
    # Let's verify the result is valid.
    # All distinct? Yes, we use a list and check `in`.
    # All coprime? Yes, by construction.
    # Ints 1..256? Yes.
    
    # Return as list of ints.
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
