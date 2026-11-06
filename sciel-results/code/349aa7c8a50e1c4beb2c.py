"""Optimized construction for large pairwise noncoprime sets containing 255."""
import math
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # 255 = 3 * 5 * 17
    # We need a subset of [1, 255] containing 255, with gcd(a,b) > 1 for all distinct pairs.
    # Maximize cardinality.

    # Key structural insight:
    # - Every element must share a factor with 255, so must be divisible by 3, 5, or 17.
    # - If 3 is in the set, every other element must be divisible by 3 (since gcd(3,x)>1 iff 3|x).
    #   Then max size = 85 (all multiples of 3 in [1,255]).
    # - If 3 is NOT in the set, we can potentially include numbers not divisible by 3,
    #   but they must still be compatible with all other elements.
    #   If 5 is in the set (and 3 is not), every element must be divisible by 5. Max = 51.
    #   If neither 3 nor 5 is in the set, elements must be divisible by 17. Max = 15.
    # - But what if we exclude 3 but include some multiples of 3 that are >= 6?
    #   E.g., include 6, 9, 12, ... but not 3. Then can we add 5?
    #   gcd(5, 6) = 1, so no. Can we add 10? gcd(10, 9) = 1, so no.
    #   In fact, if we have 9 in the set, every element must be divisible by 3.
    #   If we have 6 in the set, every element must share a factor with 6, i.e., be divisible by 2 or 3.
    #   But then gcd(255, 2) = 1, so 2 can't be in the set. And any even number not divisible by 3
    #   like 10: gcd(10, 9) = 1 if 9 is in the set.
    
    # The provable maximum is 85, achieved by all multiples of 3.
    # However, let's verify with a graph-based approach to be thorough.
    # Build the graph and use a heuristic to confirm no larger clique exists.

    # Generate all candidates: numbers in [1,255] divisible by 3, 5, or 17
    candidates = [i for i in range(1, 256) if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    
    # Build adjacency using bitsets for fast clique verification
    adj = [0] * n
    for i in range(n):
        ci = candidates[i]
        for j in range(i + 1, n):
            if math.gcd(ci, candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Known valid clique: all multiples of 3
    mult3_indices = [i for i in range(n) if candidates[i] % 3 == 0]
    best_clique = set(mult3_indices)
    best_size = len(best_clique)
    
    # Try to find a larger clique using greedy + local search
    # Strategy: try adding vertices, and if stuck, try different removal patterns
    
    def time_check():
        return time.time() < deadline
    
    def can_add(clique_set, v):
        """Check if vertex v can be added to the clique."""
        for u in clique_set:
            if not ((adj[u] >> v) & 1):
                return False
        return True
    
    # Local search: iteratively try to add any vertex
    improved = True
    while improved and time_check():
        improved = False
        for v in range(n):
            if v in best_clique:
                continue
            if can_add(best_clique, v):
                best_clique.add(v)
                best_size += 1
                improved = True
    
    # Try alternative starting points: exclude small primes
    # Start from multiples of 5 (excluding those that force 3)
    for base_prime in [5, 17]:
        if not time_check():
            break
        base_clique = set(i for i in range(n) if candidates[i] % base_prime == 0)
        # Remove elements that are also multiples of 3 if we want to explore non-3-based sets
        # Actually, keep them all and try to expand
        cur_clique = base_clique.copy()
        improved = True
        while improved and time_check():
            improved = False
            for v in range(n):
                if v in cur_clique:
                    continue
                if can_add(cur_clique, v):
                    cur_clique.add(v)
                    improved = True
        if len(cur_clique) > best_size:
            best_clique = cur_clique
            best_size = len(cur_clique)
    
    # Try a more sophisticated approach: for each possible "core" prime,
    # build the maximal clique around it
    # We've already tried 3, 5, 17 as bases. Let's also try excluding specific small numbers.
    
    # Try: all multiples of 3 except 3, then see if we can add anything
    if time_check():
        cur_clique = set(i for i in range(n) if candidates[i] % 3 == 0 and candidates[i] != 3)
        improved = True
        while improved and time_check():
            improved = False
            for v in range(n):
                if v in cur_clique:
                    continue
                if can_add(cur_clique, v):
                    cur_clique.add(v)
                    improved = True
        if len(cur_clique) > best_size:
            best_clique = cur_clique
            best_size = len(cur_clique)
    
    # Convert back to actual numbers
    result = sorted([candidates[i] for i in best_clique])
    
    # Final verification
    assert 255 in result, "255 must be in the result"
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert math.gcd(result[i], result[j]) > 1, f"gcd({result[i]},{result[j]})=1"
    
    return result

# EVOLVE-BLOCK-END
