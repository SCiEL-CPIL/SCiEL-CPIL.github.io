"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def construct():
    """
    Return a nonempty subset of [1,255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 255
    # 255 = 3 * 5 * 17
    # Any element must share a factor with 255, so must be divisible by 3, 5, or 17.
    
    # Build the candidate pool: all numbers in [1, 255] divisible by 3, 5, or 17
    candidates = set()
    for p in [3, 5, 17]:
        for k in range(1, n // p + 1):
            candidates.add(k * p)
    
    candidates = sorted(candidates)
    
    # We must include 255.
    # For any other element a, we need gcd(255, a) > 1.
    # Since 255 = 3*5*17, this means a must be divisible by 3, 5, or 17.
    # All candidates already satisfy this.
    
    # Now we need gcd(a, b) > 1 for all pairs a, b in the set.
    # This is equivalent to finding a maximum clique in the graph where
    # vertices are candidates and edges connect pairs with gcd > 1.
    # But 255 must be in the set.
    
    # Since the graph is dense (many numbers share factors), let's try a greedy approach
    # with local improvement, or use a branch-and-bound style search.
    
    # First, let's try a simple greedy: start with {255}, then add elements one by one
    # that are compatible with all current elements.
    
    # Actually, let's think about this differently.
    # If all numbers are divisible by 3, they form a clique (gcd >= 3 > 1).
    # Similarly for 5 or 17.
    # But we can mix them as long as cross-pair gcd > 1.
    
    # Let's build the graph and find maximum clique containing 255.
    
    # Precompute gcd matrix for candidates
    cand_list = list(candidates)
    idx = {v: i for i, v in enumerate(cand_list)}
    m = len(cand_list)
    
    # adjacency: adj[i][j] = True if gcd(cand_list[i], cand_list[j]) > 1
    # We can store this as a list of bitsets or just a 2D boolean array
    # m is at most ~150 (numbers divisible by 3, 5, or 17 up to 255)
    
    # Count: divisible by 3: 85, by 5: 51, by 17: 15
    # Union: 85 + 51 + 15 - (div by 15: 17) - (div by 51: 5) - (div by 85: 3) + (div by 255: 1)
    # = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127
    
    # Build adjacency list
    adj = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(cand_list[i], cand_list[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # 255 must be in the set. Find its index.
    i255 = idx[255]
    
    # Maximum clique containing vertex i255.
    # Use branch and bound with a good initial solution.
    
    # Initial solution: all multiples of 3 (including 255)
    # Multiples of 3 up to 255: 3, 6, 9, ..., 255 -> 85 elements
    init_sol = [v for v in cand_list if v % 3 == 0]
    init_size = len(init_sol)
    
    # Try to improve: can we add any multiples of 5 or 17 that aren't multiples of 3?
    # A multiple of 5 (not of 3) has gcd with a multiple of 3:
    # gcd(5k, 3m) = 5k and 3m share no common factor unless k is multiple of 3 or m is multiple of 5.
    # So if we have all multiples of 3, we can add a multiple of 5 (not of 3) only if it shares
    # a factor with ALL multiples of 3 in the set. That's hard.
    
    # Actually, let's just do a proper maximum clique search.
    # Since m ~ 127, we can use a branch and bound approach.
    
    # Let's use a simple recursive branch and bound with ordering by degree.
    
    best = [init_sol]
    
    # For branch and bound, we work with vertex indices.
    # We need a clique containing i255.
    
    # Let's use a different approach: since the graph is dense, try to find a large clique
    # by starting from i255 and greedily adding vertices, then trying local search.
    
    # Greedy: start with {i255}, repeatedly add the vertex with highest degree to current clique.
    current = {i255}
    remaining = set(range(m)) - {i255}
    
    while remaining:
        best_v = -1
        best_deg = -1
        for v in remaining:
            deg = sum(1 for u in current if adj[v][u])
            if deg > best_deg:
                best_deg = deg
                best_v = v
        if best_v == -1 or best_deg < len(current):
            break
        # Check if best_v is compatible with all in current
        if all(adj[best_v][u] for u in current):
            current.add(best_v)
            remaining.discard(best_v)
        else:
            break
    
    greedy_sol = [cand_list[v] for v in current]
    if len(greedy_sol) > len(best[0]):
        best[0] = greedy_sol
    
    # Now try local search: try to swap elements to increase size
    # This is complex; let's try a simpler approach.
    
    # Actually, let's try all "prime-based" cliques:
    # Clique based on prime p: all multiples of p.
    for p in [3, 5, 17]:
        sol = [v for v in cand_list if v % p == 0]
        if 255 in sol and len(sol) > len(best[0]):
            best[0] = sol
    
    # Also try combinations: e.g., all multiples of 3 that are also multiples of 5, plus...
    # This gets complicated. Let's just return the best we have.
    
    # Check time
    if time.time() > deadline:
        return best[0]
    
    # Try to improve with a simple local search
    # For each element not in best, try adding it and removing minimal elements to maintain clique property
    # This is essentially a maximum independent set on the complement, which is hard.
    
    # Let's just return the best solution found.
    return best[0]
# EVOLVE-BLOCK-END
