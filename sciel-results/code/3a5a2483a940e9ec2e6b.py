"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all
    distinct pairs, maximizing cardinality.
    
    Strategy:
    1. Start with the set of multiples of 3 in [1,255] (size 85).
    2. Build a conflict graph where vertices are numbers in [1,255] that share
       a factor with 255 (i.e., divisible by 3, 5, or 17), and edges connect
       pairs with gcd=1.
    3. Use Bron-Kerbosch with pivot to find maximum independent set (which is
       maximum clique in the "compatibility" graph) within the subgraph of
       numbers sharing a factor with 255.
    4. Also try greedy heuristics and local search to find larger sets.
    5. Return the best valid set found.
    """
    start_time = time.time()
    deadline = start_time + 75  # 75s deadline to be safe within 90s
    
    # 255 = 3 * 5 * 17
    # Numbers that share a factor with 255 are those divisible by 3, 5, or 17
    # But we can also include numbers that don't share a factor with 255,
    # as long as they're pairwise non-coprime with everything else.
    
    # Key insight: If we pick a set where every element shares a common factor,
    # that's easy. The largest such set is multiples of 3 (85 elements) or
    # multiples of 5 (51 elements) or multiples of 17 (15 elements).
    # 
    # But we can potentially do better with a "mixed" set where different
    # elements share different factors, as long as all pairs have gcd > 1.
    
    # Let's think about this more carefully.
    # 255 = 3 * 5 * 17
    # 
    # A number in our set must have gcd > 1 with 255, OR if it doesn't, it must
    # have gcd > 1 with every other element in the set.
    # 
    # Actually, every element a in the set must have gcd(a, 255) > 1 is NOT required.
    # What's required is gcd(a, b) > 1 for all distinct a, b in the set, and 255 is in the set.
    # So every element a must have gcd(a, 255) > 1. This means every element must be
    # divisible by 3, 5, or 17.
    
    # So our candidate pool is: {x in [1,255] : gcd(x, 255) > 1}
    # = {x : 3|x or 5|x or 17|x}
    
    # Let's count: |multiples of 3| = 85, |multiples of 5| = 51, |multiples of 17| = 15
    # |mult of 3 or 5| = 85 + 51 - 17 = 119 (mult of 15: 17)
    # |mult of 3 or 17| = 85 + 15 - 5 = 95 (mult of 51: 5)
    # |mult of 5 or 17| = 51 + 15 - 3 = 63 (mult of 85: 3)
    # |mult of 3, 5, or 17| = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127 (mult of 255: 1)
    
    # So we have 127 candidates. We need to find the largest subset where all pairs
    # have gcd > 1.
    
    # Build the candidate set
    candidates = [x for x in range(1, 256) if gcd(x, 255) > 1]
    n = len(candidates)
    
    # Build compatibility matrix: compat[i][j] = True if gcd(candidates[i], candidates[j]) > 1
    # We want maximum clique in this graph.
    
    # For efficiency, let's use bitmasks or adjacency lists
    # n = 127, so adjacency matrix is 127x127 = 16129 entries, very manageable
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i].add(j)
                adj[j].add(i)
    
    # The multiples of 3 form a clique of size 85. Let's verify.
    # All multiples of 3 have gcd >= 3, so yes, they form a clique.
    
    # Now, can we do better than 85? Let's think about what additional elements
    # we could add to the multiples-of-3 set.
    # An element x not divisible by 3 but divisible by 5 or 17 could be added
    # if it has gcd > 1 with every multiple of 3 in our set.
    # 
    # If x is divisible by 5 (but not 3), then gcd(x, y) > 1 for a multiple of 3 y
    # only if y is also divisible by 5, or if x and y share another factor.
    # Since y is a multiple of 3, y = 3k. gcd(x, 3k) > 1 means either 5|3k (i.e., 5|k,
    # so 15|y) or x and 3k share some other prime factor.
    # 
    # This is getting complex. Let's just run Bron-Kerbosch on the 127-node graph.
    
    # Bron-Kerbosch with pivot for maximum clique
    best_clique = []
    best_size = 0
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_size
        
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
        
        # Pivot selection: choose vertex u from P ∪ X with maximum |N(u) ∩ P|
        # This minimizes the number of recursive calls
        candidates_pivot = list(P | X)
        if not candidates_pivot:
            return
        
        # Pick pivot with max degree in P
        pivot = max(candidates_pivot, key=lambda u: len(adj[u] & P))
        
        # We only need to recurse on P \ N(pivot)
        for v in list(P - adj[pivot]):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
            if time.time() > deadline:
                return
    
    # Initialize
    initial_R = set()
    initial_P = set(range(n))
    initial_X = set()
    
    # To help with performance, let's order vertices by degree (descending)
    # and use a better initialization
    
    # Actually, let's first try a greedy approach to get a good initial solution
    # and use it to prune.
    
    # Greedy: start with multiples of 3, try to add others
    mult3 = [i for i in range(n) if candidates[i] % 3 == 0]
    current_set = set(mult3)
    
    # Try to add elements not in mult3
    for i in range(n):
        if i not in current_set:
            # Check if i is compatible with all in current_set
            if all(j in adj[i] for j in current_set):
                current_set.add(i)
    
    best_size = len(current_set)
    best_clique = list(current_set)
    
    # Now run Bron-Kerbosch to verify if we can do better
    # But with 127 nodes, this might be slow. Let's try a limited version.
    
    # Alternative: use a maximum clique algorithm with branch and bound
    # Let's try a different approach: since the graph is structured,
    # let's analyze it more carefully.
    
    # Key observation: The graph is the "compatibility" graph where edges
    # connect numbers sharing a factor. Let's categorize candidates by their
    # prime factors (among 3, 5, 17, and others).
    
    # Let's try a local search / hill climbing approach
    # Start with multiples of 3, try swaps
    
    # First, let's see if we can add any element to the multiples-of-3 set
    # that's compatible with all multiples of 3.
    
    # For an element x to be compatible with ALL multiples of 3 in [1,255],
    # x must share a factor with every multiple of 3.
    # The multiples of 3 include 3, 6, 9, 12, ...
    # gcd(x, 3) > 1 requires 3|x. But if x is divisible by 3, it's already in our set.
    # So no element outside the multiples of 3 can be compatible with ALL multiples of 3.
    
    # This means the maximum set containing all multiples of 3 is exactly the multiples of 3.
    # But maybe a set that doesn't contain all multiples of 3 is larger?
    
    # Let's think: if we drop some multiples of 3, can we add more elements?
    # Each dropped multiple of 3 might allow adding 1-2 non-multiples-of-3.
    # This is unlikely to be beneficial unless we can add many.
    
    # Let's try a more systematic search:
    # Partition candidates by their gcd with 255:
    # - Type 3: divisible by 3 only (not 5, not 17)
    # - Type 5: divisible by 5 only
    # - Type 17: divisible by 17 only
    # - Type 15: divisible by 3 and 5
    # - Type 51: divisible by 3 and 17
    # - Type 85: divisible by 5 and 17
    # - Type 255: divisible by 3, 5, 17 (just 255)
    
    # Let's count each type
    type3 = [x for x in candidates if x % 3 == 0 and x % 5 != 0 and x % 17 != 0]
    type5 = [x for x in candidates if x % 5 == 0 and x % 3 != 0 and x % 17 != 0]
    type17 = [x for x in candidates if x % 17 == 0 and x % 3 != 0 and x % 5 != 0]
    type15 = [x for x in candidates if x % 15 == 0 and x % 17 != 0]
    type51 = [x for x in candidates if x % 51 == 0 and x % 5 != 0]
    type85 = [x for x in candidates if x % 85 == 0 and x % 3 != 0]
    type255 = [x for x in candidates if x % 255 == 0]
    
    # Let's try to construct a larger set by being selective.
    # Strategy: include all type15, type51, type85, type255 (they're "universal" in some sense)
    # Then add type3, type5, type17 selectively.
    
    # Actually, let's just run a proper maximum clique search with time limit.
    # Use a branch-and-bound approach.
    
    # Order vertices by degree for better pruning
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    
    # Remap indices
    remap = {old: new for new, old in enumerate(order)}
    adj2 = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            adj2[remap[i]].add(remap[j])
    
    candidates2 = [candidates[order[i]] for i in range(n)]
    
    best_size2 = 0
    best_clique2 = []
    
    def bk2(R, P, X):
        nonlocal best_size2, best_clique2
        
        if time.time() > deadline:
            return
        
        if not P and not X:
            if len(R) > best_size2:
                best_size2 = len(R)
                best_clique2 = list(R)
            return
        
        # Upper bound: |R| + |P|
        if len(R) + len(P) <= best_size2:
            return
        
        # Pivot
        pivot_candidates = list(P | X)
        if not pivot_candidates:
            return
        
        pivot = max(pivot_candidates, key=lambda u: len(adj2[u] & P))
        
        for v in list(P - adj2[pivot]):
            bk2(R | {v}, P & adj2[v], X & adj2[v])
            P.remove(v)
            X.add(v)
            if time.time() > deadline:
                return
    
    bk2(set(), set(range(n)), set())
    
    # Map back to original
    if best_size2 > best_size:
        best_size = best_size2
        best_clique = [candidates2[i] for i in best_clique2]
    
    # Also try a greedy + local search approach
    # Start with multiples of 3
    current = set(mult3)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding elements
        for i in range(n):
            if i not in current and all(j in adj[i] for j in current):
                current.add(i)
                improved = True
                break
        # Try swaps: remove one, add two
        for i in list(current):
            for j in range(n):
                if j in current:
                    continue
                # Try removing i and adding j
                new_set = (current - {i}) | {j}
                if all(a in adj[b] or a == b for a, b in [(x, y) for x in new_set for y in new_set if x != y]):
                    # Check pairwise
                    valid = True
                    lst = list(new_set)
                    for a in range(len(lst)):
                        for b in range(a + 1, len(lst)):
                            if lst[b] not in adj[lst[a]]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        current = new_set
                        improved = True
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            if improved:
                break
    
    if len(current) > best_size:
        best_size = len(current)
        best_clique = list(current)
    
    # Convert back to values
    result = [candidates[i] for i in best_clique]
    
    # Verify 255 is in the result
    if 255 not in result:
        # Find index of 255 in candidates
        idx_255 = candidates.index(255)
        if idx_255 not in best_clique:
            # Try to add it
            if all(gcd(255, candidates[i]) > 1 for i in best_clique):
                result.append(255)
            else:
                # Replace an incompatible element
                for i in best_clique:
                    if gcd(255, candidates[i]) == 1:
                        best_clique.remove(i)
                        result = [candidates[j] for j in best_clique]
                        result.append(255)
                        break
    
    # Final verification
    result = list(set(result))
    assert 255 in result
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert gcd(result[i], result[j]) > 1, f"gcd({result[i]}, {result[j]}) = {gcd(result[i], result[j])}"
    
    return result
# EVOLVE-BLOCK-END
