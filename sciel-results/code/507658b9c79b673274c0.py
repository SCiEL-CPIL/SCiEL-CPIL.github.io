import math
import random
import time

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # We need a subset of [1, 255] containing 255, with gcd(a,b) > 1 for all distinct pairs.
    # 255 = 3 * 5 * 17
    # The "forced-maximum-element" formulation suggests we should focus on numbers sharing factors with 255.
    # Any number in the set must share a factor with 255 (otherwise gcd with 255 is 1).
    # So candidates are numbers in [1, 255] that are divisible by 3, 5, or 17.
    
    # Let's verify: if x is in the set, gcd(x, 255) > 1, so x shares a factor with 255.
    # The factors of 255 are 3, 5, 17.
    # So x must be divisible by at least one of 3, 5, 17.
    
    candidates = set()
    for x in range(1, 256):
        if x % 3 == 0 or x % 5 == 0 or x % 17 == 0:
            candidates.add(x)
    
    # 255 must be in the set.
    # We want to maximize the size of the subset S ⊆ candidates such that:
    # 1. 255 ∈ S
    # 2. For all a, b ∈ S, a ≠ b, gcd(a, b) > 1
    
    # This is equivalent to finding a maximum clique in a graph where:
    # - Vertices are candidates
    # - Edge between a, b if gcd(a, b) > 1
    # - We must include vertex 255
    
    # Since 255 is connected to all candidates (they all share a factor with 255),
    # we can just find the maximum clique containing 255.
    # Actually, we need the clique to be a subset of candidates that all pairwise have gcd > 1.
    # Since all candidates share a factor with 255, 255 is adjacent to all of them.
    # But two candidates a, b might not share a factor with each other.
    # E.g., a = 3, b = 5: gcd(3, 5) = 1. So 3 and 5 cannot both be in the set.
    
    # Strategy: Partition candidates by their "type" based on which prime factors of 255 they have.
    # Type 3: divisible by 3 only (not 5, not 17)
    # Type 5: divisible by 5 only
    # Type 17: divisible by 17 only
    # Type 3,5: divisible by 3 and 5, not 17
    # Type 3,17: divisible by 3 and 17, not 5
    # Type 5,17: divisible by 5 and 17, not 3
    # Type 3,5,17: divisible by all three (i.e., divisible by 255)
    
    # Two numbers are compatible (gcd > 1) if they share at least one prime factor from {3, 5, 17}.
    # Wait, that's not quite right. They need gcd > 1, which could be from any prime, not just 3, 5, 17.
    # But since both are in candidates, they each have at least one of {3, 5, 17} as a factor.
    # If they share one of {3, 5, 17}, then gcd > 1.
    # If they don't share any of {3, 5, 17}, they might still share another factor.
    # E.g., a = 3 * 7 = 21, b = 5 * 7 = 35. gcd(21, 35) = 7 > 1. But 21 is type 3, 35 is type 5.
    # So the type-based analysis is not sufficient. We need to check actual gcd.
    
    # Let's build the graph and use a heuristic for maximum clique.
    
    # Precompute gcd for all pairs? There are at most ~100 candidates.
    # Let's count: 
    # Multiples of 3: 85
    # Multiples of 5: 51
    # Multiples of 17: 15
    # Inclusion-exclusion: |A ∪ B ∪ C| = |A| + |B| + |C| - |A∩B| - |A∩C| - |B∩C| + |A∩B∩C|
    # |A∩B| = multiples of 15: 17
    # |A∩C| = multiples of 51: 5
    # |B∩C| = multiples of 85: 3
    # |A∩B∩C| = multiples of 255: 1
    # Total = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127
    
    # 127 candidates. 127*126/2 = ~8000 pairs. Precomputing gcd is fast.
    
    cand_list = sorted(candidates)
    n = len(cand_list)
    
    # Build adjacency: adj[i][j] = True if gcd(cand_list[i], cand_list[j]) > 1
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = True
        for j in range(i + 1, n):
            if math.gcd(cand_list[i], cand_list[j]) > 1:
                adj[i][j] = True
                adj[j][i] = True
    
    # Find the index of 255
    idx_255 = cand_list.index(255)
    
    # We want the maximum clique containing idx_255.
    # Greedy approach: start with {255}, then greedily add vertices that are adjacent to all current vertices.
    # Then try to improve with local search.
    
    # A better heuristic: use the fact that the problem is small.
    # We can try a branch-and-bound or a simple greedy with multiple restarts.
    
    # Let's try a greedy coloring-based upper bound approach (Tomita's algorithm) but simplified.
    # Or just use a simple greedy + local search.
    
    # First, let's try a simple greedy:
    # Start with S = {255}
    # Repeatedly add the vertex v (not in S) that is adjacent to all in S and has the highest degree in the remaining graph.
    
    def greedy_clique():
        S = {idx_255}
        remaining = set(range(n)) - {idx_255}
        
        while remaining:
            # Find candidates that are adjacent to all in S
            valid = []
            for v in remaining:
                if all(adj[v][u] for u in S):
                    valid.append(v)
            
            if not valid:
                break
            
            # Pick the one with highest degree in the induced subgraph on remaining ∪ S
            # Or just pick the one with highest degree overall
            best_v = max(valid, key=lambda v: sum(adj[v][u] for u in range(n)))
            S.add(best_v)
            remaining.discard(best_v)
        
        return S
    
    best_S = greedy_clique()
    best_size = len(best_S)
    
    # Local search: try to swap vertices to increase size
    # Actually, since we want to maximize size, and we started with a greedy, 
    # we can try to add more vertices by removing some and re-adding.
    
    # Let's try a different approach: since the graph is small, we can try to find the maximum clique
    # using a simple recursive branch and bound.
    
    # But 127 vertices might be too large for exact max clique in reasonable time.
    # Let's stick with heuristics.
    
    # Improvement: use a priority queue or sort by degree.
    
    def greedy_clique_v2():
        S = {idx_255}
        remaining = set(range(n)) - {idx_255}
        
        # Precompute degrees
        deg = [sum(adj[i][j] for j in range(n)) for i in range(n)]
        
        while remaining:
            valid = []
            for v in remaining:
                if all(adj[v][u] for u in S):
                    valid.append(v)
            
            if not valid:
                break
            
            # Pick the one with highest degree in the remaining graph
            best_v = max(valid, key=lambda v: sum(adj[v][u] for u in remaining))
            S.add(best_v)
            remaining.discard(best_v)
        
        return S
    
    best_S = greedy_clique_v2()
    best_size = len(best_S)
    
    # Try multiple random orderings
    random.seed(42)
    for _ in range(100):
        # Shuffle the order of consideration
        order = list(range(n))
        random.shuffle(order)
        
        S = {idx_255}
        for v in order:
            if v == idx_255:
                continue
            if all(adj[v][u] for u in S):
                S.add(v)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        if time.time() > deadline - 10:
            break
    
    # Convert back to values
    result = [cand_list[i] for i in best_S]
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
