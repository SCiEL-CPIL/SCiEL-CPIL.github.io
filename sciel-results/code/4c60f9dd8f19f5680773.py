"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from math import gcd

def construct():
    n = 256
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Build conflict graph: edge between i and j if gcd(i,j) > 1
    # Vertices are integers 1..256
    # We want max weight independent set where weight(v) = v
    
    # Precompute gcd conflicts
    # For each number, which other numbers conflict with it
    conflict = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        for j in range(i + 1, n + 1):
            if gcd(i, j) > 1:
                conflict[i].add(j)
                conflict[j].add(i)
    
    # Greedy initialization: sort by descending value, add if no conflict with selected
    def greedy_init():
        selected = set()
        for v in range(n, 0, -1):
            if not any(c in selected for c in conflict[v]):
                selected.add(v)
        return selected
    
    best_set = greedy_init()
    best_sum = sum(best_set)
    
    # Branch and bound for maximum weight independent set
    # Use greedy coloring upper bound on the subgraph
    
    def greedy_coloring(vertices):
        """Color the induced subgraph on vertices using greedy coloring.
        Returns list of color classes (each a set of vertices)."""
        if not vertices:
            return []
        uncolored = set(vertices)
        colors = []
        while uncolored:
            color_class = set()
            for v in sorted(uncolored, key=lambda x: -x):
                if not any(c in color_class for c in conflict[v]):
                    color_class.add(v)
            uncolored -= color_class
            colors.append(color_class)
        return colors
    
    def color_bound(vertices):
        """Upper bound on weight of independent set in induced subgraph."""
        color_classes = greedy_coloring(vertices)
        bound = 0
        for color_class in color_classes:
            bound += max(color_class)
        return bound
    
    # Branch and bound
    def branch_and_bound(candidates, current_set, current_sum):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
        
        # Upper bound
        ub = current_sum + color_bound(candidates)
        if ub <= best_sum:
            return
        
        # Pick vertex with highest degree in candidates for branching
        # (or highest weight)
        # Sort candidates by descending weight for better bounds
        sorted_cands = sorted(candidates, key=lambda x: -x)
        
        for v in sorted_cands:
            if time.time() > deadline:
                return
            
            # Include v
            new_candidates = {c for c in candidates if c != v and c not in conflict[v]}
            branch_and_bound(new_candidates, current_set | {v}, current_sum + v)
            
            # Exclude v
            candidates.discard(v)
            # Continue with remaining candidates (v excluded)
            # But we need to be careful: we're modifying candidates in place
            # Let's use a different approach
            
        # Actually, let's restructure: pick one vertex, branch include/exclude
        # Pick the vertex with highest weight
        v = sorted_cands[0]
        
        # Branch 1: Include v
        new_candidates = {c for c in candidates if c != v and c not in conflict[v]}
        branch_and_bound(new_candidates, current_set | {v}, current_sum + v)
        
        # Branch 2: Exclude v
        candidates.discard(v)
        branch_and_bound(candidates, current_set, current_sum)
    
    # Start B&B with all vertices as candidates
    # But this might be too slow. Let's use a smarter initial candidate set.
    # Start from greedy solution and try to improve
    
    # Alternative: use the greedy solution as initial, then do local search with B&B
    
    # Let's try a different approach: since the graph has structure (conflict based on primes),
    # we can exploit that. Each number's prime factors determine its conflicts.
    
    # Key insight: if we pick a number, we can't pick any number sharing a prime factor.
    # So the problem decomposes by prime factors.
    
    # For each prime p, we can pick at most one number that is divisible by p.
    # Numbers not divisible by any prime (i.e., 1) can always be picked.
    
    # Actually, a number can be divisible by multiple primes, so it's more complex.
    
    # Let's use a different strategy: group numbers by their "prime signature"
    # Two numbers conflict if they share any prime factor.
    
    # Better approach: use the fact that the maximum independent set can be found
    # by considering for each prime, which numbers to pick.
    
    # Since we have 54 primes up to 256, and each number is a product of prime powers,
    # we can model this as: for each prime p, choose at most one number divisible by p.
    # But numbers can be divisible by multiple primes, creating coupling.
    
    # Let's try a simpler exact approach: branch on primes, not numbers.
    # For each prime p, decide which number divisible by p to include (or none).
    
    # Actually, let's just do a well-optimized B&B with the greedy coloring bound.
    # The key is to order vertices well and use tight bounds.
    
    # Restart with a cleaner B&B implementation
    best_set = greedy_init()
    best_sum = sum(best_set)
    
    # Precompute for each vertex, the set of neighbors
    neighbors = [conflict[i] for i in range(n + 1)]
    
    def bb_search(candidates, current_sum, current_selected):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_selected)
            return
        
        # Upper bound using greedy coloring
        ub = current_sum + color_bound(candidates)
        if ub <= best_sum:
            return
        
        # Select vertex to branch on: highest degree in candidates
        v = max(candidates, key=lambda x: len(neighbors[x] & candidates))
        
        # Branch 1: Include v
        new_cands = candidates - {v} - neighbors[v]
        bb_search(new_cands, current_sum + v, current_selected | {v})
        
        # Branch 2: Exclude v
        new_cands = candidates - {v}
        bb_search(new_cands, current_sum, current_selected)
    
    # Start with all vertices
    all_cands = set(range(1, n + 1))
    bb_search(all_cands, 0, set())
    
    # Final validation
    result = list(best_set)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback
                result = [1] + [p for p in range(2, n + 1) if all(p % q != 0 for q in range(2, int(math.isqrt(p)) + 1))]
                break
        else:
            continue
        break
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
