"""Optimized program: exact maximum weight independent set via branch-and-bound."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n = 511
    # Compute weights: 1/(512-a) for a in 1..511
    # We want to maximize sum of weights over pairwise coprime set
    # Conflicts: two numbers conflict if gcd > 1
    
    # Precompute conflict graph
    # For efficiency, use prime factorization
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    numbers = list(range(1, n + 1))
    factors = [prime_factors(a) for a in numbers]
    
    # Build conflict graph: conflict[i][j] = True if gcd(numbers[i], numbers[j]) > 1
    # For efficiency, use bitsets or adjacency lists
    # Two numbers conflict if they share a prime factor
    # Group by prime: numbers sharing a prime are mutually conflicting
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    # Map from prime to list of indices
    prime_to_indices = {}
    for i in range(n):
        for p in factors[i]:
            if p not in prime_to_indices:
                prime_to_indices[p] = []
            prime_to_indices[p].append(i)
    
    for p, indices in prime_to_indices.items():
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                adj[indices[i]].add(indices[j])
                adj[indices[j]].add(indices[i])
    
    # Convert to sorted lists for faster iteration
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    # Weights
    weights = [1.0 / (512 - (i+1)) for i in range(n)]
    
    # Branch and bound for maximum weight independent set
    # Use greedy coloring for upper bound
    
    def greedy_coloring(vertex_set, adj):
        """Assign colors to vertices in vertex_set such that adjacent vertices have different colors.
        Returns list of color classes (each a set of vertices)."""
        if not vertex_set:
            return []
        
        uncolored = set(vertex_set)
        color_classes = []
        
        while uncolored:
            color_class = set()
            # Greedily pick vertices that don't conflict with each other in this class
            candidates = list(uncolored)
            candidates.sort(key=lambda v: len(adj[v] & uncolored), reverse=True)
            
            for v in candidates:
                if v not in uncolored:
                    continue
                # Check if v conflicts with any already in color_class
                if all(v not in adj[u] for u in color_class):
                    color_class.add(v)
                    uncolored.discard(v)
            
            if color_class:
                color_classes.append(color_class)
        
        return color_classes
    
    best_weight = 0.0
    best_set = set()
    
    def branch_and_bound(candidates, current_set, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(current_set)
            return
        
        # Upper bound: sum of weights of all remaining candidates
        # Better bound: greedy coloring bound
        # For each color class, at most one vertex can be selected
        # So upper bound = current_weight + sum of max weight in each color class
        
        color_classes = greedy_coloring(candidates, adj)
        bound = current_weight
        for cc in color_classes:
            max_w = max(weights[v] for v in cc)
            bound += max_w
        
        if bound <= best_weight:
            return
        
        # Select a vertex to branch on
        # Heuristic: pick vertex with highest weight / degree ratio
        # Or just pick the one with highest weight
        v = max(candidates, key=lambda x: weights[x])
        
        # Branch 1: include v
        new_candidates = candidates - {v}
        # Remove all neighbors of v
        for u in adj[v]:
            new_candidates.discard(u)
        
        current_set.add(v)
        branch_and_bound(new_candidates, current_set, current_weight + weights[v])
        current_set.discard(v)
        
        # Branch 2: exclude v
        branch_and_bound(candidates - {v}, current_set, current_weight)
    
    all_indices = set(range(n))
    branch_and_bound(all_indices, set(), 0.0)
    
    # Convert back to actual numbers
    result = [i + 1 for i in sorted(best_set)]
    
    # Verify pairwise coprime
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to greedy if something went wrong
                chosen = []
                for a in range(511, 0, -1):
                    if all(gcd(a, b) == 1 for b in chosen):
                        chosen.append(a)
                return chosen
    
    return result
# EVOLVE-BLOCK-END
