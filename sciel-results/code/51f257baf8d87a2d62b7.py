"""Direct structural approach for equal-LCM-triple-free subset of [1,128]."""
import time
import random
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 85
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure: for each element, which pairs with it form bad triples
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
    
    # Build adjacency: for each element x, which elements y conflict with x
    conflict_adj = {x: set() for x in range(1, N + 1)}
    for x in range(1, N + 1):
        for (y, z) in conflict_pairs[x]:
            conflict_adj[x].add(y)
            conflict_adj[x].add(z)
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for (y, z) in conflict_pairs[x]:
            if y in S and z in S:
                return False
        return True
    
    def greedy_fill(S, order=None):
        """Fill S by adding elements in given order (or natural order)."""
        if order is None:
            order = elements
        for x in order:
            if x not in S and can_add(x, S):
                S.add(x)
        return S
    
    def local_search(S, max_rounds=50):
        """Try to improve S via 1-out-1-in swaps and additions."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            # Try additions
            for x in elements:
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
            # Try swaps
            for x in list(S):
                S.remove(x)
                best_gain = -1
                best_elem = None
                for y in elements:
                    if y != x and y not in S and can_add(y, S):
                        S.add(y)
                        count = len(S)
                        S_copy = set(S)
                        for z in elements:
                            if z not in S_copy and can_add(z, S_copy):
                                S_copy.add(z)
                        gain = len(S_copy) - len(S)
                        if gain > best_gain:
                            best_gain = gain
                            best_elem = y
                        S.remove(y)
                if best_elem is not None and best_gain >= 0:
                    S.add(best_elem)
                    improved = True
                else:
                    S.add(x)
        return S
    
    def fractional_matching_lb(hyperedges):
        """
        Compute a lower bound on the minimum vertex cover of a 3-uniform hypergraph
        using a greedy fractional matching approach.
        
        Returns: lower bound on min vertex cover size
        """
        if not hyperedges:
            return 0
        
        # Greedy fractional matching: repeatedly find a hyperedge with minimum
        # total weight, assign weight 1/3 to each vertex, remove covered hyperedges
        vertex_weights = {v: 0.0 for v in range(1, N + 1)}
        remaining = list(hyperedges)
        total_weight = 0.0
        
        while remaining:
            # Find hyperedge with minimum sum of current vertex weights
            min_edge = None
            min_sum = float('inf')
            for edge in remaining:
                s = sum(vertex_weights[v] for v in edge)
                if s < min_sum:
                    min_sum = s
                    min_edge = edge
            
            if min_edge is None:
                break
            
            # Assign weight 1/3 to each vertex in this hyperedge
            for v in min_edge:
                vertex_weights[v] += 1.0 / 3.0
            
            total_weight += 1.0 / 3.0
            
            # Remove all hyperedges that intersect with min_edge
            new_remaining = []
            for edge in remaining:
                if not any(v in min_edge for v in edge):
                    new_remaining.append(edge)
            remaining = new_remaining
        
        # The lower bound is the maximum vertex weight (by duality)
        return max(vertex_weights.values()) if vertex_weights else 0
    
    def get_active_hyperedges(candidates, S):
        """Get all bad triples where all three elements are in candidates ∪ S."""
        active = []
        for (a, b, c) in bad_triples:
            if a in candidates and b in candidates and c in candidates:
                active.append((a, b, c))
        return active
    
    def branch_and_bound_v2(S, candidates, best_size, best_set, depth=0):
        """
        Deterministic branch-and-bound with fractional matching lower bound.
        
        Uses the algebraic sparsity of LCM conflicts to prune effectively.
        """
        if time.time() > deadline - 5:
            return best_size, best_set
        
        if not candidates:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return best_size, best_set
        
        # Quick upper bound
        if len(S) + len(candidates) <= best_size:
            return best_size, best_set
        
        # Compute active hyperedges (bad triples entirely within candidates)
        active_hypers = get_active_hyperedges(candidates, S)
        
        # Fractional matching lower bound on min vertex cover
        # This gives us a lower bound on how many elements we MUST exclude
        if active_hypers:
            lb = fractional_matching_lb(active_hypers)
            # Upper bound on what we can add: |candidates| - lb
            upper_bound = len(S) + len(candidates) - int(lb) + 1  # +1 for safety
            if upper_bound <= best_size:
                return best_size, best_set
        
        # Choose branching variable: element with highest "conflict density"
        # Count for each candidate x: number of active hyperedges containing x
        conflict_density = {x: 0 for x in candidates}
        for (a, b, c) in active_hypers:
            conflict_density[a] += 1
            conflict_density[b] += 1
            conflict_density[c] += 1
        
        # Pick the element with highest conflict density
        best_x = max(candidates, key=lambda x: (conflict_density[x], x))
        
        # Branch 1: Include best_x (if valid)
        if can_add(best_x, S):
            S_new = S | {best_x}
            candidates_new = set()
            for x in candidates:
                if x != best_x and can_add(x, S_new):
                    candidates_new.add(x)
            best_size, best_set = branch_and_bound_v2(S_new, candidates_new, best_size, best_set, depth + 1)
        
        # Branch 2: Exclude best_x
        candidates_new = set(candidates)
        candidates_new.discard(best_x)
        best_size, best_set = branch_and_bound_v2(S, candidates_new, best_size, best_set, depth + 1)
        
        return best_size, best_set
    
    best = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    triple_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    orderings = [
        sorted(elements, key=lambda x: (triple_count[x], x)),
        sorted(elements, key=lambda x: (triple_count[x], -x)),
        sorted(elements, key=lambda x: x),
        sorted(elements, key=lambda x: -x),
    ]
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        S = set()
        greedy_fill(S, order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Strategy 2: Branch and bound with fractional matching bound
    if time.time() < deadline - 25:
        S_start = set(best)
        candidates = set()
        for x in elements:
            if x not in S_start and can_add(x, S_start):
                candidates.add(x)
        
        if candidates:
            best_size, best = branch_and_bound_v2(S_start, candidates, best_size, best)
    
    # Strategy 3: Randomized greedy with local search
    random.seed(42)
    num_starts = 100
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        greedy_fill(S, order)
        S = local_search(S, max_rounds=30)
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Final fill
    if time.time() < deadline - 3:
        greedy_fill(best)
    
    if not best:
        best = {1}
    
    return list(best)
