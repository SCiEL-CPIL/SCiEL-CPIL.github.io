"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a triple system on 16 points without the complete 3-partite K(2,2,2).
    Uses a local search / Max-SAT style approach to maximize the number of triples
    while avoiding any configuration where three disjoint pairs have all eight
    transversal triples present.
    
    A K(2,2,2) configuration is defined by three disjoint pairs of points:
    P1 = {a1, a2}, P2 = {b1, b2}, P3 = {c1, c2}
    The eight transversal triples are all combinations (xi, yi, zi) where
    xi in P1, yi in P2, zi in P3.
    
    We want to maximize the number of triples included such that no such
    complete K(2,2,2) is present.
    """
    n = 16
    num_triples = 560  # C(16,3)
    
    # Precompute all possible triples
    all_triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple = (i, j, k)
                triple_index[triple] = len(all_triples)
                all_triples.append(triple)
    
    # Precompute all K(2,2,2) configurations
    # A K(2,2,2) is determined by three disjoint pairs
    # We need to enumerate all ways to choose 6 distinct points and partition them into 3 pairs
    # Then for each such partition, the 8 transversal triples are determined
    
    k222_configs = []  # Each is a list of 8 triple indices
    
    # Generate all sets of 6 distinct points
    from itertools import combinations, permutations
    
    # For each set of 6 points, generate all ways to partition into 3 pairs
    # There are 15 ways to partition 6 points into 3 unordered pairs
    # Actually, the number of perfect matchings on 6 vertices is 15
    
    def get_pairings(points):
        """Generate all ways to partition 6 points into 3 unordered pairs."""
        points = list(points)
        pairings = []
        
        def rec(remaining, current):
            if not remaining:
                pairings.append(current[:])
                return
            first = remaining[0]
            rest = remaining[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [(first, second)])
        
        rec(points, [])
        return pairings
    
    # To avoid duplicates, we can just iterate over all 6-point subsets and all pairings
    # But we need to be careful about overcounting
    
    seen_configs = set()
    
    for six_points in combinations(range(n), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            # pairing is a list of 3 pairs
            p1, p2, p3 = pairing
            # The 8 transversal triples
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_index[triple]
                        triples_in_config.append(idx)
            
            # Sort the triple indices to create a canonical form
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Now perform local search
    # State: boolean array of length num_triples, True means triple is included
    # Objective: maximize sum of included triples
    # Constraint: for each K(2,2,2) config, not all 8 triples are included
    
    # Start with a random initial solution
    random.seed(42)
    
    best_solution = None
    best_count = -1
    
    # Run multiple restarts
    start_time = time.time()
    time_limit = 100  # Leave some time for the function to return
    
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
        
        # Random initial solution
        solution = [random.random() < 0.7 for _ in range(num_triples)]
        
        # Local search
        for iteration in range(50000):
            if time.time() - start_time > time_limit:
                break
            
            # Count violations and total
            total = sum(solution)
            violations = 0
            for config in k222_configs:
                if all(solution[i] for i in config):
                    violations += 1
            
            # If no violations, this is a valid solution
            if violations == 0:
                if total > best_count:
                    best_count = total
                    best_solution = solution[:]
                # Try to add more triples
                break
            
            # If we have violations, we need to fix them
            # Find a violated config and flip one of its triples to False
            violated_configs = [config for config in k222_configs 
                               if all(solution[i] for i in config)]
            
            if not violated_configs:
                # No violations but we didn't break? Shouldn't happen
                if total > best_count:
                    best_count = total
                    best_solution = solution[:]
                break
            
            # Pick a random violated config
            config = random.choice(violated_configs)
            
            # Try flipping each triple in the config to False
            best_flip = None
            best_new_total = -1
            best_new_violations = float('inf')
            
            for idx in config:
                if solution[idx]:
                    solution[idx] = False
                    new_total = total - 1
                    # Count new violations
                    new_violations = 0
                    for c in k222_configs:
                        if all(solution[i] for i in c):
                            new_violations += 1
                    solution[idx] = True  # Restore
                    
                    # Prefer flips that reduce violations and keep total high
                    if new_violations < best_new_violations or \
                       (new_violations == best_new_violations and new_total > best_new_total):
                        best_new_violations = new_violations
                        best_new_total = new_total
                        best_flip = idx
            
            if best_flip is not None:
                solution[best_flip] = False
                total = best_new_total
            else:
                # Shouldn't happen
                break
        
        # After local search, check if we have a valid solution
        violations = sum(1 for config in k222_configs if all(solution[i] for i in config))
        if violations == 0:
            total = sum(solution)
            if total > best_count:
                best_count = total
                best_solution = solution[:]
    
    # If we never found a valid solution, use a simple heuristic
    if best_solution is None:
        # Fallback: include all triples except those that form K(2,2,2)
        # This is conservative but valid
        best_solution = [False] * num_triples
        # Add triples greedily
        for idx in range(num_triples):
            # Check if adding this triple creates a violation
            best_solution[idx] = True
            valid = True
            for config in k222_configs:
                if idx in config and all(best_solution[i] for i in config):
                    valid = False
                    break
            if not valid:
                best_solution[idx] = False
        best_count = sum(best_solution)
    
    # Convert solution back to list of triples
    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    
    return result
# EVOLVE-BLOCK-END
