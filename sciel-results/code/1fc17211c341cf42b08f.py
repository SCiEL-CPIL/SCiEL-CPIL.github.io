"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80  # 80s wall-clock guard (90s limit, 10s margin)
    
    # Try multiple targets to find minimum-count solution
    best_solution = None
    
    # Candidate targets: products of small primes, all denominators must divide target
    # and be >= 5. We want targets with many divisors in [5, target].
    candidates = [2520, 5040, 10080, 7560, 15120, 25200, 50400, 100800, 252000, 504000, 1008000]
    
    for target in candidates:
        if time.time() > deadline:
            break
        
        # Get all divisors of target that are >= 5
        divisors = [d for d in range(5, target + 1) if target % d == 0]
        
        if not divisors:
            continue
        
        # DP: map from partial sum (as fraction of target) to list of denominators
        # We want to minimize the count, so track best (min count) for each sum
        reached = {0: []}
        
        # Process divisors in descending order of weight (ascending denominator)
        # This tends to find solutions with fewer large denominators
        # Actually, process in order that helps find small-count solutions first
        # Sort by weight descending (i.e., denominator ascending) to pack large weights first
        divisors_sorted = sorted(divisors, key=lambda d: target // d, reverse=True)
        
        for d in divisors_sorted:
            if time.time() > deadline:
                break
            w = target // d
            new_reached = {}
            for total, values in reached.items():
                nt = total + w
                if nt <= target:
                    if nt not in new_reached or len(values) + 1 < len(new_reached[nt]):
                        new_reached[nt] = values + [d]
            # Merge: keep the best (shortest) for each sum
            for nt, vals in new_reached.items():
                if nt not in reached or len(vals) < len(reached[nt]):
                    reached[nt] = vals
            
            if target in reached:
                sol = reached[target]
                if best_solution is None or len(sol) < len(best_solution):
                    best_solution = sol
                break  # Found solution for this target, try next target for better
        
        if best_solution is not None and len(best_solution) <= 4:
            break  # Very good solution found
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: original approach
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
