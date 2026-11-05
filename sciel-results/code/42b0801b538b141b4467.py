"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from itertools import combinations
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    # Pre-compute all 4-subsets of [1,16]
    all_sets = [frozenset(c) for c in combinations(range(1, 17), 4)]
    
    # For each pair of sets, pre-compute their intersection
    # We need to check: for any three distinct sets A, B, C, 
    # not (A∩B == A∩C == B∩C)
    
    # Strategy: greedy with ordering that maximizes diversity
    # Use a heuristic: try to add sets that are "far" from existing ones
    
    chosen = []
    chosen_set = set()
    
    # Pre-compute intersections between all pairs
    # This is 1820 * 1820 / 2 ≈ 1.6M pairs, too many to precompute all
    # Instead, for each new candidate, check against all pairs in chosen
    
    # Better: for each pair (a,b) in chosen, compute a∩b
    # For new set s, check if s∩a == s∩b == a∩b for any pair (a,b)
    
    # Let's use a different approach: 
    # For each set s, and each element x in [1,16], track which sets contain x
    # Actually, let's just do the greedy but with a better ordering
    
    # Order candidates by some heuristic to get a good initial family
    # Try random restarts with different orderings
    
    best_family = []
    best_size = 0
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Shuffle the order of candidates
        candidates = list(all_sets)
        import random
        random.seed(trial)
        random.shuffle(candidates)
        
        current = []
        current_frozen = set()
        
        for s in candidates:
            if time.time() > deadline:
                break
            
            # Check if s can be added
            valid = True
            for i in range(len(current)):
                if not valid:
                    break
                a = current[i]
                sa = a & s
                for j in range(i+1, len(current)):
                    b = current[j]
                    sb = b & s
                    ab = a & b
                    if sa == sb == ab:
                        valid = False
                        break
            
            if valid:
                current.append(s)
                current_frozen.add(s)
        
        if len(current) > best_size:
            best_size = len(current)
            best_family = list(current)
        
        # Early termination if we hit the max
        if best_size >= 250:
            break
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
