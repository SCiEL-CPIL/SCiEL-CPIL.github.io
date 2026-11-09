import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    p = 101
    denom_range = range(1, 11)
    inv = {x: pow(x, -1, p) for x in denom_range}
    
    # Precompute all subsets grouped by size
    subsets_by_size = {}
    for size in range(1, 11):
        subsets_by_size[size] = []
        for combo in combinations(denom_range, size):
            subsets_by_size[size].append(combo)
    
    # For each size, map residue -> list of subsets achieving it
    # We want to assign to each residue a subset, minimizing max size used
    
    # Strategy: try to cover all residues with subsets of size <= k
    # Start with small k and increase
    
    def try_assign(k_max):
        """Try to assign subsets of size <= k_max to all 101 residues.
        Returns assignment dict or None."""
        # Build candidate pools per residue
        candidates = {r: [] for r in range(p)}
        for size in range(1, k_max + 1):
            for combo in subsets_by_size[size]:
                res = sum(inv[x] for x in combo) % p
                candidates[res].append(combo)
        
        # Check all residues have at least one candidate
        for r in range(p):
            if not candidates[r]:
                return None
        
        # Greedy: process residues with fewest options first
        unassigned = list(range(p))
        assignment = {}
        
        while unassigned:
            # Find residue with fewest remaining candidates
            min_count = float('inf')
            min_idx = -1
            for i, r in enumerate(unassigned):
                # Count candidates that don't conflict (denominators already used in assignment)
                # Actually denominators CAN recur across rows, so no conflict!
                # Just need distinct within a row, which is guaranteed by construction
                cnt = len(candidates[r])
                if cnt < min_count:
                    min_count = cnt
                    min_idx = i
            
            if min_idx == -1:
                break
            
            r = unassigned[min_idx]
            unassigned.pop(min_idx)
            
            # Pick a candidate (prefer smaller subsets)
            # Sort by size, then pick first
            best = None
            for combo in candidates[r]:
                if best is None or len(combo) < len(best):
                    best = combo
            assignment[r] = best
        
        return assignment
    
    # Try increasing k_max
    best_assignment = None
    for k in range(1, 11):
        if time.time() - start_time > DEADLINE:
            break
        result = try_assign(k)
        if result is not None:
            best_assignment = result
            break
    
    if best_assignment is None:
        # Fallback: use seed approach
        representations = {}
        for mask in range(1, 1 << 10):
            terms = [x for x in range(1, 11) if mask >> (x - 1) & 1]
            residue = sum(pow(x, -1, 101) for x in terms) % 101
            representations.setdefault(residue, terms)
        best_assignment = representations
    
    # Try to improve: for residues assigned large subsets, try to find smaller ones
    # that we haven't used yet (though reuse across rows is allowed)
    # Since denominators can recur across rows, we just want smallest per row
    # The greedy already picks smallest available, so this should be optimal
    # for the given k_max.
    
    # But wait - the greedy picks the smallest subset for each residue independently.
    # This is actually optimal since there's no cross-row constraint on denominators.
    # So the answer is just: for each residue, find the smallest subset achieving it.
    
    # Let me verify this is correct and compute directly:
    if time.time() - start_time < DEADLINE:
        # Direct computation: for each residue, find min-size subset
        optimal = {}
        for r in range(p):
            found = False
            for size in range(1, 11):
                for combo in subsets_by_size[size]:
                    if sum(inv[x] for x in combo) % p == r:
                        optimal[r] = combo
                        found = True
                        break
                if found:
                    break
            if not found:
                # Should not happen, but fallback
                optimal[r] = best_assignment[r]
        best_assignment = optimal
    
    return [list(best_assignment[r]) for r in range(101)]
