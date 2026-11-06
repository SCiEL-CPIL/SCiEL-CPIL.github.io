"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute conflict: conflict[i] is set of j where smooth[i] divides smooth[j] or vice versa
    conflict = [set() for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i == j:
                continue
            if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                conflict[i].add(j)
    
    # For each target, find minimal antichain representation
    answers = [None] * 129
    
    def find_min_antichain(target, max_terms=30):
        """Find antichain of minimal size summing to target."""
        # Try k = 1, 2, 3, ... up to max_terms
        for k in range(1, max_terms + 1):
            if time.time() > deadline:
                return None
            found = _search_k(target, k, 0, 0, set())
            if found is not None:
                return found
        return None
    
    def _search_k(target, k, start_idx, current_sum, chosen):
        """Search for antichain of exactly k elements from smooth[start_idx:] summing to target."""
        if time.time() > deadline:
            return None
        
        remaining = target - current_sum
        if k == 0:
            if remaining == 0:
                return list(chosen)
            return None
        
        # Pruning: need k more elements, each at least 1
        if remaining < k:
            return None
        # Pruning: need k more elements, each at most remaining (since all positive)
        # Also, the smallest k elements from start_idx must sum to <= remaining
        if start_idx + k > n_smooth:
            return None
        
        # Try each candidate from start_idx onwards
        for i in range(start_idx, n_smooth):
            if time.time() > deadline:
                return None
            
            val = smooth[i]
            if val > remaining:
                continue
            
            # Check if i conflicts with any already chosen
            valid = True
            for c in chosen:
                if c in conflict[i]:
                    valid = False
                    break
            if not valid:
                continue
            
            new_chosen = chosen | {i}
            result = _search_k(target, k - 1, i + 1, current_sum + val, new_chosen)
            if result is not None:
                return result
        
        return None
    
    # Solve for each target 1..128
    for n in range(1, 129):
        if time.time() > deadline:
            break
        result = find_min_antichain(n)
        if result is not None:
            answers[n] = [smooth[i] for i in result]
        else:
            # Fallback: if n is smooth, use [n]
            if n in smooth:
                answers[n] = [n]
            else:
                # Last resort: try to find any valid representation (non-minimal)
                answers[n] = _find_any(n)
    
    # Fill any remaining None entries
    for n in range(1, 129):
        if answers[n] is None:
            if n in smooth:
                answers[n] = [n]
            else:
                answers[n] = _find_any(n)
    
    # Return rows for targets 1..128
    return [answers[n] for n in range(1, 129)]

def _find_any(target):
    """Find any valid antichain representation (non-minimal)."""
    # Simple backtracking without minimization
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    smooth.sort()
    
    def backtrack(remaining, start_idx, chosen):
        if remaining == 0:
            return list(chosen)
        if start_idx >= len(smooth):
            return None
        
        # Try including smooth[start_idx]
        val = smooth[start_idx]
        if val <= remaining:
            # Check conflict
            valid = True
            for c in chosen:
                if smooth[c] % val == 0 or val % smooth[c] == 0:
                    valid = False
                    break
            if valid:
                result = backtrack(remaining - val, start_idx + 1, chosen | {start_idx})
                if result is not None:
                    return result
        
        # Try excluding smooth[start_idx]
        return backtrack(remaining, start_idx + 1, chosen)
    
    result = backtrack(target, 0, set())
    if result is not None:
        return [smooth[i] for i in result]
    return []

# EVOLVE-BLOCK-END
