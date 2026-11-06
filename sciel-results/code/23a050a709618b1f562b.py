"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

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
    
    # Precompute conflict bitmask: conflict_mask[i] is a bitmask of indices j where smooth[i] and smooth[j] conflict
    conflict_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and (smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0):
                conflict_mask[i] |= (1 << j)
    
    # Precompute suffix minimum sums: min_sum[i][k] = minimum sum of k elements from smooth[i:]
    # For pruning
    max_k = 10
    min_suffix = [[0] * (max_k + 1) for _ in range(n_smooth + 2)]
    max_suffix = [[0] * (max_k + 1) for _ in range(n_smooth + 2)]
    
    for i in range(n_smooth, -1, -1):
        for k in range(1, max_k + 1):
            if i + k > n_smooth:
                min_suffix[i][k] = float('inf')
                max_suffix[i][k] = 0
            else:
                # Minimum: take k smallest from i onwards
                # Maximum: take k largest from i onwards
                if k <= n_smooth - i:
                    # For min, we need the sum of k smallest elements starting from i
                    # This is complex; let's use a simpler bound
                    min_suffix[i][k] = sum(smooth[i:i+k])  # approximation
                    max_suffix[i][k] = sum(smooth[n_smooth-k:n_smooth])  # approximation
    
    # Better approach: precompute for each starting index and k, the min and max possible sum
    # of k elements that form an antichain. But that's expensive.
    # Instead, use simple bounds.
    
    def is_antichain(indices):
        """Check if the given indices form an antichain."""
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if smooth[i] % smooth[j] == 0 or smooth[j] % smooth[i] == 0:
                    return False
        return True
    
    # Precompute all valid antichains of size 1, 2, 3, 4, 5
    # Store as: antichains_by_size[k] = list of (sum, tuple_of_indices)
    antichains_by_size = {k: [] for k in range(1, 6)}
    
    for k in range(1, 6):
        for combo in combinations(range(n_smooth), k):
            if is_antichain(combo):
                s = sum(smooth[i] for i in combo)
                if s <= 128:
                    antichains_by_size[k].append((s, combo))
    
    # Sort each list by sum for quick lookup
    for k in range(1, 6):
        antichains_by_size[k].sort()
    
    # Build a lookup: for each target and size, find the antichain
    # Use binary search or direct dict
    lookup = {}  # (target, k) -> indices tuple
    for k in range(1, 6):
        for s, combo in antichains_by_size[k]:
            if (s, k) not in lookup:
                lookup[(s, k)] = combo
    
    # For targets not covered by precomputed antichains (up to size 5), use search
    def search_antichain(target, max_k=10):
        """Find minimal antichain summing to target using DFS."""
        for k in range(1, max_k + 1):
            if time.time() > deadline:
                return None
            # Try to find antichain of size k
            result = _dfs(target, k, 0, 0, ())
            if result is not None:
                return result
        return None
    
    def _dfs(remaining, k, start_idx, chosen_mask, chosen_indices):
        """DFS to find antichain of size k summing to remaining."""
        if k == 0:
            return chosen_indices if remaining == 0 else None
        if start_idx >= n_smooth:
            return None
        
        # Pruning: if remaining < 0 or not enough elements left
        if remaining < 0 or n_smooth - start_idx < k:
            return None
        
        # Try each candidate
        for i in range(start_idx, n_smooth):
            val = smooth[i]
            if val > remaining:
                break
            
            # Check conflict with chosen
            if chosen_mask & conflict_mask[i]:
                continue
            
            new_mask = chosen_mask | (1 << i)
            result = _dfs(remaining - val, k - 1, i + 1, new_mask, chosen_indices + (i,))
            if result is not None:
                return result
        
        return None
    
    # Solve for each target 1..128
    answers = [None] * 129
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        # First try precomputed antichains (sizes 1-5)
        found = False
        for k in range(1, 6):
            key = (n, k)
            if key in lookup:
                combo = lookup[key]
                answers[n] = [smooth[i] for i in combo]
                found = True
                break
        
        if not found:
            # Use DFS search
            result = search_antichain(n, max_k=10)
            if result is not None:
                answers[n] = [smooth[i] for i in result]
            else:
                # Fallback: try to find any representation
                answers[n] = _find_any(n, smooth, n_smooth)
    
    # Fill any remaining None entries
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _find_any(n, smooth, n_smooth)
    
    # Return rows for targets 1..128
    return [answers[n] for n in range(1, 129)]

def _find_any(target, smooth, n_smooth):
    """Find any valid antichain representation (non-minimal) using greedy + repair."""
    # Try all subsets of size 1 to 8 using DFS
    def dfs(remaining, start_idx, chosen_mask, chosen_indices):
        if remaining == 0:
            return chosen_indices
        if start_idx >= n_smooth or remaining < 0:
            return None
        
        # Try including smooth[start_idx]
        val = smooth[start_idx]
        if val <= remaining:
            # Check conflict
            conflict = 0
            for i in chosen_indices:
                if smooth[i] % val == 0 or val % smooth[i] == 0:
                    return None  # conflict found, skip this branch entirely
            # Actually need to check properly
            has_conflict = False
            for i in chosen_indices:
                if smooth[i] % val == 0 or val % smooth[i] == 0:
                    has_conflict = True
                    break
            if not has_conflict:
                result = dfs(remaining - val, start_idx + 1, chosen_mask | (1 << start_idx), chosen_indices + (start_idx,))
                if result is not None:
                    return result
        
        # Try excluding smooth[start_idx]
        return dfs(remaining, start_idx + 1, chosen_mask, chosen_indices)
    
    result = dfs(target, 0, 0, ())
    if result is not None:
        return [smooth[i] for i in result]
    
    # Last resort: return empty (should not happen)
    return []

# EVOLVE-BLOCK-END
