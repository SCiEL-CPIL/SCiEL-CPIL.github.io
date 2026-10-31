"""Rewritten program: Optimal Sidon set search for max score."""
import math

def construct():
    """
    Construct a finite set U of nonnegative integers containing 0 to maximize
    the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).
    
    Strategy: Find Sidon sets (B2 sets) with minimal diameter for various sizes n.
    Sidon sets maximize the ratio |U-U|/|U+U| for a given size.
    The score is maximized by minimizing the diameter (max element) for a given n,
    as the ratio approaches 2 for large n, and the denominator log(2*max+1) decreases
    with smaller max.
    """
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        U_set = set(U)
        max_u = max(U_set)
        if max_u == 0:
            return 0.0
        n = len(U_set)
        # For Sidon sets:
        # |U+U| = n(n+1)/2
        # |U-U| = n(n-1) + 1
        # We verify this in the search, but for scoring we can use these if we know it's Sidon.
        # However, to be safe, we compute actual sizes if not guaranteed Sidon.
        # But our search guarantees Sidon.
        
        # Let's compute actual sizes to be robust
        sum_set = set()
        diff_set = set()
        for a in U_set:
            for b in U_set:
                sum_set.add(a + b)
                diff_set.add(a - b)
                
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0:
            return 0.0
        
        log_denom = math.log(2 * max_u + 1)
        if log_denom <= 0:
            return 0.0
        
        return 1.0 + math.log(n_diff / n_sum) / log_denom

    def find_min_diameter_sidon(n):
        """
        Find a Sidon set of size n with minimal maximum element.
        Returns the set or None if not found within reasonable limits.
        """
        if n == 1:
            return [0]
        
        # Lower bound for M is roughly n^2/2
        # Upper bound for search
        max_limit = n * n + 10 
        
        for M in range(n, max_limit + 1):
            # Backtracking search
            current_set = [0]
            sums = {0}
            
            def backtrack(idx, current_max):
                if idx == n:
                    return True
                
                # Try adding next element
                # Start from 1, go up to M
                # Optimization: We can start from 1, but we must ensure distinct sums.
                for x in range(1, M + 1):
                    # Check if x can be added
                    valid = True
                    new_sums = set()
                    for u in current_set:
                        s = x + u
                        if s in sums:
                            valid = False
                            break
                        new_sums.add(s)
                    
                    if valid:
                        current_set.append(x)
                        sums.update(new_sums)
                        
                        if backtrack(idx + 1, x):
                            return True
                        
                        # Undo
                        current_set.pop()
                        for s in new_sums:
                            sums.remove(s)
                            
                return False
            
            if backtrack(1, 0):
                return current_set
        return None

    best_score = 0.0
    best_U = [0]
    
    # Try sizes n from 2 to 20
    for n in range(2, 21):
        U = find_min_diameter_sidon(n)
        if U is None:
            continue
        
        score = compute_score(U)
        if score > best_score:
            best_score = score
            best_U = U
            
    return best_U
