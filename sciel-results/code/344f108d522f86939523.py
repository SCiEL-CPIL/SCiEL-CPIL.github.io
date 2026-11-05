"""Rewritten program for divisibility-antichain representations by smooth numbers."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

def _generate_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _is_antichain(terms):
    """Check if no term divides another in the list."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def construct():
    """
    Construct 128 rows for targets 1..128.
    Each row contains 1..30 distinct positive 2^a*3^b*5^c terms summing to target,
    forming a divisibility antichain.
    Minimize total number of summands over all targets.
    """
    max_target = 128
    smooth = _generate_smooth(max_target)
    smooth_set = set(smooth)
    
    # Precompute for each target n, find the minimal antichain representation
    # We'll use a DP approach or search for minimal representations
    
    # Strategy: For each target n, try to find the smallest k such that
    # there exists an antichain of k smooth numbers summing to n.
    
    # To speed this up, we can precompute valid antichains of small sizes
    # and then use them to build representations.
    
    # For small k (1, 2, 3, ...), enumerate possible antichains and check sums.
    
    # First, let's precompute all valid antichains of size 1, 2, 3, 4, 5
    # that sum to values <= 128.
    
    # Actually, let's be smarter. For each target n, we want the minimal k.
    # Let's try k = 1, 2, 3, ... until we find a representation.
    
    # For k=1: n itself must be smooth and the antichain condition is trivially satisfied.
    # For k=2: find a, b smooth, a+b=n, a!=b, and neither divides the other.
    # For k=3: find a, b, c smooth, a+b+c=n, all distinct, antichain.
    # etc.
    
    # To make this efficient, let's precompute for each n the minimal k.
    
    # Let's use a different approach: for each n, try k from 1 upward.
    # For each k, search for an antichain of size k summing to n.
    
    # To avoid exponential blowup, we'll use memoization and pruning.
    
    # Precompute divisibility relations
    # div[i][j] = True if smooth[i] divides smooth[j]
    m = len(smooth)
    # Create index mapping
    smooth_index = {v: i for i, v in enumerate(smooth)}
    
    # For each n, find minimal antichain
    answers = []
    
    for n in range(1, max_target + 1):
        found = None
        # Try k = 1, 2, 3, ...
        for k in range(1, 31):
            if found is not None:
                break
            # Find antichain of size k summing to n
            # Use recursive search with pruning
            found = _find_antichain(n, k, smooth, smooth_set, max_target)
            if found is not None:
                break
        if found is None:
            # Should not happen, but fallback
            found = [n] if n in smooth_set else [1]
        answers.append(found)
    
    return answers

def _find_antichain(n, k, smooth, smooth_set, max_target):
    """Find k distinct smooth numbers summing to n forming an antichain."""
    if k == 1:
        if n in smooth_set:
            return [n]
        return None
    
    # For k >= 2, use recursive search
    # We need to pick k numbers from smooth that sum to n and form an antichain.
    
    # Pruning: the sum of the k smallest possible distinct smooth numbers 
    # that could work must be <= n, and the sum of the k largest must be >= n.
    
    # Let's use a DFS with memoization on (n, k, min_idx, used_set)
    # But used_set makes memoization hard. Instead, let's use a different approach.
    
    # For small k, we can enumerate combinations.
    # For k=2: try all pairs
    # For k=3: try all triples (but this could be large)
    
    # Let's use a smarter approach: generate candidate sets and check.
    
    # For efficiency, let's limit the search space.
    # The numbers must be <= n (since they're positive and sum to n).
    candidates = [x for x in smooth if x <= n]
    
    if len(candidates) < k:
        return None
    
    # For k=2
    if k == 2:
        for i in range(len(candidates)):
            a = candidates[i]
            b = n - a
            if b <= a:  # Ensure a < b to avoid duplicates and distinctness
                continue
            if b in smooth_set:
                # Check distinctness (already ensured by a < b)
                # Check antichain: a doesn't divide b
                if b % a != 0:
                    return [a, b]
        return None
    
    # For k >= 3, use recursive DFS
    # To keep it fast, we'll use a limited search
    result = [None]
    
    def dfs(remaining_sum, remaining_k, min_val, current):
        if remaining_k == 0:
            if remaining_sum == 0:
                result[0] = current[:]
                return True
            return False
        if remaining_k == 1:
            if remaining_sum >= min_val and remaining_sum in smooth_set:
                # Check that remaining_sum doesn't divide or is divided by any in current
                s = remaining_sum
                valid = True
                for c in current:
                    if c > s:
                        c, s = s, c
                    if s % c == 0:
                        valid = False
                        break
                if valid:
                    current.append(s)
                    result[0] = current[:]
                    current.pop()
                    return True
            return False
        
        # Pruning: minimum possible sum with remaining_k numbers >= min_val
        # The smallest remaining_k distinct smooth numbers >= min_val
        min_possible = 0
        count = 0
        for x in candidates:
            if x < min_val:
                continue
            min_possible += x
            count += 1
            if count == remaining_k:
                break
        if count < remaining_k or min_possible > remaining_sum:
            return False
        
        # Maximum possible: largest remaining_k distinct smooth numbers <= remaining_sum
        # Actually, we just need to ensure we can pick remaining_k numbers
        # that sum to remaining_sum. The max sum is the sum of the largest 
        # remaining_k numbers that are <= remaining_sum.
        
        # Try each candidate >= min_val
        for i in range(len(candidates)):
            a = candidates[i]
            if a < min_val:
                continue
            if a > remaining_sum:
                break
            # Check antichain with current
            valid = True
            for c in current:
                if c > a:
                    c, a = a, c
                if a % c == 0:
                    valid = False
                    break
            if not valid:
                continue
            current.append(a)
            if dfs(remaining_sum - a, remaining_k - 1, a + 1, current):
                return True
            current.pop()
        return False
    
    dfs(n, k, 1, [])
    return result[0]

# EVOLVE-BLOCK-END
