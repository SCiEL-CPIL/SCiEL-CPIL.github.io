"""CSP-based construction of a Sidon set containing 0 with all differences 1..30."""
import sys
sys.setrecursionlimit(10000)

def construct():
    # We need a Sidon set S containing 0 such that every d in 1..30 is a difference.
    # Sidon condition: all positive differences y-x (x<y in S) are distinct.
    # We want to minimize max(S).
    
    # Key insight: If S is a Sidon set with |S| = n, then the number of positive differences is n*(n-1)/2.
    # We need at least 30 differences (for d=1..30), so n*(n-1)/2 >= 30 => n >= 9 (since 8*7/2=28 < 30, 9*8/2=36 >= 30).
    # Actually we need exactly the differences 1..30 to appear, but other differences may also appear.
    # The set must have at least 9 elements.
    
    # Strategy: Use backtracking with forward checking.
    # State: current set of elements (sorted), set of used differences.
    # We build the set incrementally.
    
    # Let's try a more systematic approach:
    # 1. Start with {0}.
    # 2. We need to add elements such that all differences 1..30 are covered.
    # 3. Use backtracking: at each step, try adding a new element x > max(S) (or possibly between existing elements, but for simplicity start with x > max).
    
    # Actually, let me think about this differently. A perfect difference set of order k has k elements and all differences 1..k^2-k cover exactly. For k=6, k^2-k=30. So a (6,6,1) difference set would have 6 elements and differences 1..30. But wait, a (v,k,λ) difference set has v elements, each block has k elements, and every nonzero difference appears exactly λ times. For λ=1, we have a Sidon set.
    
    # A (31,6,1) difference set exists (projective plane of order 5). Its elements can be taken from Z_31. But we need integers in 0..10^6 with 0 included, and differences as integers (not mod 31).
    
    # Let me try a different approach: use the known (31,6,1) difference set, map it to integers, and then adjust.
    
    # The (31,6,1) difference set: {0, 1, 3, 8, 12, 18} in Z_31.
    # Differences (mod 31): Let's compute all pairwise differences.
    # 1-0=1, 3-0=3, 8-0=8, 12-0=12, 18-0=18
    # 3-1=2, 8-1=7, 12-1=11, 18-1=17
    # 8-3=5, 12-3=9, 18-3=15
    # 12-8=4, 18-8=10
    # 18-12=6
    # So differences mod 31: 1,2,3,4,5,6,7,8,9,10,11,12,15,17,18
    # Missing: 13,14,16,19,20,21,22,23,24,25,26,27,28,29,30
    # That's not all 1..30. So this isn't a perfect difference set for our purposes.
    
    # Let me try a greedy/backtracking approach with pruning.
    
    def solve():
        # We'll use backtracking to find a Sidon set containing 0 with all diffs 1..30.
        # State: current elements (sorted list), used differences (set)
        # We want to minimize the maximum element.
        
        # Try different sizes: start with minimum needed.
        # n=9: 36 diffs total, need to cover 1..30 (30 of them), 6 extra.
        # n=10: 45 diffs total.
        
        # Let's use a priority-based backtracking.
        # At each step, pick the smallest missing difference and try to realize it.
        
        import itertools
        
        def all_diffs(s):
            """Return list of all positive differences in sorted set s."""
            return [y - x for i, x in enumerate(s) for y in s[i+1:]]
        
        def is_sidon(s):
            """Check if s is a Sidon set."""
            d = all_diffs(s)
            return len(d) == len(set(d))
        
        # Backtracking search
        # We'll try to build the set by adding one element at a time.
        # Key: we can add elements in any order, but for efficiency, we add elements in increasing order.
        
        # Actually, let's use a smarter approach:
        # We know 0 is in the set. Let's say the set is {0, a1, a2, ..., an-1} with 0 < a1 < ... < an-1.
        # We need all differences 1..30 to appear.
        
        # Use iterative deepening on the maximum element.
        
        best = None
        
        # Try max element from small to large
        # Lower bound: with n elements, max element must be at least n-1 (if elements are 0,1,...,n-1, but that's not Sidon for n>3).
        # For n=9, a rough lower bound: the 36 differences must be distinct and at most max(S). So max(S) >= 36.
        # But we also need specific differences 1..30.
        
        # Let's try a direct backtracking with good heuristics.
        
        def backtrack(elements, used_diffs, max_val):
            nonlocal best
            # Check if we've covered all required differences
            missing = [d for d in range(1, 31) if d not in used_diffs]
            if not missing:
                # Found a valid set
                if best is None or max(elements) < max(best):
                    best = list(elements)
                return True
            
            # Pruning: if max(elements) >= max_val, stop
            if elements and max(elements) >= max_val:
                return False
            
            # Heuristic: pick the smallest missing difference
            d = missing[0]
            
            # Try to add a new element x such that x - y = d or y - x = d for some y in elements
            # i.e., x = y + d or x = y - d for some y in elements
            # x must be >= 0, not in elements, and adding x must preserve Sidon property
            
            candidates = set()
            for y in elements:
                for cand in (y + d, y - d):
                    if cand >= 0 and cand not in elements:
                        candidates.add(cand)
            
            # Sort candidates by value (try smaller first to keep max small)
            for x in sorted(candidates):
                new_elements = sorted(elements + [x])
                new_diffs = all_diffs(new_elements)
                
                # Check Sidon property
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                
                # Check that no new difference conflicts with used ones (should be fine since we check Sidon)
                new_used = used_diffs | set(new_diffs)
                
                # Pruning: if any new difference > max_val - 1, skip (since max element would be too large)
                # Actually, the max element is max(new_elements), and all diffs are <= max(new_elements)
                if max(new_elements) >= max_val:
                    continue
                
                if backtrack(new_elements, new_used, max_val):
                    return True
            
            return False
        
        # Iterative deepening on max value
        # Lower bound: need at least 30 distinct differences, all <= max(S), so max(S) >= 30.
        # But with Sidon constraint, it's larger.
        # For n elements, max(S) >= n*(n-1)/2 is a lower bound (all diffs distinct and positive, max diff = max(S)).
        # For n=9: 36. For n=10: 45.
        
        # Let's try increasing max_val
        for max_val in range(35, 200):
            # Try with different initial sizes
            # Start with just {0}
            result = backtrack([0], set(), max_val)
            if best is not None:
                break
        
        if best is not None:
            return best
        
        # Fallback: greedy approach (shouldn't reach here)
        a = [0]
        for d in range(1, 31):
            found = False
            for x in range(1, 1000000):
                trial = sorted(a + [x])
                dd = all_diffs(trial)
                if d in dd and len(set(dd)) == len(dd):
                    a = trial
                    found = True
                    break
            if not found:
                # Force add
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    dd = all_diffs(trial)
                    if len(set(dd)) == len(dd):
                        a = trial
                        break
                    y += 1
        return a
    
    return solve()
