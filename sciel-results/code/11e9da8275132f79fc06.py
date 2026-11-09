"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from fractions import Fraction

def construct():
    """
    Find the largest subset of [1, 24] such that for no selected element x,
    1/x equals the sum of reciprocals of any non-empty subset of the OTHER
    selected elements.
    
    Uses backtracking with forward checking:
    - Precompute all possible reciprocal sums for subsets of [1,24]
    - Use constraint propagation to prune candidates early
    """
    N = 24
    elements = list(range(1, N + 1))
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = {x: Fraction(1, x) for x in elements}
    
    # Precompute all possible subset sums of reciprocals for each element
    # For each element x, we need to know which subsets S of (elements \ {x})
    # have sum(S) == 1/x
    
    # Strategy: Use backtracking to build the largest valid subset.
    # We process elements in decreasing order (larger elements first) since
    # they are less likely to create conflicts (their reciprocals are smaller,
    # so fewer subsets will sum to them).
    
    # Actually, let's think about this differently.
    # A conflict occurs when for some selected x, there exists a non-empty subset T of (selected \ {x})
    # such that sum(1/t for t in T) == 1/x.
    
    # Key insight: If we sort elements, smaller elements have larger reciprocals.
    # The sum of reciprocals of any subset will be at least as large as the largest
    # reciprocal in that subset. So if 1/x is very small (x large), it's harder
    # to match.
    
    # Let's use a greedy/backtracking approach:
    # 1. Start with all elements
    # 2. Check for violations
    # 3. Remove elements that are involved in violations, preferring to remove
    #    elements that are part of many violations
    
    # Actually, let's do a proper search. Given N=24, we can't brute force 2^24.
    # But we can use the structure of the problem.
    
    # Observation: For x >= 13, 1/x <= 1/13. The sum of reciprocals of any
    # non-empty subset of {1,...,24} \ {x} is at least 1/24 (if the subset
    # is just {24}) or larger. So conflicts for large x are less common.
    
    # Let's precompute: for each x, which subsets of the other elements sum to 1/x?
    # This is expensive for all subsets, but we can be smarter.
    
    # Alternative approach: Since N is small (24), let's use a constraint-based
    # backtracking search.
    
    # Sort elements in descending order for the search
    elements_desc = sorted(elements, reverse=True)
    
    best_set = None
    best_size = 0
    
    # We'll use a recursive backtracking with pruning
    # State: current set of selected elements, index of next element to consider
    # Pruning: if adding element x creates a conflict with already-selected elements, skip it
    
    def has_conflict(x, selected):
        """Check if adding x to selected creates a conflict.
        A conflict exists if:
        1. 1/x == sum of reciprocals of some non-empty subset of selected
        2. For some y in selected, 1/y == sum of reciprocals of some non-empty subset of (selected \ {y}) ∪ {x}
        
        For efficiency, we only need to check:
        1. Does any non-empty subset of selected sum to 1/x?
        2. For each y in selected, does any non-empty subset of (selected \ {y}) ∪ {x} sum to 1/y?
           This means: does any non-empty subset of (selected \ {y}) sum to 1/y - 1/x? 
           OR does {x} alone sum to 1/y? (i.e., 1/x == 1/y, impossible for distinct x,y)
           OR does some subset of (selected \ {y}) plus {x} sum to 1/y?
           That is: sum(S) + 1/x == 1/y for some non-empty S ⊆ (selected \ {y})
           => sum(S) == 1/y - 1/x
           Also: 1/x alone == 1/y is impossible.
        """
        # Check 1: any non-empty subset of selected sums to 1/x?
        if any_subset_sums_to(selected, recip[x]):
            return True
        
        # Check 2: for each y in selected, does adding x create a new way to sum to 1/y?
        # The new subsets involving x that sum to 1/y are:
        # - {x} alone: 1/x == 1/y -> impossible for distinct x, y
        # - {x} ∪ S where S is a non-empty subset of (selected \ {y}):
        #   1/x + sum(S) == 1/y => sum(S) == 1/y - 1/x
        for y in selected:
            if y == x:
                continue
            target = recip[y] - recip[x]
            if target > 0:
                # Check if any non-empty subset of (selected \ {y}) sums to target
                others = [z for z in selected if z != y]
                if any_subset_sums_to(others, target):
                    return True
            elif target == 0:
                # 1/x == 1/y, impossible for distinct x, y
                pass
            # If target < 0, no subset of positive reciprocals can sum to a negative number
        
        return False
    
    def any_subset_sums_to(elements_list, target):
        """Check if any non-empty subset of elements_list sums to target (as Fraction)."""
        if not elements_list:
            return False
        # Use meet-in-the-middle or just enumerate for small sets
        # Since |elements_list| can be up to 23, we need to be careful
        # But in practice, during backtracking, the selected set is small
        n = len(elements_list)
        if n <= 1:
            return n == 1 and recip[elements_list[0]] == target
        # For small n, enumerate all subsets
        if n <= 15:
            for size in range(1, n + 1):
                for subset in itertools.combinations(elements_list, size):
                    s = sum(recip[e] for e in subset)
                    if s == target:
                        return True
            return False
        else:
            # For larger sets, use a more efficient approach
            # Meet in the middle
            half = n // 2
            left = elements_list[:half]
            right = elements_list[half:]
            
            left_sums = set()
            for size in range(1, len(left) + 1):
                for subset in itertools.combinations(left, size):
                    left_sums.add(sum(recip[e] for e in subset))
            
            right_sums = set()
            for size in range(0, len(right) + 1):
                if size == 0:
                    right_sums.add(Fraction(0, 1))
                else:
                    for subset in itertools.combinations(right, size):
                        right_sums.add(sum(recip[e] for e in subset))
            
            for ls in left_sums:
                if target - ls in right_sums:
                    # Need to ensure the combined subset is non-empty
                    # Since left_sums only contains non-empty sums, this is fine
                    return True
            return False
    
    # Backtracking search
    selected = []
    
    def backtrack(idx):
        nonlocal best_set, best_size
        
        if idx == len(elements_desc):
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = list(selected)
            return
        
        # Pruning: even if we add all remaining elements, can we beat best_size?
        if len(selected) + (len(elements_desc) - idx) <= best_size:
            return
        
        x = elements_desc[idx]
        
        # Option 1: Skip x
        backtrack(idx + 1)
        
        # Option 2: Add x if it doesn't create a conflict
        if not has_conflict(x, selected):
            selected.append(x)
            backtrack(idx + 1)
            selected.pop()
    
    backtrack(0)
    
    if best_set is None:
        # Fallback: return a known valid set
        return list(range(13, 25))
    
    return best_set

# EVOLVE-BLOCK-END
