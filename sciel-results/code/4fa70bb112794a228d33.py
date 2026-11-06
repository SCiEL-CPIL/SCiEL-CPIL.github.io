"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from fractions import Fraction

def _is_valid(S):
    """Check if set S is reciprocal-subsum-free."""
    S_set = set(S)
    S_list = list(S)
    n = len(S_list)
    for i, x in enumerate(S_list):
        target = Fraction(1, x)
        # Try all subsets of S_list excluding x
        others = [S_list[j] for j in range(n) if j != i]
        m = len(others)
        for size in range(1, m + 1):
            for subset in combinations(others, size):
                s = sum(Fraction(1, z) for z in subset)
                if s == target:
                    return False
                # Early termination: if s > target and all remaining are positive, 
                # we can skip larger subsets? No, because we're adding more positive terms, so s will only increase.
                # But combinations are generated in lexicographic order, not sorted by sum.
                # So we can't easily prune.
    return True

def _is_valid_fast(S):
    """
    Faster validity check using integer arithmetic and early termination.
    For each x in S, check if 1/x can be represented as sum of 1/y for y in S\{x}.
    This is equivalent to: is there a subset T of S\{x} such that sum_{y in T} (1/y) = 1/x?
    Multiply by LCM of all elements in S? That might be huge.
    
    Alternative: Use Fraction but with memoization or smarter search.
    Given the small size (max 24 elements), and we are building up to maybe 20, 
    a direct check with Fraction is okay for small sets, but might be slow for large ones.
    
    However, the key insight from the prompt is that we can likely include almost all large numbers.
    The constraint is mainly about small numbers.
    If we pick a set of large numbers, say {10, 11, ..., 24}, it's very likely valid.
    Can we add smaller numbers?
    1/9 = 1/10 + ...? No, 1/10 < 1/9.
    1/9 = 1/18 + 1/9? No, 1/9 is in the set, but we can't use it twice.
    1/9 = 1/18 + 1/10 + ...?
    
    Let's try a greedy approach:
    Start with all numbers 1..24.
    Remove the smallest number that causes a violation?
    Or start from large numbers and add small ones if valid.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit
    
    # Strategy:
    # 1. Start with a known valid set, e.g., {13, ..., 24} (size 12).
    # 2. Try to add elements from 1 to 12 one by one, checking validity.
    # 3. Use local search to swap elements if stuck.
    
    # Initial set: large numbers
    current_set = set(range(13, 25))
    
    # Candidate elements to add
    candidates = list(range(1, 13))
    
    # Sort candidates by "difficulty" or just try in order
    # Adding smaller numbers is harder because they have larger reciprocals
    # So maybe try larger candidates first? 12, 11, ..., 1
    candidates.sort(reverse=True)
    
    while candidates and time.time() < deadline:
        added = False
        new_candidates = []
        for c in candidates:
            if time.time() >= deadline:
                break
            trial_set = current_set | {c}
            # Quick check: is c already in current_set? No, we removed it.
            # Check validity
            if _is_valid(trial_set):
                current_set = trial_set
                added = True
                # Restart candidate list? Or continue with remaining?
                # If we add c, we might need to re-check other candidates?
                # Actually, adding an element can only introduce new violations involving c.
                # It doesn't invalidate previous subsets that didn't involve c.
                # So we can continue with the remaining candidates in the same order.
                # But wait, if we added 12, does it affect 11?
                # 1/11 might now be formable by including 12?
                # No, 1/12 < 1/11, so 1/12 alone doesn't help form 1/11.
                # But 1/12 + 1/x might help?
                # So we should probably restart the candidate list to be safe, or at least re-check.
                # However, for efficiency, let's just add it and continue.
                # If a violation was introduced, it would be involving the new element c.
                # So other elements' validity is preserved.
                # Thus, we can just remove c from candidates and continue.
                candidates.remove(c)
                break
            else:
                new_candidates.append(c)
        if not added:
            # No more candidates can be added directly
            # Try local search: swap an element in current_set with one in candidates
            # Or just break and return current_set
            # For now, let's try a simple swap heuristic
            # Pick the largest element in current_set and try to replace it with a candidate
            # Actually, let's just return the current set for now
            break
        # If we added an element, continue the loop to try adding more
    
    # Ensure non-empty
    if not current_set:
        current_set = {13}
    
    return list(current_set)

# EVOLVE-BLOCK-END
