"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _valid_subset_sums(seq):
    """Check that all subset sums are distinct. Returns True if valid."""
    n = len(seq)
    sums = set()
    # Use iterative subset sum generation
    sums.add(0)
    for x in seq:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        if len(new_sums) != len(sums):
            return False
        sums |= new_sums
    return True

def _greedy_construct(n, max_val, deadline):
    """
    Greedy construction: pick smallest positive integer that keeps all subset sums distinct.
    This is the standard 'complete sequence' / Mian-Chowla-like approach for distinct subset sums.
    """
    start = time.time()
    seq = []
    sums = {0}
    
    for i in range(n):
        if time.time() > deadline:
            break
        found = False
        # Try candidates starting from 1
        # Optimization: the next element must be at least 1, but we can skip many
        # For distinct subset sums, if current max subset sum is M, next element must be > M? No.
        # Actually, for the greedy 'smallest unused' approach:
        # We need to ensure no two subsets have the same sum.
        # A sufficient condition for a sequence to have distinct subset sums is that each element is greater than the sum of all previous elements.
        # But that gives powers of 2. We want smaller.
        # The actual constraint is: for any candidate x, x + s1 != s2 for any existing subset sums s1, s2.
        # Equivalently, x != s2 - s1 for any pair of existing subset sums.
        # So x must not be in the set of all differences of existing subset sums.
        
        # Let's maintain the set of forbidden values (differences between existing subset sums)
        # Actually, it's easier to just try candidates and check.
        # But checking each candidate by generating all new subset sums is O(2^i) per candidate.
        # For i=12, that's manageable.
        
        candidate = 1
        while candidate <= max_val:
            if time.time() > deadline:
                return None
            # Check if candidate is valid
            # New subset sums = old_sums union (old_sums + candidate)
            # They must be disjoint (except for the overlap which is impossible since candidate > 0)
            # Actually, we need: (s + candidate) not in sums for any s in sums.
            # And candidate itself not in sums (which is covered if 0 is in sums, then candidate+0 = candidate).
            
            valid = True
            for s in sums:
                if s + candidate in sums:
                    valid = False
                    break
            
            if valid:
                seq.append(candidate)
                # Update sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + candidate)
                sums = new_sums
                found = True
                break
            candidate += 1
        
        if not found:
            return None
    
    if len(seq) == n:
        return seq
    return None

def _backtrack_construct(n, max_val, deadline):
    """
    Backtracking search for a sequence of n distinct positive integers <= max_val
    with all distinct subset sums. We want to minimize the largest element.
    
    Strategy: Try to find a sequence with max element = M, starting from a lower bound.
    Lower bound: sum of first k elements must be < 2^k for distinct subset sums? 
    Actually, the minimum possible largest element for n=12 is known to be small.
    Powers of 2 give 2^11 = 2048. We can do much better.
    
    Let's use a branch-and-bound / backtracking approach.
    We build the sequence element by element. At each step, we try the smallest possible next element.
    To minimize the largest element, we should try to keep elements small.
    
    However, pure greedy might not be optimal. Let's try a smarter search.
    """
    start = time.time()
    
    # First, try the greedy approach which is fast
    result = _greedy_construct(n, max_val, deadline)
    if result is not None:
        return result
    
    # If greedy fails (unlikely for n=12), try backtracking
    # For n=12, greedy should always succeed within reasonable time
    
    return None

def construct():
    n = 12
    max_val = 1048576
    deadline = time.time() + 80  # Leave 10s margin
    
    # The greedy construction for distinct subset sums is well-known to work.
    # It produces the sequence where each term is the smallest integer not in the 
    # difference set of previous subset sums.
    
    # Let's implement a more efficient version of the greedy check
    # Instead of checking all pairs, we maintain the set of subset sums.
    
    start = time.time()
    seq = []
    sums = {0}
    
    for i in range(n):
        if time.time() > deadline:
            # Return best effort
            break
        
        candidate = 1
        # Optimization: we can skip candidates that are too small
        # The minimum possible next value is at least 1
        # But we can also use the fact that if max(sums) = M, then candidate must be such that
        # no s + candidate is in sums
        
        while candidate <= max_val:
            if time.time() > deadline:
                break
            
            # Check validity: for all s in sums, s + candidate not in sums
            valid = True
            for s in sums:
                if s + candidate in sums:
                    valid = False
                    break
            
            if valid:
                seq.append(candidate)
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + candidate)
                sums = new_sums
                break
            
            candidate += 1
        else:
            # No candidate found
            break
    
    if len(seq) == n:
        return seq
    
    # Fallback: return powers of 2 (always valid)
    return [1 << i for i in range(n)]
# EVOLVE-BLOCK-END
