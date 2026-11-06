import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Check if a set S is valid
    def is_valid(S):
        S_set = set(S)
        for a in S_set:
            target = Fraction(1, a)
            others = [b for b in S_set if b != a]
            # Generate all subset sums of reciprocals of others
            # Use a set of Fractions to track reachable sums
            # To optimize, we can prune sums > target
            sums = {Fraction(0)}
            for b in others:
                rb = Fraction(1, b)
                new_sums = set()
                for s in sums:
                    ns = s + rb
                    if ns <= target:
                        new_sums.add(ns)
                sums.update(new_sums)
                if target in sums:
                    return False
        return True

    # Try to find the largest K such that the top K numbers form a valid set
    # Start from K=24 down to 1
    for K in range(24, 0, -1):
        # Candidate: largest K numbers
        S = list(range(25 - K, 25))
        
        if is_valid(S):
            return S
        
        # If the top K is invalid, we might still find a valid set of size K
        # by swapping elements. However, the top K is the "safest" candidate.
        # If it fails, smaller K is more likely to succeed with top K.
        # But let's do a quick local search for this K if time permits.
        if time.time() > deadline - 10.0:
            break
            
        # Simple local search: try to swap the smallest element in S with a smaller one
        # Actually, if S is invalid, removing the target of the conflict is a good move.
        # But we want to keep size K.
        # Let's just stick to the top K for now, as it's the most robust.
        pass

    # Fallback: return the seed
    return list(range(13, 25))
