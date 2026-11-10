"""Combined approach: greedy initialization + multi-strategy local search for 3-AP-free sets."""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    def can_add(x, S):
        """Check if x can be added to S without forming a 3-term AP."""
        # Case 1: x is the largest term: a, b, x where 2b = a + x
        for a in S:
            if a >= x:
                continue
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        # Case 2: x is the smallest term: x, b, c where 2b = x + c
        for c in S:
            if c <= x:
                continue
            s = x + c
            if s % 2 == 0:
                b = s // 2
                if b in S:
                    return False
        # Case 3: x is the middle term: a, x, c where 2x = a + c
        for a in S:
            if a >= x:
                continue
            c = 2 * x - a
            if 1 <= c <= N and c in S:
                return False
        return True

    def get_sum(S):
        """Compute exact reciprocal sum using Fraction."""
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total

    def greedy_fill(forbidden, S_start):
        """Fill S by adding elements in ascending order if they don't create a 3-AP."""
        S = set(S_start)
        for x in all_nums:
            if x in forbidden or x in S:
                continue
            if can_add(x, S):
                S.add(x)
        return S

    # Step 1: Initial greedy fill (ascending to prioritize small numbers)
    S_best = greedy_fill(set(), set())
    sum_best = get_sum(S_best)
    
    # Step 2: Local search with multiple strategies
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        
        # Strategy A: Try swapping large elements in S with smaller elements not in S
        # This directly increases the reciprocal sum if successful
        for x in sorted(S_best, reverse=True):
            if (time.time() - start_time) > deadline:
                return list(S_best)
            
            # Try replacing x with smaller y
            for y in range(1, x):
                if y in S_best:
                    continue
                
                S_candidate = S_best - {x}
                if can_add(y, S_candidate):
                    S_candidate.add(y)
                    sum_new = get_sum(S_candidate)
                    if sum_new > sum_best:
                        S_best = S_candidate
                        sum_best = sum_new
                        improved = True
                        break  # Restart outer loop with new S_best
            if improved:
                break
        
        if improved:
            continue
        
        # Strategy B: Remove an element and re-greedy-fill
        # This can find solutions where removing one blocker allows adding multiple small numbers
        for x in sorted(S_best, reverse=True):
            if (time.time() - start_time) > deadline:
                return list(S_best)
            
            forbidden = {x}
            S_new = greedy_fill(forbidden, S_best - {x})
            sum_new = get_sum(S_new)
            
            if sum_new > sum_best:
                S_best = S_new
                sum_best = sum_new
                improved = True
                break  # Restart with new S_best
        
        # Strategy C: Try adding a small element not in S (if it fits)
        # This might be possible if previous removals created space
        for y in range(1, N + 1):
            if (time.time() - start_time) > deadline:
                return list(S_best)
            
            if y in S_best:
                continue
            if can_add(y, S_best):
                S_best.add(y)
                sum_new = get_sum(S_best)
                if sum_new > sum_best:
                    sum_best = sum_new
                    improved = True
                    break  # Restart with new S_best
    
    return list(S_best)
