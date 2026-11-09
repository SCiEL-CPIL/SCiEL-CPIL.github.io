"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 85.0

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_A = [1]
    best_B = [2]
    best_score = 2
    
    def check_sidon(elements):
        """Check if elements form a Sidon set (all pair sums a+b with a<=b are unique)."""
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = elements[i] + elements[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(elements):
        """Get set of absolute differences between distinct elements."""
        diffs = set()
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                d = abs(elements[j] - elements[i])
                diffs.add(d)
        return diffs
    
    def try_add(elements, x, sums_set, diffs_set, other_diffs):
        """Try to add x to elements. Returns True if successful, updates in-place."""
        if x in elements:
            return False
        # Check Sidon: new pair sums x+a for a in elements, and 2*x
        for a in elements:
            if (x + a) in sums_set:
                return False
        if (2 * x) in sums_set:
            return False
        # Check diff conflict
        for a in elements:
            d = abs(x - a)
            if d > 0 and d in other_diffs:
                return False
        # Add
        elements.append(x)
        elements.sort()
        for a in elements[:-1]:  # all except the newly added last element
            sums_set.add(x + a)
        sums_set.add(2 * x)
        for a in elements[:-1]:
            d = abs(x - a)
            if d > 0:
                diffs_set.add(d)
        return True
    
    for attempt in range(200):
        if time.time() - t0 > DEADLINE - 3:
            break
        
        A = []
        B = []
        sumA = set()
        sumB = set()
        diffA = set()
        diffB = set()
        
        # Greedy with random ordering
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - t0 > DEADLINE - 3:
                break
            
            added = False
            # Try A first
            if len(A) < MAX_EACH:
                if try_add(A, x, sumA, diffA, diffB):
                    added = True
            # Try B
            if not added and len(B) < MAX_EACH:
                if try_add(B, x, sumB, diffB, diffA):
                    added = True
            
            # If neither worked, try to add to whichever has more room
            if not added:
                if len(A) < MAX_EACH:
                    if try_add(A, x, sumA, diffA, diffB):
                        pass
                elif len(B) < MAX_EACH:
                    if try_add(B, x, sumB, diffB, diffA):
                        pass
        
        # Local improvement: try to fill remaining slots
        if time.time() - t0 > DEADLINE - 3:
            break
        
        # Try to add more elements to A
        for x in range(1, N + 1):
            if time.time() - t0 > DEADLINE - 3:
                break
            if len(A) >= MAX_EACH:
                break
            if x not in A:
                if try_add(A, x, sumA, diffA, diffB):
                    pass
        
        # Try to add more elements to B
        for x in range(1, N + 1):
            if time.time() - t0 > DEADLINE - 3:
                break
            if len(B) >= MAX_EACH:
                break
            if x not in B:
                if try_add(B, x, sumB, diffB, diffA):
                    pass
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)
            
            if best_score >= 2 * MAX_EACH:
                break
    
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
