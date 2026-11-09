import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 45.0  # Safety margin for 90s limit (imports + overhead)
    
    n = 256
    max_size = 64
    
    # We will maintain A and B as sorted lists for ease of iteration, 
    # and their difference and sum sets for fast checking.
    
    def get_diffs(lst):
        if len(lst) < 2:
            return set()
        return {b - a for a, b in combinations(lst, 2)}
    
    def get_sums(lst):
        if not lst:
            return set()
        return {a + b for a, b in combinations_with_replacement(lst, 2)}

    def can_add(lst, diffs, sums, x, other_diffs):
        # Check Sidon property for lst + [x]
        # New sums: x + y for y in lst, and x + x
        # New diffs: x - y for y in lst, and y - x for y in lst
        
        # Check sum uniqueness
        # The new sums are x+x and x+y for y in lst.
        # These must not be in existing sums, and must be unique among themselves.
        # Note: x+x is unique if 2x not in sums.
        # x+y must not be in sums.
        # x+y1 != x+y2 implies y1!=y2, which is true since lst is a set.
        # So we just need:
        # 1. 2x not in sums
        # 2. {x+y for y in lst} disjoint from sums
        # 3. {x+y for y in lst} disjoint from {2x} (implied if x not in lst, but x could be in lst? No, we add new x)
        #    Wait, if x is already in lst, we can't add it again. We assume x not in lst.
        
        if x in lst:
            return False

        new_sums = {x + x} | {x + y for y in lst}
        if new_sums & sums:
            return False
        
        # Check difference disjointness
        # New diffs: {x - y for y in lst} U {y - x for y in lst}
        new_diffs = {x - y for y in lst} | {y - x for y in lst}
        
        # 1. New diffs must be disjoint from existing diffs of this set (implied by Sidon? No, Sidon is sums. Diffs can repeat in a Sidon set? 
        #    Actually, if a-b = c-d, then a+d = c+b. If a,b,c,d distinct, this violates Sidon. 
        #    If a=c, b=d, trivial. If a=d, b=c, then a-b = b-a => 2(a-b)=0 => a=b, impossible for distinct.
        #    So in a Sidon set, all nonzero differences are unique? 
        #    Let's check: a-b = c-d => a+d = b+c. 
        #    If {a,d} != {b,c}, then we have two pairs with same sum. 
        #    If {a,d} = {b,c}, then a=b, d=c or a=c, d=b.
        #    If a=b, then a-b=0, not nonzero.
        #    If a=c, then a-b = b-a => 2a=2b => a=b.
        #    So yes, in a Sidon set, all nonzero differences are distinct.
        #    Therefore, new_diffs must be disjoint from existing diffs of the SAME set.
        
        if new_diffs & diffs:
            return False
            
        # 2. New diffs must be disjoint from other_diffs
        if new_diffs & other_diffs:
            return False
            
        return True

    def add_element(lst, diffs, sums, x, other_diffs):
        # Returns updated lst, diffs, sums
        new_lst = sorted(lst + [x])
        # Update diffs
        # New diffs are x-y and y-x for y in lst
        new_diffs = diffs.union({x - y for y in lst}).union({y - x for y in lst})
        # Update sums
        # New sums are x+x and x+y for y in lst
        new_sums = sums.union({x + x}).union({x + y for y in lst})
        return new_lst, new_diffs, new_sums

    def remove_element(lst, diffs, sums, x):
        # Returns updated lst, diffs, sums
        # This is expensive to recompute from scratch, but lst is small (<64).
        # Recomputing is O(k^2) which is ~4000 ops. Acceptable.
        new_lst = [y for y in lst if y != x]
        new_diffs = get_diffs(new_lst)
        new_sums = get_sums(new_lst)
        return new_lst, new_diffs, new_sums

    def greedy_fill(A, dA, sA, B, dB, sB):
        # Try to add any element from 1 to n
        for x in range(1, n + 1):
            if len(A) >= max_size and len(B) >= max_size:
                break
            if len(A) < max_size and can_add(A, dA, sA, x, dB):
                A, dA, sA = add_element(A, dA, sA, x, dB)
            if len(B) < max_size and can_add(B, dB, sB, x, dA):
                B, dB, sB = add_element(B, dB, sB, x, dA)
        return A, dA, sA, B, dB, sB

    # Initialize
    random.seed(42)
    best_A, best_B = [], []
    best_size = 0
    
    # Start with empty sets
    A, dA, sA = [], set(), set()
    B, dB, sB = [], set(), set()
    
    # Initial greedy pass
    A, dA, sA, B, dB, sB = greedy_fill(A, dA, sA, B, dB, sB)
    best_A, best_B = A[:], B[:]
    best_size = len(A) + len(B)
    
    # Iterated Local Search
    while time.time() - start_time < time_limit:
        # Perturbation: remove k random elements
        k = random.randint(2, 5)
        # Remove from A
        if A and random.random() < 0.5:
            for _ in range(min(k, len(A))):
                if A:
                    idx = random.randint(0, len(A) - 1)
                    x = A[idx]
                    A, dA, sA = remove_element(A, dA, sA, x)
        else:
            # Remove from B
            if B:
                for _ in range(min(k, len(B))):
                    if B:
                        idx = random.randint(0, len(B) - 1)
                        x = B[idx]
                        B, dB, sB = remove_element(B, dB, sB, x)
        
        # Re-optimize
        A, dA, sA, B, dB, sB = greedy_fill(A, dA, sA, B, dB, sB)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A, best_B = A[:], B[:]
            
            # Optional: try to swap elements to improve further?
            # For now, just keep the best found.
            
        # Check time
        if time.time() - start_time > time_limit - 5:
            break

    # Final check and return
    if not best_A: best_A = [1]
    if not best_B: best_B = [2]
    
    return [best_A, best_B]
