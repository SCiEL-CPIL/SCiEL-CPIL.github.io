import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    # Helper to check if x can be added to set S (list), given diffs and sums of S and other
    # Returns True if valid
    def can_add(x, S, diffs_S, sums_S, diffs_other):
        # Check against existing diffs in S
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
        
        # Check if new diffs collide with each other
        # This happens if |x - a1| == |x - a2| for a1 != a2
        # Which means 2x == a1 + a2
        if 2 * x in sums_S:
            # This check is sufficient if sums_S contains all pair sums a_i + a_j (i<=j)
            # If 2x = a+a, then x=a, but x is not in S.
            # If 2x = a+b (a!=b), then |x-a| = |x-b|.
            # So we just need to check if 2x is in sums_S.
            return False
            
        return True

    def add_to_set(x, S, diffs_S, sums_S):
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)
        
    def remove_from_set(x, S, diffs_S, sums_S):
        # This is expensive to do correctly by just removing.
        # Better to rebuild or maintain counts.
        # For simplicity in SA, we might just rebuild or use a different representation.
        # Given the small size, rebuilding is fast.
        pass

    # Since remove is hard to do incrementally without counts, 
    # and SA requires many moves, we will stick to Greedy + simple Local Search 
    # that only adds or swaps if easy.
    
    # Actually, let's use a different approach:
    # Represent sets as frozensets for hashing? No, we need to modify.
    # Let's just do many restarts of Greedy with different heuristics.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts * 12345 + 67890)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Candidate list
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        # Greedy Phase
        for x in candidates:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                # Heuristic: Add to the set with more "room" or smaller size
                # Smaller size is a good proxy for more room
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        # Local Search Phase: Try to add more elements by shuffling and re-adding?
        # Or just try to add any remaining elements.
        # The greedy pass already tried all elements once.
        # We can try a second pass with a different order.
        
        remaining = [x for x in range(1, 257) if x not in A and x not in B]
        random.shuffle(remaining)
        
        for x in remaining:
            valid_A = can_add(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_to_set(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_to_set(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, diffs_A, sums_A)
                    else:
                        add_to_set(x, B, diffs_B, sums_B)
            elif valid_A:
                add_to_set(x, A, diffs_A, sums_A)
            elif valid_B:
                add_to_set(x, B, diffs_B, sums_B)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()
            
            # Early stop if we hit a very high score?
            # Max possible is limited.
            if best_score >= 40:
                # Keep going to see if we can do better, but this is a good baseline.
                pass

    return [best_A, best_B]
