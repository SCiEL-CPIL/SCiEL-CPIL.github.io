import random

N = 256

def construct():
    # Precompute all possible x in [1, N]
    candidates = list(range(1, N + 1))
    
    def is_sidon(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_covered(A):
        # Returns set of y in [1, N] such that y = b+c-a for a,b,c in A
        covered = set()
        for a in A:
            for b in A:
                for c in A:
                    y = b + c - a
                    if 1 <= y <= N:
                        covered.add(y)
        # Also A itself is "covered" in the sense that we don't need to add them
        # But the condition is [1, N] subset A U S_diff
        # So we return A U S_diff
        return covered.union(A)

    A = []
    covered = set()
    
    while True:
        # Check if all [1, N] are covered
        if covered == set(range(1, N + 1)):
            break
            
        best_x = None
        best_gain = -1
        best_new_covered = None
        
        for x in candidates:
            if x in A:
                continue
                
            # Check if A U {x} is Sidon
            new_A = A + [x]
            if not is_sidon(new_A):
                continue
                
            # Calculate new covered set
            new_covered = get_covered(new_A)
            
            # Calculate gain
            gain = len(new_covered - covered)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_new_covered = new_covered
        
        if best_x is None:
            # No valid move found? This shouldn't happen if we start empty
            # But if we are stuck, we might need to backtrack or accept non-maximal
            # For now, break and return A (which might not be maximal)
            # But we want a valid maximal set.
            # If no x can be added to maintain Sidon, then A is maximal?
            # No, A is maximal if NO x can be added to maintain Sidon.
            # Wait. The condition for maximality is:
            # For all x not in A, A U {x} is NOT Sidon.
            # So if we iterate all x and NONE are valid (Sidon), then A IS maximal.
            # But we also need [1, N] to be covered?
            # No, the definition of maximal Sidon set is purely:
            # 1. A is Sidon.
            # 2. For all x in [1, N] \ A, A U {x} is NOT Sidon.
            
            # My "covered" logic was an equivalent condition?
            # Let's re-verify.
            # A U {x} is NOT Sidon iff there is a collision.
            # Collision involves x: x+a = b+c.
            # This implies x = b+c-a.
            # So if x is NOT in {b+c-a}, then no collision involving x exists?
            # Yes, because A is Sidon, so no collisions within A.
            # So A U {x} is Sidon iff x is NOT in S_diff(A).
            # So A is maximal iff for all x not in A, x IS in S_diff(A).
            # So [1, N] \ A subset S_diff(A).
            # So [1, N] subset A U S_diff(A).
            
            # So if best_x is None, it means for all x not in A, A U {x} is NOT Sidon.
            # Which means A is maximal.
            # So we can break.
            break
            
        A.append(best_x)
        covered = best_new_covered

    return A
