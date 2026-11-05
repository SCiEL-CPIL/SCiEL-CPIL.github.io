import time
import random

def construct():
    deadline = time.time() + 50.0
    best_size = 0
    best_A = []
    best_B = []
    
    def can_add(x, S, sums_S, diffs_S, diffs_other):
        # Check differences
        new_diffs = set()
        for s in S:
            d = abs(x - s)
            if d in diffs_other:
                return False
            new_diffs.add(d)
        
        # Check sums
        # New sums are x+s for s in S, and x+x
        # We need to ensure these don't collide with existing sums in sums_S
        # And don't collide with each other (which they won't as argued above)
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
            
        return True

    def update_sums_diffs(x, S, sums_S, diffs_S):
        for s in S:
            sums_S.add(x + s)
            diffs_S.add(abs(x - s))
        sums_S.add(2 * x)
        S.append(x)

    def try_local_search(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to improve by swapping elements or adding new ones."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any element not yet in either set
            for x in range(1, 257):
                if x in A or x in B:
                    continue
                if can_add(x, A, sums_A, diffs_A, diffs_B):
                    update_sums_diffs(x, A, sums_A, diffs_A)
                    improved = True
                    break
                if can_add(x, B, sums_B, diffs_B, diffs_A):
                    update_sums_diffs(x, B, sums_B, diffs_B)
                    improved = True
                    break
            
            # Try swaps: remove an element from A, try to add to B or vice versa
            if not improved:
                # Try removing from A and adding to B
                for i, a in enumerate(A):
                    if time.time() > deadline - 2:
                        return A, B, sums_A, sums_B, diffs_A, diffs_B
                    # Temporarily remove a from A
                    temp_sums_A = set(sums_A)
                    temp_diffs_A = set(diffs_A)
                    temp_A = A.copy()
                    for s in temp_A:
                        if s != a:
                            temp_sums_A.discard(a + s)
                            temp_diffs_A.discard(abs(a - s))
                    temp_sums_A.discard(2 * a)
                    temp_A.remove(a)
                    
                    # Try to add a to B
                    if can_add(a, B, sums_B, diffs_B, diffs_A - {abs(a - b) for b in B if b != a}):
                        # This is complex; skip for now
                        pass

        return A, B, sums_A, sums_B, diffs_A, diffs_B

    # Run multiple random trials
    while time.time() < deadline:
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Randomize the order of candidates to explore different configurations
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            can_add_A = can_add(x, A, sums_A, diffs_A, diffs_B)
            can_add_B = can_add(x, B, sums_B, diffs_B, diffs_A)
            
            if can_add_A and can_add_B:
                # Balance: prefer the set with fewer elements
                if len(A) < len(B):
                    update_sums_diffs(x, A, sums_A, diffs_A)
                elif len(B) < len(A):
                    update_sums_diffs(x, B, sums_B, diffs_B)
                else:
                    if random.random() < 0.5:
                        update_sums_diffs(x, A, sums_A, diffs_A)
                    else:
                        update_sums_diffs(x, B, sums_B, diffs_B)
            elif can_add_A:
                update_sums_diffs(x, A, sums_A, diffs_A)
            elif can_add_B:
                update_sums_diffs(x, B, sums_B, diffs_B)
            # else: skip x
        
        # Local search: try to add more elements
        A, B, sums_A, sums_B, diffs_A, diffs_B = try_local_search(A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B
            
    return [best_A, best_B]
