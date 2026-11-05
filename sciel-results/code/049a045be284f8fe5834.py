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
                # Tie-breaking: pick the set with fewer elements, or random
                if random.random() < 0.5:
                    update_sums_diffs(x, A, sums_A, diffs_A)
                else:
                    update_sums_diffs(x, B, sums_B, diffs_B)
            elif can_add_A:
                update_sums_diffs(x, A, sums_A, diffs_A)
            elif can_add_B:
                update_sums_diffs(x, B, sums_B, diffs_B)
            # else: skip x
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B
            
    return [best_A, best_B]
