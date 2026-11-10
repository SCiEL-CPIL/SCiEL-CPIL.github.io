import time
import numpy as np
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin
    
    n = 256
    max_size_each = 64
    
    # We will try to construct A and B simultaneously.
    # State: A, B, diff_A, diff_B, sums_A, sums_B
    # To be efficient, we precompute potential conflicts? No, dynamic is better for flexibility.
    
    def is_valid_state(A, B, diff_A, diff_B, sums_A, sums_B, x, target_set):
        """
        Check if adding x to target_set (A or B) maintains validity.
        target_set: 'A' or 'B'
        """
        if target_set == 'A':
            current_set = A
            current_diffs = diff_A
            current_sums = sums_A
            other_diffs = diff_B
        else:
            current_set = B
            current_diffs = diff_B
            current_sums = sums_B
            other_diffs = diff_A
            
        # 1. Check Sidon condition for current_set + x
        # New sums involving x: x + y for y in current_set, and x + x
        new_sums = [x + y for y in current_set] + [2*x]
        if len(new_sums) != len(set(new_sums)):
            return False
        if len(set(new_sums) & current_sums) > 0:
            return False
            
        # 2. Check Difference conditions
        # New diffs involving x: x - y and y - x for y in current_set
        new_diffs = set()
        for y in current_set:
            new_diffs.add(x - y)
            new_diffs.add(y - x)
        
        # Internal uniqueness of new diffs (implied by Sidon if sums unique? 
        # Actually Sidon on S means a+b=c+d => {a,b}={c,d}.
        # Diffs d1=a-b, d2=c-d. a-b=c-d => a+d=b+c. 
        # If {a,d} != {b,c}, sum collision. 
        # So if Sidon holds, diffs are unique. 
        # However, we must check new_diffs against current_diffs for uniqueness within the set.
        # Wait, if we just added x, and Sidon holds for the whole set, then all diffs are unique.
        # But we need to ensure new_diffs don't collide with current_diffs.
        if len(new_diffs & current_diffs) > 0:
            return False
            
        # 3. Check Disjointness from other set's diffs
        if len(new_diffs & other_diffs) > 0:
            return False
            
        return True

    def get_conflict_score(x, A, B, diff_A, diff_B, target_set):
        """
        Heuristic: How many 'future' options does adding x block?
        We can approximate by counting how many elements in the other set's potential range
        would be blocked. 
        Simpler: Just return a random value or 0 to allow pure randomization with pruning.
        Or: Count overlaps with the *other* set's existing elements' potential diffs?
        Let's stick to simple randomization with strong pruning for now, 
        but prioritize candidates that are 'far' from existing elements to spread out.
        """
        if target_set == 'A':
            other = B
        else:
            other = A
            
        # Score based on distance from other set's elements?
        # If we pick x close to B, its diffs are small. 
        # If we pick x far from B, its diffs are large.
        # Large diffs are "safer" if the other set is compact?
        # Let's just use random order but check validity.
        return 0

    best_A = []
    best_B = []
    
    rng = random.Random(42)
    
    # Strategy: Multiple restarts. 
    # In each restart, we build A and B alternately or one by one.
    # Let's try building A first, then B? No, simultaneous is better for disjoint diffs.
    # But Sidon is local to each set.
    # Let's try: 
    # 1. Pick a candidate for A or B (randomly choose which set to grow).
    # 2. Pick a value from [1, 256] not in either set (sets can overlap, but distinct values help Sidon? 
    #    Actually, sets can overlap. If x in A and x in B, that's fine. 
    #    But if x in A, its diffs are in diff_A. If x in B, its diffs are in diff_B.
    #    If A and B share an element, it doesn't directly cause diff collision unless the diffs collide.
    #    However, to maximize size, we should probably use distinct values if possible, 
    #    or at least ensure we have enough room.
    #    The problem says "A and B need not themselves be disjoint".
    #    But if they overlap, we are using the same value. 
    #    Let's allow overlap but track used values? No, values can be reused.
    #    However, reusing values doesn't help size. |A|+|B| is the goal.
    #    If A={1}, B={1}, size 2. Diffs: A: empty (size 1 set has 0 diffs). B: empty.
    #    If A={1, 2}, B={1, 3}.
    #    Let's just pick values from [1, 256].
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        A = []
        B = []
        diff_A = set()
        diff_B = set()
        sums_A = set()
        sums_B = set()
        
        # We will add elements one by one.
        # Total elements target: maybe 20-30?
        # We'll just keep adding until we can't or time runs out.
        
        # To avoid getting stuck in small sets, we can bias towards larger numbers initially?
        # Or just random.
        
        # Pre-shuffle candidates?
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        # We'll iterate through candidates and decide if they go to A or B or neither.
        # Actually, a better approach: 
        # Maintain a list of available candidates.
        # At each step, pick a random candidate.
        # Try to add to A. If valid, add.
        # Else, try to add to B. If valid, add.
        # Else, discard (or keep for later?).
        # This is a greedy packing.
        
        # To improve, we can look ahead. 
        # But simple greedy with restarts is robust.
        
        # Let's try a "balanced" growth:
        # Alternate between trying to add to A and B.
        
        success = True
        for x in candidates:
            if time.time() > deadline:
                break
                
            # Try A
            if is_valid_state(A, B, diff_A, diff_B, sums_A, sums_B, x, 'A'):
                A.append(x)
                # Update diffs and sums
                for y in A[:-1]:
                    diff_A.add(x - y)
                    diff_A.add(y - x)
                sums_A.add(2*x)
                for y in A[:-1]:
                    sums_A.add(x + y)
            # Try B
            elif is_valid_state(A, B, diff_A, diff_B, sums_A, sums_B, x, 'B'):
                B.append(x)
                for y in B[:-1]:
                    diff_B.add(x - y)
                    diff_B.add(y - x)
                sums_B.add(2*x)
                for y in B[:-1]:
                    sums_B.add(x + y)
            else:
                # Skip x
                pass
                
        # End of candidates
        if len(A) + len(B) > len(best_A) + len(best_B):
            best_A = A[:]
            best_B = B[:]
            
            # If we have a good size, we might stop early? 
            # No, keep searching for better.
            
    # Fallback if nothing found
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    # Final Validation
    def _valid(A, B):
        A = np.asarray(A, dtype=np.int64)
        B = np.asarray(B, dtype=np.int64)
        if A.size == 0 or B.size == 0:
            return False
        if A.max() > 256 or B.max() > 256 or A.min() < 1 or B.min() < 1:
            return False
        sa = A[:, None] + A[None, :]
        if sa.size != len(np.unique(sa)):
            return False
        sb = B[:, None] + B[None, :]
        if sb.size != len(np.unique(sb)):
            return False
        da = np.unique(A[:, None] - A[None, :])
        db = np.unique(B[:, None] - B[None, :])
        if 0 in da:
            da = da[da != 0]
        if 0 in db:
            db = db[db != 0]
        if len(np.intersect1d(da, db, assume_unique=False)) > 0:
            return False
        return True

    if not _valid(best_A, best_B):
        # Emergency fallback
        return [1, 2]
        
    return [best_A, best_B]
