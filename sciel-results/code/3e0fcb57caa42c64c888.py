import time
import random

def construct():
    # Deadline guard: 90s limit, use 55s for safety
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    # Bitsets for fast collision checks
    # Sums: max sum is 256+256=512. We need bits 0..512.
    # Diffs: range is [-255, 255]. We offset by 255, so bits 0..510.
    
    def try_add(x, A, sums_A, diffs_A, diffs_B):
        # Check Sidon sums for A
        # New sums: x+x and x+a for all a in A
        new_sums_bits = 1 << (x + x)
        for a in A:
            new_sums_bits |= 1 << (x + a)
        
        if new_sums_bits & sums_A:
            return False
        
        # Check Difference constraints
        # New diffs: x-a and a-x for all a in A
        new_diffs_bits = 0
        for a in A:
            d1 = x - a
            d2 = a - x
            # Offset by 255 to handle negative indices
            new_diffs_bits |= 1 << (d1 + 255)
            new_diffs_bits |= 1 << (d2 + 255)
            
        # The new differences must not conflict with existing diffs in A or B
        if new_diffs_bits & (diffs_A | diffs_B):
            return False
            
        return True

    def add_to_set(x, A, sums_A, diffs_A):
        # Update sums
        sums_A |= 1 << (x + x)
        # Update diffs
        new_diffs_bits = 0
        for a in A:
            sums_A |= 1 << (x + a)
            d1 = x - a
            d2 = a - x
            new_diffs_bits |= 1 << (d1 + 255)
            new_diffs_bits |= 1 << (d2 + 255)
        diffs_A |= new_diffs_bits
        A.append(x)

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.seed(attempts)
        
        # Generate candidates
        candidates = list(range(1, 257))
        
        # Different ordering strategies
        strategy = attempts % 3
        if strategy == 0:
            random.shuffle(candidates)
        elif strategy == 1:
            # Small to large
            pass 
        else:
            # Large to small
            candidates.sort(reverse=True)
            
        A = []
        B = []
        sums_A = 0
        sums_B = 0
        diffs_A = 0
        diffs_B = 0
        
        for x in candidates:
            valid_A = try_add(x, A, sums_A, diffs_A, diffs_B)
            valid_B = try_add(x, B, sums_B, diffs_A, diffs_B)
            
            if valid_A and valid_B:
                # Tie-breaking: Balance sizes
                if len(A) < len(B):
                    add_to_set(x, A, sums_A, diffs_A)
                elif len(B) < len(A):
                    add_to_set(x, B, sums_B, diffs_B)
                else:
                    if random.random() < 0.5:
                        add_to_set(x, A, sums_A, diffs_A)
                    else:
                        add_to_set(x, B, sums_B, diffs_B)
            elif valid_A:
                add_to_set(x, A, sums_A, diffs_A)
            elif valid_B:
                add_to_set(x, B, sums_B, diffs_B)
                
        # Ensure non-empty
        if not A:
            # Try to add 1
            if try_add(1, A, sums_A, diffs_A, diffs_B):
                add_to_set(1, A, sums_A, diffs_A)
            else:
                A = [1]
                sums_A = 1 << 2
                diffs_A = 0
        if not B:
            # Try to add 2
            if try_add(2, B, sums_B, diffs_A, diffs_B):
                add_to_set(2, B, sums_B, diffs_B)
            else:
                B = [2]
                sums_B = 1 << 4
                diffs_B = 0
                
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A.copy()
            best_B = B.copy()
            
            # Optional: Early exit if we hit a very high score?
            # Max possible is likely around 30-32.
            if best_score >= 32:
                # We can break or continue to try for 33+
                # But 32 is likely near optimal for N=256.
                # Let's keep running to be sure.
                pass

    return [best_A, best_B]
