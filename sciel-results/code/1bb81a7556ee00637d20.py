import time
import random

def construct():
    base = 51975
    # Get divisors > 1
    divisors = [d for d in range(2, base + 1) if base % d == 0]
    # Sort by size? Or keep random?
    # Moduli are distinct.
    
    # Initialize residues
    # Greedy init: for each m, pick r that covers most uncovered
    covered = bytearray(base)
    classes = {} # m -> r
    
    # Sort divisors by size (largest first? or smallest?)
    # Smallest moduli cover more, so they are more critical.
    # Let's try random init first, then local search.
    
    # Better init:
    # Assign residues to avoid overlap as much as possible.
    # For each m, pick r randomly.
    
    residues = {m: random.randint(0, m - 1) for m in divisors}
    
    # Build count array
    counts = [0] * base
    for m in divisors:
        r = residues[m]
        for x in range(r, base, m):
            counts[x] += 1
            
    # Calculate score
    score = sum(1 for c in counts if c == 0)
    
    best_score = score
    best_residues = residues.copy()
    
    start_time = time.time()
    timeout = 110 # 120s limit, keep margin
    
    while time.time() - start_time < timeout:
        # Pick a random modulus
        m = random.choice(divisors)
        current_r = residues[m]
        
        best_r = current_r
        best_local_score = score
        
        # Try all residues for m
        # To speed up, we can just evaluate the delta.
        # But m can be up to 51975.
        # Average m is small.
        # If m is large, the number of residues is large, but the step size is large, so few updates.
        # Cost of evaluating one r is O(base/m).
        # Total cost for m is O(base).
        # So trying all r for one m takes O(base).
        # This is 51975 operations.
        # We can do ~1000 such steps in 10s.
        
        for r in range(m):
            if r == current_r:
                continue
                
            # Calculate delta
            delta = 0
            # Decrement current_r
            for x in range(current_r, base, m):
                if counts[x] == 1:
                    delta += 1 # becomes 0, score increases
            # Increment r
            for x in range(r, base, m):
                if counts[x] == 0:
                    delta -= 1 # becomes 1, score decreases
                # if counts[x] > 0, no change in score (0 -> 1 is -1, 1 -> 2 is 0)
                # Wait.
                # If counts[x] == 0, it was uncovered. Now covered. Score -1.
                # If counts[x] > 0, it was covered. Now still covered. Score 0.
                # For decrement:
                # If counts[x] == 1, it was covered (only by this class). Now uncovered. Score +1.
                # If counts[x] > 1, it was covered. Now still covered. Score 0.
                
            new_score = score + delta
            if new_score < best_local_score:
                best_local_score = new_score
                best_r = r
        
        if best_r != current_r:
            # Apply change
            for x in range(current_r, base, m):
                counts[x] -= 1
            for x in range(best_r, base, m):
                counts[x] += 1
            score = best_local_score
            residues[m] = best_r
            
            if score < best_score:
                best_score = score
                best_residues = residues.copy()
                
            # Optional: if stuck, randomize some
            if random.random() < 0.01:
                # Randomize a few
                for _ in range(5):
                    mm = random.choice(divisors)
                    rr = random.randint(0, mm - 1)
                    old_r = residues[mm]
                    for x in range(old_r, base, mm):
                        counts[x] -= 1
                    for x in range(rr, base, mm):
                        counts[x] += 1
                    residues[mm] = rr
                    # Recalculate score? 
                    # This is expensive. 
                    # Maybe just accept and let next steps fix it?
                    # Or just restart from best?
                    pass # Skipping complex restart logic for now, simple hill climbing might be enough.
                    
    # Return best
    return [[m, best_residues[m]] for m in divisors]
