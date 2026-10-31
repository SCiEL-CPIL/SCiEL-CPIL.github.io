import time
import random
import math

def calculate_score(U):
    if not U or 0 not in U:
        return 0.0
    U = sorted(list(set(U)))
    if not U:
        return 0.0
    max_u = max(U)
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0.0
    
    # Calculate U+U and U-U
    # For small sets, brute force is fine.
    # For larger sets, we might need optimization, but let's stick to small N first.
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    
    if len(sums) == 0:
        return 0.0
        
    ratio = len(diffs) / len(sums)
    if ratio <= 1.0:
        return 1.0 # Or 0? The formula gives < 1 if ratio < 1.
        # But score is "lower bound". If ratio < 1, log is negative.
        # The problem asks to MAXIMIZE the sum.
        # Usually we want ratio > 1.
        
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial candidate: Powers of 2 minus 1
    # 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023
    # Let's check size 6: 0, 1, 3, 7, 15, 31
    best_U = [0, 1, 3, 7, 15, 31]
    best_score = calculate_score(best_U)
    
    # Local search
    # We can try to add elements, remove elements, or shift elements.
    # Given the "two-block" hint, maybe we should try structures like
    # A = [0..m] and B = [k, 2k, ...]
    # But random search from a good seed is robust.
    
    # Let's try a few random restarts or just refine the current one.
    # We can also try specific known good sets if we had them, but we don't.
    
    # Strategy:
    # 1. Try sets of form {0, 1, 3, ..., 2^k-1} for k=3 to 10.
    for k in range(3, 12):
        U = [2**i - 1 for i in range(k+1)]
        U[0] = 0
        s = calculate_score(U)
        if s > best_score:
            best_score = s
            best_U = U.copy()
            
    # 2. Local search refinement
    # We will perform random moves:
    # - Add a random number in [1, 2*max+10]
    # - Remove a random number (not 0)
    # - Replace a number with a nearby number
    
    current_U = best_U.copy()
    current_score = best_score
    
    # Use a fixed seed for reproducibility in case, but random is fine too.
    random.seed(42)
    
    max_steps = 5000
    steps = 0
    while time.time() < deadline and steps < max_steps:
        steps += 1
        U_new = current_U.copy()
        
        # Random move
        move = random.randint(0, 2)
        if move == 0 and len(U_new) < 20:
            # Add element
            max_val = max(U_new) if U_new else 0
            new_val = random.randint(1, max_val + 10)
            if new_val not in U_new:
                U_new.append(new_val)
        elif move == 1 and len(U_new) > 1:
            # Remove element
            idx = random.randint(1, len(U_new) - 1) # Don't remove 0
            U_new.pop(idx)
        else:
            # Modify element
            if len(U_new) > 1:
                idx = random.randint(1, len(U_new) - 1)
                old_val = U_new[idx]
                # Shift by small amount
                new_val = old_val + random.randint(-5, 5)
                if new_val >= 0 and new_val not in U_new:
                    U_new[idx] = new_val
        
        if not U_new or 0 not in U_new:
            continue
            
        U_new = sorted(list(set(U_new)))
        
        # Check time before expensive calculation?
        # calculate_score is O(N^2). If N is small (< 50), it's fast.
        s_new = calculate_score(U_new)
        
        if s_new > current_score:
            current_score = s_new
            current_U = U_new
            if current_score > best_score:
                best_score = current_score
                best_U = current_U.copy()
        else:
            # Simple hill climbing: revert if worse?
            # Or keep for diversity?
            # Let's just revert to keep it simple and fast.
            pass
            
    return best_U

# EVOLVE-BLOCK-START
def construct():
    # The function above is defined outside? No, I need to put everything in construct or use globals.
    # The prompt says "Keep the function names and signatures of the seed program".
    # Seed program only has `construct`.
    # So I must define helper functions inside `construct` or as globals if allowed.
    # "Python standard library... available".
    # I will define helpers inside construct to be safe, or just inline the logic.
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(list(set(U)))
        max_u = max(U)
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        if len(sums) == 0:
            return 0.0
        ratio = len(diffs) / len(sums)
        if ratio <= 1.0:
            return 1.0
        return 1 + math.log(ratio) / denom

    # Initial candidates
    best_U = [0, 1, 3, 7, 15, 31]
    best_score = calc_score(best_U)
    
    # Try geometric sequences
    for k in range(3, 12):
        U = [2**i - 1 for i in range(k+1)]
        U[0] = 0
        s = calc_score(U)
        if s > best_score:
            best_score = s
            best_U = U.copy()
            
    # Local search
    current_U = best_U.copy()
    current_score = best_score
    random.seed(42)
    
    max_steps = 10000
    steps = 0
    while time.time() < deadline and steps < max_steps:
        steps += 1
        U_new = current_U.copy()
        
        move = random.randint(0, 2)
        if move == 0 and len(U_new) < 25:
            max_val = max(U_new) if U_new else 0
            new_val = random.randint(1, max_val + 20)
            if new_val not in U_new:
                U_new.append(new_val)
        elif move == 1 and len(U_new) > 1:
            idx = random.randint(1, len(U_new) - 1)
            U_new.pop(idx)
        else:
            if len(U_new) > 1:
                idx = random.randint(1, len(U_new) - 1)
                old_val = U_new[idx]
                new_val = old_val + random.randint(-10, 10)
                if new_val >= 0 and new_val not in U_new:
                    U_new[idx] = new_val
        
        if not U_new or 0 not in U_new:
            continue
            
        U_new = sorted(list(set(U_new)))
        s_new = calc_score(U_new)
        
        if s_new > current_score:
            current_score = s_new
            current_U = U_new
            if current_score > best_score:
                best_score = current_score
                best_U = current_U.copy()
                
    return best_U
# EVOLVE-BLOCK-END
