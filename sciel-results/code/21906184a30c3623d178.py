import time
import random
import itertools

def count_3aps(S):
    """Count distinct 3-APs in set S."""
    S = set(S)
    count = 0
    for a in S:
        for b in S:
            if a != b:
                c = 2*b - a
                if c in S and c != a and c != b:
                    # Ensure we count each AP only once
                    # An AP is defined by its min, mid, max.
                    # If we iterate a, b, c such that a < b < c, we count once.
                    # But simple iteration counts 6 times per AP (3! permutations? No, (a,b,c) is specific).
                    # Let's just count triples (a,b,c) with a < b < c and b-a == c-b.
                    pass
    # Better way:
    count = 0
    S_list = sorted(S)
    for i in range(len(S_list)):
        for j in range(i+1, len(S_list)):
            a = S_list[i]
            b = S_list[j]
            d = b - a
            c = b + d
            if c in S:
                count += 1
    return count

def is_4ap_free(S):
    """Check if set S has no 4-AP."""
    S = set(S)
    for a in S:
        for b in S:
            if a == b: continue
            d = b - a
            if d == 0: continue
            # Check if a, b, b+d, b+2d are in S
            # Or more generally, check all 4-APs containing a and b
            # It's easier to check all possible 4-APs in [1, 256]
            pass
    # Efficient check:
    # For each pair (a, b) in S, let d = b-a.
    # Check if a-d, a, b, b+d in S (b is 3rd term of 4-AP starting a-d? No.)
    # A 4-AP is {x, x+d, x+2d, x+3d}.
    # If a and b are in the AP, they can be any two of the four.
    # Let's just iterate d from 1 to 255.
    # For each d, iterate x from 1 to 256-3d.
    # If {x, x+d, x+2d, x+3d} subset of S, return False.
    for d in range(1, 256):
        for x in range(1, 257 - 3*d):
            if x in S and x+d in S and x+2*d in S and x+3*d in S:
                return False
    return True

def can_add(x, S):
    """Check if x can be added to S without creating a 4-AP."""
    S = set(S)
    # x must not be in S
    if x in S:
        return False
    # Check if adding x creates a 4-AP
    # The new 4-AP must contain x.
    # x can be the 1st, 2nd, 3rd, or 4th term.
    # 1st: x, x+d, x+2d, x+3d. Need x+d, x+2d, x+3d in S.
    # 2nd: x-d, x, x+d, x+2d. Need x-d, x+d, x+2d in S.
    # 3rd: x-2d, x-d, x, x+d. Need x-2d, x-d, x+d in S.
    # 4th: x-3d, x-2d, x-d, x. Need x-3d, x-2d, x-d in S.
    
    max_d = 255
    for d in range(1, max_d + 1):
        # 1st term
        if x + 3*d <= 256:
            if (x+d in S) and (x+2*d in S) and (x+3*d in S):
                return False
        # 2nd term
        if x - d >= 1 and x + 2*d <= 256:
            if (x-d in S) and (x+d in S) and (x+2*d in S):
                return False
        # 3rd term
        if x - 2*d >= 1 and x + d <= 256:
            if (x-2*d in S) and (x-d in S) and (x+d in S):
                return False
        # 4th term
        if x - 3*d >= 1:
            if (x-3*d in S) and (x-2*d in S) and (x-d in S):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin
    
    # 1. Greedy Construction
    S = set()
    # Sort candidates by potential? Just 1 to 256 for now.
    # A better greedy: pick element that forms most 3-APs with existing S.
    # Since S is empty initially, all are equal. Pick 1.
    # Then pick next that forms most 3-APs with S.
    
    # Let's do a simple greedy first, then local search.
    candidates = list(range(1, 257))
    random.shuffle(candidates) # Randomize to escape local optima? 
    # Actually, deterministic greedy is reproducible. Let's try multiple restarts with random shuffles.
    
    best_S = set()
    best_score = -1
    
    for restart in range(50): # 50 restarts
        if time.time() > deadline:
            break
            
        S = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        
        # Local Search: Swap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = count_3aps(S)
            
            # Try swapping each x in S with y not in S
            # To speed up, we can just try adding y and removing x
            # But we need to ensure validity.
            
            # Optimization: Precompute for each x in S, which y's can replace it?
            # Too complex. Just iterate.
            
            for x in list(S):
                if time.time() > deadline:
                    break
                S.remove(x)
                
                best_y = None
                best_y_score = -1
                
                # Find best y to add
                # We want to maximize 3-APs.
                # Approximate score: 3-APs involving y and S\{x}
                
                for y in range(1, 257):
                    if y in S:
                        continue
                    if can_add(y, S):
                        # Estimate gain
                        # Count 3-APs formed by y with S
                        # A 3-AP is (a, b, c) with b-a = c-b.
                        # y can be a, b, or c.
                        # If y is b: a, y, c. c = 2y-a.
                        # If y is a: y, b, c. b = y+d, c=y+2d.
                        # If y is c: a, b, y. a = y-2d, b=y-d.
                        
                        gain = 0
                        for b in S:
                            # y is middle
                            a = 2*b - y
                            if a in S and a != b and a != y:
                                gain += 1
                            # y is first
                            c = 2*b - y # No, if y is first, b is second, c is third.
                            # b = y+d => d = b-y. c = y+2d = y+2(b-y) = 2b-y.
                            # Wait, if y is first, the AP is y, b, c.
                            # Then b-y = c-b => c = 2b-y.
                            # So same formula.
                            # If y is third, AP is a, b, y.
                            # b-a = y-b => a = 2b-y.
                            # So for each b in S, if 2b-y is in S, we have a 3-AP.
                            # But we must ensure distinctness.
                            # If 2b-y == y, then b=y, impossible since y not in S.
                            # If 2b-y == b, then b=y, impossible.
                            # So just check if 2b-y in S.
                            
                        # Actually, the loop above counts:
                        # For each b in S, if a=2b-y in S, we found one.
                        # This covers y as middle (a,b,y) and y as third (a,b,y)?
                        # If y is middle: a, y, c. Then 2y-a=c. This is not covered by 2b-y.
                        # Let's re-evaluate.
                        # 3-APs involving y:
                        # 1. y, b, c: b-y = c-b => c = 2b-y. For each b in S, if c in S, gain++.
                        # 2. a, y, c: y-a = c-y => c = 2y-a. For each a in S, if c in S, gain++.
                        # 3. a, b, y: b-a = y-b => a = 2b-y. For each b in S, if a in S, gain++.
                        
                        # So we need 3 checks.
                        # But wait, case 1 and 3 are symmetric?
                        # Case 1: y is min.
                        # Case 3: y is max.
                        # Case 2: y is mid.
                        
                        # Let's just count properly.
                        # This inner loop is O(N). Total O(N^2) per swap.
                        # N=256, N^2=65k. 256 swaps * 65k = 16M ops.
                        # 50 restarts * 16M = 800M ops. Might be too slow in Python.
                        
                        # Optimization:
                        # We can just add y, count 3-APs, remove y.
                        # Counting 3-APs is O(N^2).
                        # This is still slow.
                        
                        # Let's stick to the gain calculation.
                        gain = 0
                        for b in S:
                            c = 2*b - y
                            if c in S and c != y and c != b:
                                gain += 1
                        for a in S:
                            c = 2*y - a
                            if c in S and c != y and c != a:
                                gain += 1
                        for b in S:
                            a = 2*b - y
                            if a in S and a != y and a != b:
                                gain += 1
                        
                        # Note: Some might be double counted?
                        # If y, b, c is an AP, it is counted in loop 1.
                        # If a, y, c is an AP, it is counted in loop 2.
                        # If a, b, y is an AP, it is counted in loop 3.
                        # These are distinct sets of positions for y.
                        
                        if gain > best_y_score:
                            best_y_score = gain
                            best_y = y
                            
                if best_y is not None:
                    S.add(best_y)
                    new_score = count_3aps(S)
                    if new_score > current_score:
                        improved = True
                    else:
                        S.remove(best_y)
                        S.add(x)
                else:
                    S.add(x)
            
            if best_score < count_3aps(S):
                best_score = count_3aps(S)
                best_S = set(S)
        
        if time.time() > deadline:
            break

    return sorted(best_S)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s margin
    
    # Helper functions defined inside to avoid global pollution? 
    # Or just define them globally if allowed. 
    # The prompt says "Keep the function names and signatures of the seed program".
    # I will define helpers inside or outside. Outside is cleaner.
    
    best_S = set()
    best_score = -1
    
    for restart in range(100): # 100 restarts
        if time.time() > deadline:
            break
            
        S = set()
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.add(x)
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = count_3aps(S)
            
            # Try swaps
            # To make it faster, we can limit the number of swaps per iteration
            # or just do one pass.
            
            for x in list(S):
                if time.time() > deadline:
                    break
                S.remove(x)
                
                best_y = None
                best_gain = -1
                
                # Estimate gain for each y
                # We can sample y's if too slow, but 256 is small.
                for y in range(1, 257):
                    if y in S:
                        continue
                    if can_add(y, S):
                        # Calculate gain
                        gain = 0
                        for b in S:
                            c = 2*b - y
                            if c in S and c != y and c != b:
                                gain += 1
                        for a in S:
                            c = 2*y - a
                            if c in S and c != y and c != a:
                                gain += 1
                        for b in S:
                            a = 2*b - y
                            if a in S and a != y and a != b:
                                gain += 1
                            
                        if gain > best_gain:
                            best_gain = gain
                            best_y = y
                            
                if best_y is not None:
                    S.add(best_y)
                    # Check if actually better
                    # We can just accept if gain > 0? 
                    # Gain is an approximation of new 3-APs minus lost 3-APs?
                    # No, gain is just new 3-APs formed by y.
                    # We lost 3-APs formed by x.
                    # So we should compare (current_score - lost_x + gain_y) vs current_score.
                    # But we don't know lost_x easily.
                    # So we must count_3aps(S) after adding.
                    
                    new_score = count_3aps(S)
                    if new_score > current_score:
                        improved = True
                    else:
                        S.remove(best_y)
                        S.add(x)
                else:
                    S.add(x)
            
            if count_3aps(S) > best_score:
                best_score = count_3aps(S)
                best_S = set(S)
        
        if time.time() > deadline:
            break

    if not best_S:
        # Fallback
        S = set()
        for x in range(1, 257):
            if can_add(x, S):
                S.add(x)
        return sorted(S)
        
    return sorted(best_S)
# EVOLVE-BLOCK-END
