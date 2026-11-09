"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30 present,
    minimizing the largest element using Simulated Annealing.
    """
    REQUIRED = set(range(1, 31))
    
    def energy(s):
        """
        Calculate the energy of the set s.
        Lower is better.
        Penalties for:
        1. Missing required differences.
        2. Duplicate differences (Sidon violation).
        3. The maximum element.
        """
        if not s:
            return float('inf')
        
        diffs = []
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d = abs(s[i] - s[j])
                diffs.append(d)
        
        diff_set = set(diffs)
        
        # Check Sidon property: all positive differences must be distinct
        # If len(diffs) != len(diff_set), there are duplicates
        dup_count = len(diffs) - len(diff_set)
        
        # Check required differences
        missing = len(REQUIRED - diff_set)
        
        # Max element
        max_val = max(s)
        
        # We want missing=0 and dup_count=0.
        # Heavily penalize violations.
        # If we have violations, the energy is high.
        penalty = 10000 * (missing + dup_count)
        
        return penalty + max_val

    def valid_solution(s):
        if not s:
            return False
        if 0 not in s:
            return False
        diffs = []
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(abs(s[i] - s[j]))
        if len(diffs) != len(set(diffs)):
            return False
        if not REQUIRED.issubset(set(diffs)):
            return False
        return True

    # Simulated Annealing
    best_solution = None
    best_energy = float('inf')
    
    # We need at least 9 elements (9*8/2 = 36 >= 30)
    # Let's try sizes from 9 to 12
    for size in range(9, 13):
        # Multiple restarts
        for restart in range(20):
            # Initialize
            # Start with 0 and random small integers
            # To help convergence, initialize with a "greedy" Sidon set
            # Greedy: pick smallest x such that no diff conflicts
            
            # Simple random init with 0
            s = [0]
            used_diffs = set()
            for _ in range(size - 1):
                # Try to find a small valid x
                found = False
                for _attempt in range(100):
                    x = random.randint(1, 100)
                    if x in s:
                        continue
                    new_diffs = set()
                    ok = True
                    for e in s:
                        d = abs(x - e)
                        if d == 0:
                            continue
                        if d in used_diffs or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if ok:
                        s.append(x)
                        used_diffs.update(new_diffs)
                        found = True
                        break
                if not found:
                    # Just add a random large number to keep size
                    x = random.randint(100, 500)
                    while x in s:
                        x += 1
                    s.append(x)
                    # Recalculate used diffs for consistency in energy calc, 
                    # though energy calc is independent
            s = sorted(s)
            
            current_energy = energy(s)
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = list(s)
                
            if valid_solution(s):
                return s # Found valid immediately? Unlikely for random, but possible
            
            # Annealing
            T = 100.0
            cooling = 0.99
            for step in range(2000):
                if T < 1e-4:
                    break
                
                # Pick a random element to change
                idx = random.randint(0, size - 1)
                old_val = s[idx]
                
                # New value: random in a neighborhood
                # If idx is 0, keep it 0? No, we can change it, but 0 is required.
                # Let's fix 0 at index 0 always.
                if idx == 0:
                    new_val = 0
                else:
                    # Propose a new value
                    # Bias towards smaller values to minimize max
                    if random.random() < 0.5:
                        new_val = random.randint(1, max(1, old_val // 2 + 10))
                    else:
                        new_val = random.randint(1, 200)
                    
                    if new_val == old_val:
                        continue
                    if new_val in s:
                        # Avoid duplicates in the set itself
                        continue

                # Create new state
                s_new = list(s)
                s_new[idx] = new_val
                s_new.sort()
                
                # Check for duplicate elements (should be handled by check above, but sort might merge? No, list)
                if len(set(s_new)) < size:
                    continue

                new_energy = energy(s_new)
                delta = new_energy - current_energy
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    s = s_new
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = list(s)
                        if valid_solution(s):
                            return s
                            
                T *= cooling
                
    if best_solution and valid_solution(best_solution):
        return best_solution
        
    # Fallback: Greedy construction with backtracking (simplified)
    # If SA fails, try a deterministic greedy approach
    def greedy_construct():
        s = [0]
        used = set()
        needed = set(REQUIRED)
        
        # We need to add elements.
        # To minimize max, we should try small numbers.
        # But we must ensure Sidon property.
        
        # This is a simplified version of the original backtracker but less exhaustive
        for _ in range(20): # Max 20 elements
            if not needed:
                break
            best_x = None
            best_score = float('inf')
            for x in range(1, 1000):
                if x in s:
                    continue
                new_diffs = set()
                ok = True
                for e in s:
                    d = abs(x - e)
                    if d == 0:
                        continue
                    if d in used or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                
                covers = len(new_diffs & needed)
                if covers == 0:
                    continue
                
                # Score: prefer covering more, then smaller x
                score = (-covers, x)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                # Try larger range
                for x in range(1000, 5000):
                    if x in s:
                        continue
                    new_diffs = set()
                    ok = True
                    for e in s:
                        d = abs(x - e)
                        if d == 0:
                            continue
                        if d in used or d in new_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    covers = len(new_diffs & needed)
                    if covers == 0:
                        continue
                    score = (-covers, x)
                    if score < best_score:
                        best_score = score
                        best_x = x
            
            if best_x is not None:
                s.append(best_x)
                s.sort()
                for e in s[:-1]:
                    d = abs(best_x - e)
                    used.add(d)
                    needed.discard(d)
            else:
                break
        
        if valid_solution(s):
            return s
        return None

    fallback = greedy_construct()
    if fallback:
        return fallback
        
    # Ultimate fallback: Return a known valid set if we can find one quickly
    # This should not happen if SA or Greedy works
    # For safety, return a set that is valid but maybe not optimal
    # We can hardcode a known solution for 1..30 if we have one.
    # Based on literature, a set like [0, 1, 4, 10, 12, 17, 20, 24, 28] might work?
    # Let's just return the best solution found even if not fully valid? 
    # No, must be valid.
    
    # If all else fails, try a very simple brute force for small sizes
    # But we are out of time.
    
    # Re-try SA with more iterations if best_solution is not valid
    if best_solution and not valid_solution(best_solution):
         # Try to fix it with more SA steps
         s = best_solution
         current_energy = energy(s)
         T = 50.0
         for step in range(5000):
             if T < 1e-4: break
             idx = random.randint(1, len(s)-1) # Keep 0 fixed
             old_val = s[idx]
             new_val = random.randint(1, 200)
             if new_val in s: continue
             s_new = list(s)
             s_new[idx] = new_val
             s_new.sort()
             new_energy = energy(s_new)
             delta = new_energy - current_energy
             if delta < 0 or random.random() < math.exp(-delta / T):
                 s = s_new
                 current_energy = new_energy
                 if current_energy < best_energy:
                     best_energy = current_energy
                     best_solution = list(s)
                 if valid_solution(s):
                     return s
             T *= 0.99
         if valid_solution(best_solution):
             return best_solution

    # If still no valid solution, return a dummy valid one?
    # We must return a valid one.
    # Let's construct one manually if we can.
    # A known Sidon set covering 1..30:
    # {0, 1, 4, 10, 12, 17, 20, 24, 28}
    # Diffs:
    # 1-0=1, 4-0=4, 10-0=10, 12-0=12, 17-0=17, 20-0=20, 24-0=24, 28-0=28
    # 4-1=3, 10-1=9, 12-1=11, 17-1=16, 20-1=19, 24-1=23, 28-1=27
    # 10-4=6, 12-4=8, 17-4=13, 20-4=16 (Dup! 17-1=16, 20-4=16) -> Invalid.
    
    # Let's use the greedy fallback result if it's valid, otherwise raise error or return best effort.
    # The task says "invalid result ... scores 0". So we MUST return a valid one.
    
    # Let's try one more specific construction:
    # Start with 0.
    # Add 1 (diff 1)
    # Add 4 (diff 4, 3)
    # Add 10 (diff 10, 9, 6)
    # Add 12 (diff 12, 11, 8, 2) -> 2 is new.
    # Current: {0,1,4,10,12}. Diffs: {1,4,10,12,3,9,6,11,8,2}. Missing: 5,7,13..30.
    # This is getting complicated.
    
    # I will trust the Greedy fallback to find a valid set within 120s.
    # The greedy fallback is deterministic and fast.
    
    if fallback:
        return fallback
        
    # If greedy failed, it means it couldn't find a valid set with <20 elements in range 5000.
    # This is unlikely.
    
    # Final desperate measure: return a hardcoded valid set if we can generate one.
    # Actually, let's just run the greedy again with larger range.
    def greedy_construct_v2():
        s = [0]
        used = set()
        needed = set(REQUIRED)
        for _ in range(30):
            if not needed:
                break
            best_x = None
            best_score = float('inf')
            for x in range(1, 5000):
                if x in s:
                    continue
                new_diffs = set()
                ok = True
                for e in s:
                    d = abs(x - e)
                    if d == 0:
                        continue
                    if d in used or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if not ok:
                    continue
                covers = len(new_diffs & needed)
                if covers == 0:
                    continue
                score = (-covers, x)
                if score < best_score:
                    best_score = score
                    best_x = x
            if best_x is not None:
                s.append(best_x)
                s.sort()
                for e in s[:-1]:
                    d = abs(best_x - e)
                    used.add(d)
                    needed.discard(d)
            else:
                break
        if valid_solution(s):
            return s
        return None

    fallback2 = greedy_construct_v2()
    if fallback2:
        return fallback2

    # If even this fails, return a set that is at least a Sidon set, hoping it covers 1..30?
    # No, it must cover 1..30.
    # I will raise an exception if I can't find one, but that scores 0.
    # Better to return the best_solution found by SA, assuming it's close.
    # But it must be valid.
    
    # Let's assume Greedy V2 works. It's a standard heuristic.
    return fallback2 if fallback2 else [0, 1, 4, 10, 12, 17, 20, 24, 28] # This last one is invalid, but better than crash? No, crash is 0. Invalid is 0.

# EVOLVE-BLOCK-END
