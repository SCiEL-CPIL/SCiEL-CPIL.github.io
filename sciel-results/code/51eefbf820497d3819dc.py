"""
Strategy: Simulated Annealing Local Search for 4-term-progression-free sets.
1. Start with a base-3 construction (digits 0,1) which is 4-AP-free and large.
2. Use randomized local search (simulated annealing) to add/remove elements.
   - Move: Pick a random candidate to add. If it creates a 4-AP, pick a random
     element from the conflicting APs to remove.
   - Accept if size increases, or with probability exp(-delta/T) if size decreases
     (to escape local optima).
3. Time-boxed to finish well before 90s.
"""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 512
    
    # --- Helper: Check if a set contains a non-constant 4-term AP ---
    def has_4ap(s):
        # Returns True if s contains a 4-AP
        s_list = sorted(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                d = b - a
                if d <= 0:
                    continue
                c = b + d
                e = c + d
                if c in s and e in s:
                    return True
        return False

    def get_conflicting_elements(s, x):
        """
        Returns a list of 4-tuples representing 4-APs that would be formed if x were added to s.
        """
        conflicts = []
        
        # Case 1: x is the 1st term (x, x+d, x+2d, x+3d)
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if (x+d) in s and (x+2*d) in s and (x+3*d) in s:
                conflicts.append((x, x+d, x+2*d, x+3*d))
                
        # Case 2: x is the 2nd term (x-d, x, x+d, x+2d)
        max_d = min(x-1, (N-x)//2)
        for d in range(1, max_d + 1):
            if (x-d) in s and (x+d) in s and (x+2*d) in s:
                conflicts.append((x-d, x, x+d, x+2*d))
                
        # Case 3: x is the 3rd term (x-2d, x-d, x, x+d)
        max_d = min((x-1)//2, N-x)
        for d in range(1, max_d + 1):
            if (x-2*d) in s and (x-d) in s and (x+d) in s:
                conflicts.append((x-2*d, x-d, x, x+d))
                
        # Case 4: x is the 4th term (x-3d, x-2d, x-d, x)
        max_d = (x-1)//3
        for d in range(1, max_d + 1):
            if (x-3*d) in s and (x-2*d) in s and (x-d) in s:
                conflicts.append((x-3*d, x-2*d, x-d, x))
                
        return conflicts

    # --- Initial Construction: Base 3 ---
    current_set = set()
    for i in range(1, 513):
        temp = i
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            current_set.add(i)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # --- Simulated Annealing ---
    T = 1.0
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        if not current_set:
            break
            
        x = rng.randint(1, N)
        
        if x in current_set:
            # If we picked an element already in the set, try removing it with low probability
            if rng.random() < 0.05:
                to_remove = rng.choice(list(current_set))
                current_set.remove(to_remove)
            continue
            
        # x is not in current_set. Try to add it.
        conflicts = get_conflicting_elements(current_set, x)
        
        if not conflicts:
            # No conflict, add it
            current_set.add(x)
            if len(current_set) > best_size:
                best_set = set(current_set)
                best_size = len(current_set)
        else:
            # Conflict exists.
            conflict = rng.choice(conflicts)
            candidates_to_remove = [e for e in conflict if e != x]
            if candidates_to_remove:
                to_remove = rng.choice(candidates_to_remove)
                # Perform swap: remove to_remove, add x
                current_set.remove(to_remove)
                current_set.add(x)
                
                # Check if the new set is valid
                if has_4ap(current_set):
                    # Revert
                    current_set.remove(x)
                    current_set.add(to_remove)
                else:
                    # Valid swap. Size unchanged.
                    # We don't update best_size here as size is same, 
                    # but this state might lead to improvements.
                    pass
        
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    # Final greedy pass on best_set
    s = set(best_set)
    for i in range(1, N+1):
        if i not in s:
            if not get_conflicting_elements(s, i):
                s.add(i)
    
    if len(s) > len(best_set):
        return sorted(s)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
