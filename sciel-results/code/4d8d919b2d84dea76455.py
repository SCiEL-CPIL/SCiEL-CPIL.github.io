import random
import time
import sys

def construct():
    n = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def has_4ap_with_new_element(S, x):
        """
        Check if adding x to S creates a 4-AP.
        S is a set of integers.
        Returns True if a 4-AP is formed, False otherwise.
        """
        # x can be at position 0, 1, 2, 3 in the AP
        # 1. x is 1st term: x, x+d, x+2d, x+3d
        #    d can be 1 to (n-x)//3
        for d in range(1, (n - x) // 3 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return True
        # 2. x is 2nd term: x-d, x, x+d, x+2d
        #    d can be 1 to min(x-1, (n-x)//2)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return True
        # 3. x is 3rd term: x-2d, x-d, x, x+d
        #    d can be 1 to min((x-1)//2, n-x)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return True
        # 4. x is 4th term: x-3d, x-2d, x-d, x
        #    d can be 1 to (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return True
        return False

    def get_conflicts(S, x):
        """
        Returns a set of elements in S that must be removed to add x.
        i.e., elements involved in 4-APs with x.
        """
        conflicts = set()
        # 1. x is 1st term
        for d in range(1, (n - x) // 3 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                conflicts.update([x+d, x+2*d, x+3*d])
        # 2. x is 2nd term
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                conflicts.update([x-d, x+d, x+2*d])
        # 3. x is 3rd term
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                conflicts.update([x-2*d, x-d, x+d])
        # 4. x is 4th term
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                conflicts.update([x-3*d, x-2*d, x-d])
        return conflicts

    def greedy_init():
        S = set()
        for x in range(1, n + 1):
            if not has_4ap_with_new_element(S, x):
                S.add(x)
        return S

    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local Search with Swaps
    # We try to improve the set. 
    # Strategy: 
    # 1. Try to add any element. 
    # 2. If stuck, try to swap: remove one element, add another.
    #    We prioritize swaps that result in a valid set.
    #    To find good swaps, we can look for y not in S with fewest conflicts.
    #    If min conflicts is 1, we can swap.
    #    If min conflicts is 0, we can add.
    
    # To handle time, we run multiple restarts or long loops.
    # Let's do a loop that tries to improve, and if no improvement in a while, 
    # we perturb the set and continue.
    
    for restart in range(100): # Max restarts, but will break on time
        if time.time() > deadline:
            break
            
        # Start from best known or a perturbed version
        if restart == 0:
            current_set = set(best_set)
        else:
            # Perturb: remove k random elements, then greedy fill
            current_set = set(best_set)
            k = 5
            elems = list(current_set)
            random.shuffle(elems)
            for e in elems[:k]:
                current_set.remove(e)
            # Greedy fill
            for x in range(1, n + 1):
                if x not in current_set:
                    if not has_4ap_with_new_element(current_set, x):
                        current_set.add(x)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add
            added = False
            # Sort candidates? Or just iterate?
            # Iterating all y not in S is O(N). 
            # Checking each is O(N^2) worst case? No, O(N) per check roughly.
            # Total O(N^2) per pass. 512^2 ~ 260k ops. Very fast.
            
            candidates = [y for y in range(1, n + 1) if y not in current_set]
            random.shuffle(candidates) # Randomize order to avoid bias
            
            for y in candidates:
                if time.time() > deadline:
                    break
                if not has_4ap_with_new_element(current_set, y):
                    current_set.add(y)
                    improved = True
                    added = True
                    break # Restart check with new set
            
            if added:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            # If no add found, try swaps
            # Find y with minimum conflicts
            min_conflicts = n + 1
            best_y = -1
            best_conflict = -1
            
            for y in candidates:
                if time.time() > deadline:
                    break
                conflicts = get_conflicts(current_set, y)
                if len(conflicts) < min_conflicts:
                    min_conflicts = len(conflicts)
                    best_y = y
                    best_conflict = list(conflicts)[0] if conflicts else -1
                    if min_conflicts == 0:
                        # Should have been caught by add check, but just in case
                        break
            
            if min_conflicts == 1 and best_y != -1:
                # Swap best_conflict out, best_y in
                # We need to ensure removing best_conflict and adding best_y is valid.
                # Since best_conflict was the *only* conflict, removing it should make it valid.
                # But wait, get_conflicts returns ALL elements involved in APs with y.
                # If there is only 1, removing it breaks all APs involving y.
                # So the new set is valid.
                current_set.remove(best_conflict)
                current_set.add(best_y)
                improved = True
                # Size didn't change, so best_size doesn't update, 
                # but we might be able to add more later.
                
            elif min_conflicts == 0:
                # This should have been caught earlier
                current_set.add(best_y)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    # The actual implementation is above, but we need to wrap it in the expected format
    # The prompt asks to rewrite the program. The seed had construct() inside EVOLVE-BLOCK.
    # I will place the logic inside construct.
    
    n = 512
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def has_4ap_with_new_element(S, x):
        for d in range(1, (n - x) // 3 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return True
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return True
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return True
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return True
        return False

    def get_conflicts(S, x):
        conflicts = set()
        for d in range(1, (n - x) // 3 + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                conflicts.update([x+d, x+2*d, x+3*d])
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                conflicts.update([x-d, x+d, x+2*d])
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                conflicts.update([x-2*d, x-d, x+d])
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                conflicts.update([x-3*d, x-2*d, x-d])
        return conflicts

    def greedy_init():
        S = set()
        for x in range(1, n + 1):
            if not has_4ap_with_new_element(S, x):
                S.add(x)
        return S

    best_set = greedy_init()
    best_size = len(best_set)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        if restart == 0:
            current_set = set(best_set)
        else:
            current_set = set(best_set)
            k = 5
            elems = list(current_set)
            random.shuffle(elems)
            for e in elems[:k]:
                current_set.remove(e)
            for x in range(1, n + 1):
                if x not in current_set:
                    if not has_4ap_with_new_element(current_set, x):
                        current_set.add(x)
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            
            added = False
            candidates = [y for y in range(1, n + 1) if y not in current_set]
            random.shuffle(candidates)
            
            for y in candidates:
                if time.time() > deadline:
                    break
                if not has_4ap_with_new_element(current_set, y):
                    current_set.add(y)
                    improved = True
                    added = True
                    break
            
            if added:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                continue
            
            min_conflicts = n + 1
            best_y = -1
            best_conflict = -1
            
            for y in candidates:
                if time.time() > deadline:
                    break
                conflicts = get_conflicts(current_set, y)
                if len(conflicts) < min_conflicts:
                    min_conflicts = len(conflicts)
                    best_y = y
                    best_conflict = list(conflicts)[0] if conflicts else -1
                    if min_conflicts == 0:
                        break
            
            if min_conflicts == 1 and best_y != -1:
                current_set.remove(best_conflict)
                current_set.add(best_y)
                improved = True
            elif min_conflicts == 0:
                current_set.add(best_y)
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)

    return sorted(best_set)
# EVOLVE-BLOCK-END
