"""Construct a large 4-AP-free subset of [1,512]."""
import time
import random

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    
    n = 512
    k = 4  # 4-term AP
    
    def has_4ap(a, b, c, d):
        """Check if a, b, c, d form a non-constant 4-term AP."""
        return (b - a) == (c - b) == (d - c) and (b - a) != 0
    
    def can_add(x, selected_set, selected_arr):
        """Check if adding x to selected_set creates a 4-AP."""
        # x could be the first, second, third, or fourth term
        # If x is 1st: x, x+d, x+2d, x+3d all in set
        # If x is 2nd: x-d, x, x+d, x+2d all in set
        # If x is 3rd: x-2d, x-d, x, x+d all in set
        # If x is 4th: x-3d, x-2d, x-d, x all in set
        
        # x is 1st term: need x+d, x+2d, x+3d in set
        for d in range(1, (n - x) // 3 + 1):
            if selected_arr[x + d] and selected_arr[x + 2*d] and selected_arr[x + 3*d]:
                return False
        
        # x is 2nd term: need x-d, x+d, x+2d in set
        for d in range(1, x):
            if x + 2*d > n:
                break
            if selected_arr[x - d] and selected_arr[x + d] and selected_arr[x + 2*d]:
                return False
        
        # x is 3rd term: need x-2d, x-d, x+d in set
        for d in range(1, (x + 1) // 2):
            if x + d > n:
                break
            if selected_arr[x - 2*d] and selected_arr[x - d] and selected_arr[x + d]:
                return False
        
        # x is 4th term: need x-3d, x-2d, x-d in set
        for d in range(1, (x + 2) // 3 + 1):
            if selected_arr[x - 3*d] and selected_arr[x - 2*d] and selected_arr[x - d]:
                return False
        
        return True
    
    # Phase 1: Base-3 Cantor set (numbers with no 1 in base-3, shifted to [1,512])
    selected_arr = [False] * (n + 1)
    selected_set = set()
    
    # Cantor set: numbers in [1, 3^m] whose base-3 digits are only 0 or 2
    # 3^6 = 729 > 512, so use up to 3^5 = 243, then extend
    # Actually, let's generate all numbers in [1,512] with base-3 digits in {0,2}
    # But we need to be careful: the standard Cantor construction gives a 3-AP-free set
    # For 4-AP-free, we need a different approach.
    
    # Better: use the base-4 construction. Numbers with base-4 digits in {0,1}
    # give a 4-AP-free set (Salem-Spencer type).
    # 4^4 = 256, 4^5 = 1024. So numbers in [0, 255] with base-4 digits in {0,1}
    # gives 2^4 = 16 numbers... that's too few.
    
    # Actually, the standard result: numbers with base-4 digits in {0,1} are 4-AP-free.
    # For [1, 512], we can use numbers whose base-4 representation uses only digits 0 and 1.
    # 4^4 = 256, so numbers 0..255 with base-4 digits in {0,1}: 2^4 = 16 numbers.
    # That's way too few.
    
    # Let me reconsider. The greedy algorithm in the seed gives ~120-130.
    # A better approach: use a greedy with randomization and multiple restarts.
    
    # Phase 1: Greedy construction (deterministic, same as seed but optimized)
    for x in range(1, n + 1):
        if can_add(x, selected_set, selected_arr):
            selected_arr[x] = True
            selected_set.add(x)
    
    best_size = len(selected_set)
    best_set = set(selected_set)
    
    # Phase 2: Try adding elements that greedy missed (in different orders)
    # Phase 3: Local search - try to add elements by removing others
    
    # Local search: try to improve by adding one element and removing minimal others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        current_arr = [False] * (n + 1)
        for x in current_set:
            current_arr[x] = True
        
        # Try adding each unselected element
        for x in range(1, n + 1):
            if current_arr[x]:
                continue
            
            # Find all 4-APs that would be created by adding x
            conflicts = set()
            
            # x is 1st: x, x+d, x+2d, x+3d
            for d in range(1, (n - x) // 3 + 1):
                if current_arr[x + d] and current_arr[x + 2*d] and current_arr[x + 3*d]:
                    conflicts.update([x + d, x + 2*d, x + 3*d])
            
            # x is 2nd: x-d, x, x+d, x+2d
            for d in range(1, x):
                if x + 2*d > n:
                    break
                if current_arr[x - d] and current_arr[x + d] and current_arr[x + 2*d]:
                    conflicts.update([x - d, x + d, x + 2*d])
            
            # x is 3rd: x-2d, x-d, x, x+d
            for d in range(1, (x + 1) // 2):
                if x + d > n:
                    break
                if current_arr[x - 2*d] and current_arr[x - d] and current_arr[x + d]:
                    conflicts.update([x - 2*d, x - d, x + d])
            
            # x is 4th: x-3d, x-2d, x-d, x
            for d in range(1, (x + 2) // 3 + 1):
                if current_arr[x - 3*d] and current_arr[x - 2*d] and current_arr[x - d]:
                    conflicts.update([x - 3*d, x - 2*d, x - d])
            
            # If we can remove fewer than 1 element (i.e., 0 conflicts), just add
            if len(conflicts) == 0:
                current_arr[x] = True
                current_set.add(x)
                best_size += 1
                improved = True
                continue
            
            # Try removing the minimum set of conflicting elements
            # Simple heuristic: if conflicts size < 2, try removing them
            if len(conflicts) <= 2:
                new_set = current_set.copy()
                new_arr = current_arr.copy()
                for c in conflicts:
                    new_set.discard(c)
                    new_arr[c] = False
                new_set.add(x)
                new_arr[x] = True
                
                # Verify no 4-AP exists
                valid = True
                for a in new_set:
                    for b in new_set:
                        if b <= a:
                            continue
                        d1 = b - a
                        c = a + 2 * d1
                        dd = a + 3 * d1
                        if c in new_set and dd in new_set:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(new_set) > best_size:
                    best_set = new_set
                    best_size = len(new_set)
                    improved = True
                    break
        
        if time.time() >= deadline:
            break
    
    return sorted(best_set)
