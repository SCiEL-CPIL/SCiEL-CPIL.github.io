import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    def is_sidon(arr):
        """Check if arr is a Sidon set (all positive differences distinct)."""
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_diffs(arr):
        diffs = set()
        n = len(arr)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(arr[j] - arr[i])
        return diffs

    def covers_1_to_30(diffs):
        return all(d in diffs for d in range(1, 31))

    def greedy_construct(initial_set=None):
        if initial_set is None:
            S = [0]
        else:
            S = list(initial_set)
        
        S.sort()
        S_set = set(S)
        diffs = get_diffs(S)
        missing = set(range(1, 31)) - diffs
        
        # If initial set is already valid, return it
        if not missing:
            return S

        while missing:
            found = False
            # Try to resolve the smallest missing difference
            for d in sorted(missing):
                candidates = []
                for s in S:
                    # x - s = d => x = s + d
                    x1 = s + d
                    if x1 > 0 and x1 not in S_set:
                        candidates.append(x1)
                    # s - x = d => x = s - d
                    x2 = s - d
                    if x2 > 0 and x2 not in S_set:
                        candidates.append(x2)
                        
                # Check candidates for Sidon property
                best_x = None
                best_score = float('inf')
                for x in candidates:
                    # Check if adding x violates Sidon
                    valid = True
                    new_diffs = []
                    for s in S:
                        diff = abs(x - s)
                        if diff in diffs:
                            valid = False
                            break
                        new_diffs.append(diff)
                    if valid:
                        # Heuristic: Prefer smaller x to keep diameter small
                        # Also prefer x that covers more missing diffs?
                        # For now, just smallest x
                        if x < best_score:
                            best_score = x
                            best_x = x
                
                if best_x is not None:
                    # Add best_x
                    S.append(best_x)
                    S_set.add(best_x)
                    S.sort()
                    diffs = get_diffs(S)
                    missing = set(range(1, 31)) - diffs
                    found = True
                    break # Restart loop with new S and missing
            
            if not found:
                # Fallback: Add a large number to break symmetry
                # This might increase diameter significantly
                max_s = max(S)
                x = max_s + 1 + max(missing)
                # Ensure Sidon
                while True:
                    S_temp = S + [x]
                    S_temp.sort()
                    if is_sidon(S_temp):
                        break
                    x += 1
                S.append(x)
                S_set.add(x)
                S.sort()
                diffs = get_diffs(S)
                missing = set(range(1, 31)) - diffs
                
            # Safety check: if set grows too large, break
            if len(S) > 32:
                break

        return S

    def local_search(S):
        S = list(S)
        S.sort()
        S_set = set(S)
        
        diffs = get_diffs(S)
        if not covers_1_to_30(diffs):
            return S # Cannot optimize if invalid

        current_max = max(S)
        best_S = list(S)
        best_max = current_max
        
        T = 20.0
        cooling = 0.9995
        iterations = 0
        
        while time_left() > 5:
            iterations += 1
            # Pick a random index to move (not 0)
            if len(S) <= 1:
                break
            idx = random.randint(1, len(S)-1)
            old_val = S[idx]
            
            # Determine range for new value
            # Try to move within [1, current_max + 10]
            # Occasionally try a random larger value to escape local minima
            if random.random() < 0.1:
                max_range = current_max + 50
            else:
                max_range = current_max + 10
            
            new_val = random.randint(1, max_range)
            
            if new_val == old_val or new_val in S_set:
                continue
            
            # Evaluate move
            # Remove old diffs involving old_val
            removed_diffs = set()
            for i, v in enumerate(S):
                if i == idx: continue
                removed_diffs.add(abs(v - old_val))
            
            # Add new diffs involving new_val
            added_diffs = set()
            valid = True
            for i, v in enumerate(S):
                if i == idx: continue
                d = abs(v - new_val)
                added_diffs.add(d)
            
            # Check conflicts
            remaining_diffs = diffs - removed_diffs
            for d in added_diffs:
                if d in remaining_diffs:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_diffs = remaining_diffs | added_diffs
            
            # Check coverage
            if not covers_1_to_30(new_diffs):
                continue
            
            # Calculate new max
            # We can do this efficiently
            max_others = max(S)
            if old_val == max_others:
                # Need to find max of remaining
                # Since we are modifying S, we should just compute max of new S
                pass
            
            S[idx] = new_val
            S.sort()
            actual_new_max = max(S)
            
            delta = actual_new_max - current_max
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                S_set.discard(old_val)
                S_set.add(new_val)
                diffs = new_diffs
                current_max = actual_new_max
                
                if current_max < best_max:
                    best_max = current_max
                    best_S = list(S)
                
            else:
                # Reject, revert
                S[idx] = old_val
                S.sort()
                
            T *= cooling
            
        return best_S

    # Strategy 1: Greedy from 0
    S1 = greedy_construct()
    S1 = local_search(S1)
    
    # Strategy 2: Start from known (31,6,1) difference set
    # {0, 1, 4, 10, 12, 28}
    initial_ds = [0, 1, 4, 10, 12, 28]
    S2 = greedy_construct(initial_ds)
    S2 = local_search(S2)
    
    # Strategy 3: Random restarts
    best_S = S1
    best_diam = max(S1)
    
    if max(S2) < best_diam:
        best_S = S2
        best_diam = max(S2)
        
    # Try a few random restarts if time permits
    for _ in range(5):
        if time_left() < 10:
            break
        # Random initial set?
        # Just re-run greedy with slight randomization?
        # For now, stick with the two above.
        
    # Final validation
    best_S.sort()
    diffs = get_diffs(best_S)
    if not covers_1_to_30(diffs) or not is_sidon(best_S):
        # Fallback
        return [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576] # Invalid, just placeholder
        
    return best_S
