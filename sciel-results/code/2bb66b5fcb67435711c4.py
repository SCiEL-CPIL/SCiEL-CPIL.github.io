import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    N = 1000
    MAX_SUM = 2 * N
    
    # Greedy initialization
    S = set()
    counts = np.zeros(MAX_SUM + 1, dtype=np.int32)
    
    for x in range(1, N + 1):
        # Check if x can be added
        # New sums: 2*x and x+y for y in S
        valid = True
        if counts[2*x] >= 2:
            valid = False
        else:
            # Check x+y for y in S
            # We can check if any count[x+y] >= 2
            # To be efficient, we can just iterate if S is small, or use numpy
            # Since this is init, S grows slowly.
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
        if valid:
            S.add(x)
            counts[2*x] += 1
            for y in S:
                if y == x:
                    continue
                counts[x + y] += 1
    
    best_S = frozenset(S)
    best_size = len(S)
    
    # Local search with NumPy
    random.seed(42)
    current_S = set(S)
    S_list = list(S)
    S_arr = np.array(S_list, dtype=np.int32)
    current_counts = counts.copy()
    
    # Precompute candidates? No, just random.
    # To speed up "y not in S", maintain a boolean array or just check set.
    # Checking set is O(1).
    
    for it in range(100000):
        if time.time() > deadline:
            break
        
        # Try to add a random element
        y = random.randint(1, N)
        if y in current_S:
            continue
            
        # Tentative add
        old_counts = current_counts.copy()
        
        # Update counts
        current_counts[2*y] += 1
        # y + S_arr
        # We need to add 1 to counts[y + z] for all z in S_arr
        # Note: if y is in S_arr, we already handled 2y. 
        # The loop in greedy did: counts[2x] += 1, then for y in S (which includes x? No, S.add(x) happened before loop? 
        # In greedy: S.add(x) then loop y in S. If y==x, skip.
        # So counts[2x] is incremented once. counts[x+y] for y!=x incremented once.
        # Here: current_counts[2*y] += 1.
        # Then current_counts[y + S_arr] += 1.
        # This will increment counts[2*y] again if y is in S_arr? No, y is NOT in current_S.
        # So y + S_arr contains y+z for z in S. None are 2y.
        # So this is correct.
        
        sums = y + S_arr
        current_counts[sums] += 1
        
        if np.any(current_counts > 2):
            current_counts = old_counts
            # Try swap
            if current_S:
                x = random.choice(list(current_S))
                # Remove x
                # Restore counts for x
                # counts[2*x] -= 1
                # counts[x + z] -= 1 for z in S \ {x}
                
                # We need to be careful.
                # Let's just rollback to before adding y, then remove x, then try add y.
                # current_counts is already restored to state before adding y.
                
                # Remove x
                current_counts[2*x] -= 1
                # x + z for z in S, z != x
                mask = S_arr != x
                if np.any(mask):
                    sums_x = x + S_arr[mask]
                    current_counts[sums_x] -= 1
                
                # Update S
                current_S.remove(x)
                S_arr = S_arr[mask]
                
                # Now try to add y
                old_counts2 = current_counts.copy()
                current_counts[2*y] += 1
                sums_y = y + S_arr
                current_counts[sums_y] += 1
                
                if np.any(current_counts > 2):
                    # Rollback swap
                    current_counts = old_counts2
                    # Restore x
                    current_counts[2*x] += 1
                    if np.any(mask):
                         current_counts[sums_x] += 1
                    current_S.add(x)
                    S_arr = np.append(S_arr, x)
                    # Sort S_arr? Not strictly necessary for logic, but good for consistency
                    # S_arr = np.sort(S_arr)
                else:
                    # Accept swap
                    current_S.add(y)
                    S_arr = np.append(S_arr, y)
                    # S_arr = np.sort(S_arr)
                    
                    if len(current_S) > best_size:
                        best_S = frozenset(current_S)
                        best_size = len(current_S)
        else:
            # Accept add
            current_S.add(y)
            S_arr = np.append(S_arr, y)
            # S_arr = np.sort(S_arr)
            
            if len(current_S) > best_size:
                best_S = frozenset(current_S)
                best_size = len(current_S)
                
    return list(best_S)
