import random
import time
import numpy as np

def _greedy_build(N, order, deadline):
    """Build a B2[2] set using greedy with given order."""
    S = set()
    counts = np.zeros(2 * N + 1, dtype=np.int32)
    
    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        valid = True
        if counts[2 * x] >= 2:
            valid = False
        else:
            for y in S:
                if counts[x + y] >= 2:
                    valid = False
                    break
        if valid:
            S.add(x)
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
    
    return S, counts

def _local_search(S, counts, N, deadline, seed=42):
    """Perform local search to improve the B2[2] set."""
    random.seed(seed)
    current_S = set(S)
    S_arr = np.array(list(S), dtype=np.int32)
    current_counts = counts.copy()
    best_S = frozenset(current_S)
    best_size = len(current_S)
    
    # Precompute candidate pool
    all_candidates = set(range(1, N + 1))
    
    for it in range(50000):
        if time.time() > deadline:
            break
        
        # Try to add a random element from candidates
        available = all_candidates - current_S
        if not available:
            break
        
        y = random.choice(list(available))
        
        # Tentative add - track changes
        changed_indices = []
        
        # Check 2*y
        if current_counts[2 * y] >= 2:
            continue
        changed_indices.append(2 * y)
        
        # Check y + z for z in S
        valid = True
        for z in S_arr:
            s = y + z
            if current_counts[s] >= 2:
                valid = False
                break
            changed_indices.append(s)
        
        if not valid:
            # Try swap: remove a random element x, then add y
            if current_S:
                x = random.choice(list(current_S))
                
                # Remove x
                current_counts[2 * x] -= 1
                for z in S_arr:
                    if z != x:
                        current_counts[x + z] -= 1
                
                current_S.remove(x)
                mask = S_arr != x
                S_arr = S_arr[mask]
                
                # Now try to add y
                valid2 = True
                if current_counts[2 * y] >= 2:
                    valid2 = False
                else:
                    for z in S_arr:
                        if current_counts[y + z] >= 2:
                            valid2 = False
                            break
                
                if valid2:
                    # Accept swap
                    current_counts[2 * y] += 1
                    for z in S_arr:
                        current_counts[y + z] += 1
                    current_S.add(y)
                    S_arr = np.append(S_arr, y)
                    
                    if len(current_S) > best_size:
                        best_S = frozenset(current_S)
                        best_size = len(current_S)
                else:
                    # Rollback swap
                    current_counts[2 * x] += 1
                    for z in S_arr:
                        current_counts[x + z] += 1
                    current_S.add(x)
                    S_arr = np.append(S_arr, x)
        
        else:
            # Accept add
            current_counts[2 * y] += 1
            for z in S_arr:
                current_counts[y + z] += 1
            current_S.add(y)
            S_arr = np.append(S_arr, y)
            
            if len(current_S) > best_size:
                best_S = frozenset(current_S)
                best_size = len(current_S)
    
    return set(best_S)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    N = 1000
    
    best_S = set()
    best_size = 0
    
    # Try multiple greedy orderings
    orderings = [
        range(1, N + 1),  # ascending
        range(N, 0, -1),  # descending
        list(range(1, N + 1, 2)),  # odd first
        list(range(2, N + 1, 2)),  # even first
        list(range(1, N + 1, 3)),  # every 3rd
    ]
    
    for i, order in enumerate(orderings):
        if time.time() > deadline:
            break
        
        S, counts = _greedy_build(N, order, deadline)
        
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
        
        # Local search on this solution
        if time.time() < deadline - 20:  # Leave time for more orderings
            improved_S = _local_search(S, counts, N, deadline, seed=42 + i)
            if len(improved_S) > best_size:
                best_S = improved_S
                best_size = len(improved_S)
    
    # Final local search on best solution found
    if time.time() < deadline - 5 and best_S:
        # Rebuild counts for final search
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        for x in best_S:
            counts[2 * x] += 1
            for y in best_S:
                if x < y:
                    counts[x + y] += 1
        
        final_S = _local_search(best_S, counts, N, deadline, seed=100)
        if len(final_S) > best_size:
            best_S = final_S
            best_size = len(final_S)
    
    return list(best_S)
