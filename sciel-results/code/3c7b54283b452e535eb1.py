import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    N = 1000
    max_sum = 2 * N
    
    def greedy_build(order, counts, in_set, S):
        """Build set greedily following given order."""
        for y in order:
            if in_set[y]:
                continue
            # Check if y can be added
            # Sums: y+x for x in S, and 2y
            if len(S) > 0:
                sums = np.array(S) + y
                if np.any(counts[sums] >= 2):
                    continue
            if counts[2 * y] >= 2:
                continue
            # Add y
            if len(S) > 0:
                counts[np.array(S) + y] += 1
            counts[2 * y] += 1
            in_set[y] = True
            S.append(y)
    
    def local_search(S, counts, in_set, deadline):
        """Improve set via insertions and swaps using vectorized checks."""
        current_size = len(S)
        
        while time.time() < deadline:
            # Phase 1: Try to insert any element
            if in_set.any():
                candidates = np.where(~in_set)[0]
                candidates = candidates[candidates >= 1]
                random.shuffle(candidates.tolist())
                
                inserted = False
                for y in candidates:
                    y = int(y)
                    if len(S) > 0:
                        sums = np.array(S) + y
                        if np.any(counts[sums] >= 2):
                            continue
                    if counts[2 * y] >= 2:
                        continue
                    # Add y
                    if len(S) > 0:
                        counts[np.array(S) + y] += 1
                    counts[2 * y] += 1
                    in_set[y] = True
                    S.append(y)
                    inserted = True
                    break
                
                if inserted:
                    continue
            
            # Phase 2: Try swaps to enable further insertions
            if len(S) < 2 or len(candidates) < 2:
                break
                
            found_swap = False
            # Sample limited swaps to keep time bounded
            n_swap_attempts = min(200, len(S) * len(candidates) // 10)
            
            for _ in range(n_swap_attempts):
                x_idx = random.randint(0, len(S) - 1)
                x = S[x_idx]
                y = int(random.choice(candidates))
                
                # Try swap: remove x, add y
                # First check if removing x helps
                S_backup = list(S)
                counts_backup = counts.copy()
                in_set_backup = in_set.copy()
                
                # Remove x
                if len(S) > 1:
                    others = np.array([v for v in S if v != x])
                    counts[others + x] -= 1
                counts[2 * x] -= 1
                in_set[x] = False
                S.remove(x)
                
                # Check if y can be added
                valid = True
                if len(S) > 0:
                    sums = np.array(S) + y
                    if np.any(counts[sums] >= 2):
                        valid = False
                if valid and counts[2 * y] >= 2:
                    valid = False
                
                if valid:
                    # Add y
                    if len(S) > 0:
                        counts[np.array(S) + y] += 1
                    counts[2 * y] += 1
                    in_set[y] = True
                    S.append(y)
                    
                    # Check if we can now insert something
                    new_candidates = np.where(~in_set)[0]
                    new_candidates = new_candidates[new_candidates >= 1]
                    if len(new_candidates) > 0:
                        random.shuffle(new_candidates.tolist())
                        for z in new_candidates[:50]:  # Limit check
                            z = int(z)
                            if len(S) > 0:
                                sums_z = np.array(S) + z
                                if np.any(counts[sums_z] >= 2):
                                    continue
                            if counts[2 * z] >= 2:
                                continue
                            # Insert z
                            if len(S) > 0:
                                counts[np.array(S) + z] += 1
                            counts[2 * z] += 1
                            in_set[z] = True
                            S.append(z)
                            found_swap = True
                            break
                    
                    if found_swap:
                        break
                    else:
                        # Revert: remove y, add x back
                        if len(S) > 1:
                            others_y = np.array([v for v in S if v != y])
                            counts[others_y + y] -= 1
                        counts[2 * y] -= 1
                        in_set[y] = False
                        S.remove(y)
                        
                        # Add x back
                        if len(S) > 0:
                            counts[np.array(S) + x] += 1
                        counts[2 * x] += 1
                        in_set[x] = True
                        S.append(x)
                else:
                    # Revert: add x back
                    if len(S) > 0:
                        counts[np.array(S) + x] += 1
                    counts[2 * x] += 1
                    in_set[x] = True
                    S.append(x)
                
                if found_swap:
                    break
            
            if not found_swap and time.time() > deadline - 5:
                break
        
        return S, counts, in_set
    
    best_size = 0
    best_S = []
    
    # Try multiple greedy orderings
    orderings = [
        list(range(1, N + 1)),           # Ascending
        list(range(N, 0, -1)),           # Descending
        sorted(range(1, N + 1), key=lambda x: random.random()),  # Random
    ]
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        in_set = np.zeros(N + 1, dtype=bool)
        S = []
        
        greedy_build(order, counts, in_set, S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        
        # Local search
        S, counts, in_set = local_search(S, counts, in_set, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    return best_S

# Test locally
if __name__ == "__main__":
    res = construct()
    print(len(res))
