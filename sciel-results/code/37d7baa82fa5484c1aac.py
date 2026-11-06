import time

def construct():
    start_time = time.time()
    MAX_TARGET = 128
    
    # 1. Generate 2-3-5 smooth numbers up to MAX_TARGET
    smooth_numbers = []
    for i in range(1, MAX_TARGET + 1):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
            
    # Sort them
    smooth_numbers.sort()
    N = len(smooth_numbers)
    
    # Pre-compute divisibility matrix
    # divs[i][j] is True if smooth_numbers[i] divides smooth_numbers[j]
    divs = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                if smooth_numbers[j] % smooth_numbers[i] == 0:
                    divs[i][j] = True
    
    # Pre-compute all valid antichains (subsets where no element divides another)
    # An antichain is a subset where no two elements have a divisibility relation
    # We'll use bitmask representation for efficiency
    # Since N can be up to ~38, we can't enumerate all 2^N subsets directly
    # Instead, we'll build antichains incrementally
    
    # For each possible size k, find all antichains of that size
    # Store them as tuples of indices along with their sum
    
    # Key insight: we only need antichains with sum <= MAX_TARGET
    # Also, we want minimal k for each target
    
    # Build all antichains up to size 5 (should cover all targets)
    antichains_by_size = {k: [] for k in range(1, 6)}
    
    # Helper to check if a set of indices forms an antichain
    def is_antichain(indices):
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if divs[indices[i]][indices[j]] or divs[indices[j]][indices[i]]:
                    return False
        return True
    
    # Generate antichains of size 1
    for i in range(N):
        antichains_by_size[1].append((tuple([i]), smooth_numbers[i]))
    
    # Generate antichains of size 2
    for i in range(N):
        for j in range(i + 1, N):
            if not (divs[i][j] or divs[j][i]):
                s = smooth_numbers[i] + smooth_numbers[j]
                if s <= MAX_TARGET:
                    antichains_by_size[2].append((tuple([i, j]), s))
    
    # Generate antichains of size 3
    for i in range(N):
        for j in range(i + 1, N):
            if divs[i][j] or divs[j][i]:
                continue
            for k in range(j + 1, N):
                if divs[i][k] or divs[k][i] or divs[j][k] or divs[k][j]:
                    continue
                s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k]
                if s <= MAX_TARGET:
                    antichains_by_size[3].append((tuple([i, j, k]), s))
    
    # Generate antichains of size 4
    for i in range(N):
        for j in range(i + 1, N):
            if divs[i][j] or divs[j][i]:
                continue
            for k in range(j + 1, N):
                if divs[i][k] or divs[k][i] or divs[j][k] or divs[k][j]:
                    continue
                for l in range(k + 1, N):
                    if divs[i][l] or divs[l][i] or divs[j][l] or divs[l][j] or divs[k][l] or divs[l][k]:
                        continue
                    s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[l]
                    if s <= MAX_TARGET:
                        antichains_by_size[4].append((tuple([i, j, k, l]), s))
    
    # Generate antichains of size 5 (only if needed, and with pruning)
    for i in range(N):
        for j in range(i + 1, N):
            if divs[i][j] or divs[j][i]:
                continue
            for k in range(j + 1, N):
                if divs[i][k] or divs[k][i] or divs[j][k] or divs[k][j]:
                    continue
                for l in range(k + 1, N):
                    if divs[i][l] or divs[l][i] or divs[j][l] or divs[l][j] or divs[k][l] or divs[l][k]:
                        continue
                    for m in range(l + 1, N):
                        if divs[i][m] or divs[m][i] or divs[j][m] or divs[m][j] or divs[k][m] or divs[m][k] or divs[l][m] or divs[m][l]:
                            continue
                        s = smooth_numbers[i] + smooth_numbers[j] + smooth_numbers[k] + smooth_numbers[l] + smooth_numbers[m]
                        if s <= MAX_TARGET:
                            antichains_by_size[5].append((tuple([i, j, k, l, m]), s))
    
    # Now find minimal representation for each target
    best_representation = [None] * (MAX_TARGET + 1)
    best_size = [float('inf')] * (MAX_TARGET + 1)
    
    # Iterate by size k from 1 to 5
    for k in range(1, 6):
        if time.time() - start_time > 50:
            break
        
        all_found = all(best_size[t] != float('inf') for t in range(1, MAX_TARGET + 1))
        if all_found:
            break
        
        for (indices, s) in antichains_by_size[k]:
            if best_size[s] == float('inf'):
                best_size[s] = k
                best_representation[s] = [smooth_numbers[i] for i in indices]
    
    # Construct the final answer
    answers = [None] * MAX_TARGET
    for t in range(1, MAX_TARGET + 1):
        if best_representation[t] is None:
            # Fallback: try larger k values if needed
            # This should be rare
            pass
        else:
            answers[t-1] = best_representation[t]
    
    return answers
