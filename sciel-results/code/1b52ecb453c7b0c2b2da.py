import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    total = q ** n  # 6561
    
    # Precompute coverage: cov_matrix[i, j] = 1 if word j is within distance radius of word i
    # For radius=1: word i covers itself and all single-position changes
    cov_matrix = np.zeros((total, total), dtype=bool)
    
    for i in range(total):
        # Get digits of i
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        # Self
        cov_matrix[i, i] = True
        
        # Single position changes
        for pos in range(n):
            for val in range(q):
                if val == digits[pos]:
                    continue
                digits2 = digits[:]
                digits2[pos] = val
                j = 0
                for d in digits2:
                    j = j * q + d
                cov_matrix[i, j] = True
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    
    while uncovered.any():
        # For each candidate, count uncovered words it covers
        counts = cov_matrix[:, uncovered].sum(axis=1)
        best_i = np.argmax(counts)
        if counts[best_i] == 0:
            break
        
        chosen.append(best_i)
        uncovered[cov_matrix[best_i]] = False
        
        if time.time() > deadline:
            break
    
    # Local search: remove redundant words
    chosen_set = set(chosen)
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in chosen:
        cover_count[cov_matrix[cw]] = True  # just mark, we'll recount
    
    # Actually, let's maintain cover_count properly
    cover_count = np.zeros(total, dtype=np.int32)
    for cw in chosen:
        cover_count[cov_matrix[cw]] += 1
    
    # Remove redundant words
    removed = True
    while removed:
        removed = False
        for cw in list(chosen_set):
            if (cover_count[cov_matrix[cw]] > 1).all():
                chosen_set.discard(cw)
                cover_count[cov_matrix[cw]] -= 1
                removed = True
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    # Try swaps: add one word, remove multiple
    for _it in range(100):
        if time.time() > deadline:
            break
        
        best_gain = 0
        best_B = -1
        best_removals = []
        
        # Sample candidates
        candidates = np.random.choice(total, size=min(1000, total), replace=False)
        
        for B in candidates:
            if B in chosen_set:
                continue
            
            # Temporarily add B
            cover_count[cov_matrix[B]] += 1
            
            # Find removable words
            removable = []
            for A in chosen_set:
                if (cover_count[cov_matrix[A]] > 1).all():
                    removable.append(A)
            
            gain = len(removable) - 1
            if gain > best_gain:
                # Verify
                temp_count = cover_count.copy()
                for A in removable:
                    temp_count[cov_matrix[A]] -= 1
                if (temp_count > 0).all():
                    best_gain = gain
                    best_B = B
                    best_removals = removable
            
            # Undo
            cover_count[cov_matrix[B]] -= 1
        
        if best_gain > 0:
            B = best_B
            chosen_set.add(B)
            cover_count[cov_matrix[B]] += 1
            for A in best_removals:
                chosen_set.discard(A)
                cover_count[cov_matrix[A]] -= 1
    
    # Final cleanup
    removed = True
    while removed:
        removed = False
        for cw in list(chosen_set):
            if (cover_count[cov_matrix[cw]] > 1).all():
                chosen_set.discard(cw)
                cover_count[cov_matrix[cw]] -= 1
                removed = True
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    # Convert to words
    result = []
    for i in sorted(chosen_set):
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
    
    # Verify
    covered = np.zeros(total, dtype=bool)
    for w in result:
        idx = 0
        for d in w:
            idx = idx * q + d
        covered[cov_matrix[idx]] = True
    
    if not covered.all():
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return result
