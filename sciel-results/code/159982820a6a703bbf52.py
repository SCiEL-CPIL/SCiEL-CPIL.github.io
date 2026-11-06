import time
import random
import bisect
import numpy as np

N = 256

def construct():
    start_time = time.time()
    deadline = 85.0

    def can_add_descending(S, a):
        """Check if 'a' can be added to S, where all elements in S are > a.
        Condition: No pair b, c in S (b < c) such that a divides (b+c)."""
        if not S:
            return True
        
        # Use numpy for speed
        S_arr = np.array(S)
        residues = S_arr % a
        
        # Count residues
        counts = np.bincount(residues, minlength=a)
        
        # Check for conflicts
        # For each residue r, check if there's a partner (-r) % a
        # If r == partner, need count >= 2; otherwise need both > 0
        partners = (-np.arange(a)) % a
        
        # For r != partner: need counts[r] > 0 and counts[partner] > 0
        # For r == partner: need counts[r] >= 2
        
        # Vectorized check
        # Create a mask where r != partner
        diff_mask = np.arange(a) != partners
        
        # For different residues: check if both exist
        # We need to avoid double-counting, so only check r < partner
        check_mask = diff_mask & (np.arange(a) < partners)
        conflict_diff = (counts[check_mask] > 0) & (counts[partners[check_mask]] > 0)
        
        # For same residues: check if count >= 2
        same_mask = ~diff_mask
        conflict_same = counts[same_mask] >= 2
        
        return not (np.any(conflict_diff) or np.any(conflict_same))

    def greedy_descending(start_from=None, end_at=None):
        S = []
        if start_from is None:
            start_from = N
        if end_at is None:
            end_at = 1
        for a in range(start_from, end_at - 1, -1):
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
        return S

    def local_search(S, max_time=None):
        """Improve set S using local search."""
        if max_time is None:
            max_time = deadline - (time.time() - start_time)
        
        S_set = set(S)
        improved = True
        while improved and time.time() - start_time < deadline:
            improved = False
            
            # 1. Try to add any missing element
            for a in range(1, N + 1):
                if a in S_set:
                    continue
                if can_add_descending(S, a):
                    pos = bisect.bisect_left(S, a)
                    S.insert(pos, a)
                    S_set.add(a)
                    improved = True
                    break  # Restart to check for new additions
            
            if improved:
                continue
                
            # 2. If no direct addition, try swaps
            for a in range(1, N + 1):
                if a in S_set:
                    continue
                
                # Identify conflicts for a
                n = len(S)
                conflict_count = [0] * n
                pairs = []
                
                for i in range(n):
                    bi = S[i]
                    target = (-bi) % a
                    for j in range(i + 1, n):
                        if S[j] % a == target:
                            pairs.append((i, j))
                            conflict_count[i] += 1
                            conflict_count[j] += 1
                            
                if not pairs:
                    continue
                    
                if len(pairs) == 1:
                    i, j = pairs[0]
                    for rem_idx in [i, j]:
                        S_new = S[:]
                        S_new.pop(rem_idx)
                        if can_add_descending(S_new, a):
                            pos = bisect.bisect_left(S_new, a)
                            S_new.insert(pos, a)
                            S = S_new
                            S_set = set(S)
                            improved = True
                            break
                    if improved:
                        break
                else:
                    # Find most frequent index
                    sorted_indices = sorted(range(n), key=lambda x: conflict_count[x], reverse=True)
                    for rem_idx in sorted_indices[:5]:
                        S_new = S[:]
                        S_new.pop(rem_idx)
                        if can_add_descending(S_new, a):
                            pos = bisect.bisect_left(S_new, a)
                            S_new.insert(pos, a)
                            S = S_new
                            S_set = set(S)
                            improved = True
                            break
                    if improved:
                        break
        
        # 3. Final polish: try adding elements again
        S_set = set(S)
        for a in range(1, N + 1):
            if a in S_set:
                continue
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
                S_set.add(a)
        
        return S

    best_set = []
    best_len = 0
    
    # Multi-start with different strategies
    # Strategy 1: Standard greedy from top
    S = greedy_descending()
    S = local_search(S)
    if len(S) > best_len:
        best_set = S[:]
        best_len = len(S)
    
    # Strategy 2: Greedy from top but skip some small numbers
    for skip_start in range(2, 20):
        if time.time() - start_time > deadline - 5:
            break
        S = greedy_descending(start_from=N, end_at=skip_start)
        # Then try to add smaller elements
        S_set = set(S)
        for a in range(skip_start - 1, 0, -1):
            if a in S_set:
                continue
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
                S_set.add(a)
        S = local_search(S)
        if len(S) > best_len:
            best_set = S[:]
            best_len = len(S)
    
    # Strategy 3: Randomized greedy
    random.seed(42)
    for trial in range(10):
        if time.time() - start_time > deadline - 5:
            break
        
        # Randomly decide to include or skip each number
        S = []
        for a in range(N, 0, -1):
            # With some probability, skip this number
            if random.random() < 0.1:
                continue
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
        
        S = local_search(S)
        if len(S) > best_len:
            best_set = S[:]
            best_len = len(S)
    
    # Strategy 4: Start with large numbers only, then fill in
    for threshold in [128, 100, 80, 64]:
        if time.time() - start_time > deadline - 5:
            break
        S = []
        for a in range(N, threshold, -1):
            if can_add_descending(S, a):
                pos = bisect.bisect_left(S, a)
                S.insert(pos, a)
        S = local_search(S)
        if len(S) > best_len:
            best_set = S[:]
            best_len = len(S)
    
    # Final polish on best set
    best_set = local_search(best_set)
    
    return best_set

