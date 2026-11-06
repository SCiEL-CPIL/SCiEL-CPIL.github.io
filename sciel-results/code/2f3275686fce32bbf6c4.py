import time
import sys
from functools import lru_cache

def construct():
    t0 = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill
    
    # Step 1: Generate all smooth numbers (2^a * 3^b * 5^c) <= 128
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 1 <= v <= 128 and v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # is_smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True
    
    # Map value to index for quick lookup
    val_to_idx = {v: i for i, v in enumerate(smooth)}
    
    # Step 2: Precompute divisibility pairs
    # div_conflict[i][j] = True if smooth[i] divides smooth[j] or vice versa (i != j)
    div_conflict = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        si = smooth[i]
        for j in range(i + 1, n_smooth):
            sj = smooth[j]
            if sj % si == 0 or si % sj == 0:
                div_conflict[i][j] = True
                div_conflict[j][i] = True
    
    # Step 3: For each target 1..128, find minimum-size antichain representation
    # Use DFS with pruning for each target
    best_for_sum = [None] * 129  # index 0..128, stores (count, tuple_of_indices)
    
    # Size 1: single smooth number
    for i in range(n_smooth):
        s = smooth[i]
        if s <= 128:
            best_for_sum[s] = (1, (i,))
    
    # For targets not yet covered, try size 2, 3, etc.
    # We'll do a BFS-like approach: for each target, try to find smallest antichain
    uncovered = [n for n in range(1, 129) if best_for_sum[n] is None]
    
    if uncovered:
        # Precompute for each index, the list of indices that can follow (no conflict, larger value)
        # compatible_after[i] = sorted list of j > i such that not div_conflict[i][j]
        compatible_after = [[] for _ in range(n_smooth)]
        for i in range(n_smooth):
            for j in range(i + 1, n_smooth):
                if not div_conflict[i][j]:
                    compatible_after[i].append(j)
        
        # For each uncovered target, do DFS to find minimum-size antichain summing to target
        for target in uncovered:
            if time.time() - t0 > DEADLINE:
                break
            
            found = False
            max_size = 8  # limit search depth
            
            # DFS: find antichain of size k summing to target
            # We'll try sizes 2, 3, 4, ... until found
            for target_size in range(2, max_size + 1):
                if time.time() - t0 > DEADLINE:
                    break
                
                # Backtracking search
                # State: (start_idx, remaining_sum, depth, chosen)
                # We want to find any valid antichain of exactly target_size elements summing to target
                
                def dfs(start, rem, depth, chosen):
                    if time.time() - t0 > DEADLINE:
                        return None
                    if depth == target_size:
                        if rem == 0:
                            return tuple(chosen)
                        else:
                            return None
                    # Pruning: if rem <= 0, fail
                    if rem <= 0:
                        return None
                    # Pruning: minimum possible sum with remaining depth
                    # The smallest smooth number is 1, but we need distinct and antichain
                    # Simple bound: rem must be at least (target_size - depth) * 1
                    if rem < (target_size - depth):
                        return None
                    # Pruning: maximum possible sum
                    # The largest we can pick from start onwards
                    # For simplicity, use rem <= (max smooth) * (target_size - depth)
                    
                    # Iterate through possible next elements
                    for i in range(start, n_smooth):
                        si = smooth[i]
                        if si >= rem:
                            break
                        # Check if si can be in antichain with all chosen
                        can_add = True
                        for c in chosen:
                            if div_conflict[i][c]:
                                can_add = False
                                break
                        if not can_add:
                            continue
                        chosen.append(i)
                        result = dfs(i + 1, rem - si, depth + 1, chosen)
                        chosen.pop()
                        if result is not None:
                            return result
                    return None
                
                result = dfs(0, target, 0, [])
                if result is not None:
                    best_for_sum[target] = (target_size, result)
                    found = True
                    break
                if time.time() - t0 > DEADLINE:
                    break
        
        # If still uncovered, try a different approach: direct enumeration for remaining
        still_uncovered = [n for n in range(1, 129) if best_for_sum[n] is None]
        for target in still_uncovered:
            if time.time() - t0 > DEADLINE:
                break
            # Brute force with early termination
            # Try all antichain subsets up to size 10
            found = False
            for size in range(2, 11):
                if time.time() - t0 > DEADLINE:
                    break
                # Recursive backtracking
                def bt(start, rem, depth, chosen):
                    if time.time() - t0 > DEADLINE:
                        return None
                    if depth == size:
                        return tuple(chosen) if rem == 0 else None
                    if rem <= 0:
                        return None
                    for i in range(start, n_smooth):
                        si = smooth[i]
                        if si >= rem:
                            break
                        ok = True
                        for c in chosen:
                            if div_conflict[i][c]:
                                ok = False
                                break
                        if not ok:
                            continue
                        chosen.append(i)
                        res = bt(i + 1, rem - si, depth + 1, chosen)
                        chosen.pop()
                        if res is not None:
                            return res
                    return None
                result = bt(0, target, 0, [])
                if result is not None:
                    best_for_sum[target] = (size, result)
                    found = True
                    break
    
    # Step 4: Build answers
    answers = []
    for n in range(1, 129):
        if best_for_sum[n] is not None:
            count, indices = best_for_sum[n]
            row = [smooth[i] for i in indices]
            row.sort()
            answers.append(row)
        else:
            # Fallback - should not happen
            if is_sm[n]:
                answers.append([n])
            else:
                # Find any valid representation as last resort
                # Try 1 + (n-1) if n-1 is smooth and 1 doesn't divide n-1 (it does, so this is invalid)
                # Try to find two smooth numbers that sum to n and form antichain
                found_row = None
                for i in range(n_smooth):
                    si = smooth[i]
                    if si >= n:
                        break
                    rem = n - si
                    if rem in smooth_set and rem != si:
                        j = val_to_idx[rem]
                        if i != j and not div_conflict[i][j]:
                            found_row = sorted([si, rem])
                            break
                if found_row is None:
                    # Last resort: just output something (will be invalid but better than crash)
                    found_row = [n] if is_sm[n] else [1, n - 1]
                answers.append(found_row)
    
    return answers
