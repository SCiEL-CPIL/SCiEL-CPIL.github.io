import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin for 120s limit
    
    # 1. Generate standard recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
        
    S = list(word)
    L = len(S)
    
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = L
    
    def try_remove(S, counts, i):
        n_local = n
        start_old = max(0, i - n_local + 1)
        end_old = min(i, len(S) - n_local)
        
        old_windows = []
        for j in range(start_old, end_old + 1):
            old_windows.append(tuple(S[j:j+n_local]))
            
        start_new = max(0, i - n_local + 1)
        end_new = min(i - 1, len(S) - 1 - n_local)
        
        new_windows_set = set()
        for k in range(start_new, end_new + 1):
            w = []
            for idx in range(k, k + n_local):
                if idx < i:
                    w.append(S[idx])
                else:
                    w.append(S[idx + 1])
            new_windows_set.add(tuple(w))
            
        for p in old_windows:
            if counts.get(p, 0) == 1:
                if p not in new_windows_set:
                    return False, None, None
                    
        for p in old_windows:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
                
        for k in range(start_new, end_new + 1):
            w = []
            for idx in range(k, k + n_local):
                if idx < i:
                    w.append(S[idx])
                else:
                    w.append(S[idx + 1])
            p_new = tuple(w)
            counts[p_new] = counts.get(p_new, 0) + 1
            
        S.pop(i)
        
        return True, S, counts

    def try_swap(S, counts, i):
        n_local = n
        # Swap i and i+1
        if i + 1 >= len(S):
            return False, None, None
            
        old_w = []
        for k in range(max(0, i-n_local+1), min(i+1, len(S)-n_local) + 1):
            old_w.append(tuple(S[k:k+n_local]))
            
        S_temp = S[:]
        S_temp[i], S_temp[i+1] = S_temp[i+1], S_temp[i]
        
        new_w = []
        for k in range(max(0, i-n_local+1), min(i+1, len(S)-n_local) + 1):
            new_w.append(tuple(S_temp[k:k+n_local]))
            
        new_w_set = set(new_w)
        
        for p in old_w:
            if counts.get(p, 0) == 1:
                if p not in new_w_set:
                    return False, None, None
        
        for p in old_w:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        for p in new_w:
            counts[p] = counts.get(p, 0) + 1
            
        S[i], S[i+1] = S[i+1], S[i]
        
        return True, S, counts

    # Main Search Loop
    while time.time() < deadline:
        made_change = False
        
        # 1. Greedy Removal (Single)
        indices = list(range(len(S)))
        random.shuffle(indices)
        for i in indices:
            if time.time() > deadline:
                break
            if i >= len(S):
                continue
            success, new_S, new_counts = try_remove(S, counts, i)
            if success:
                S = new_S
                counts = new_counts
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                break
        
        if made_change:
            continue
            
        # 2. Greedy Removal (Pair)
        pair_indices = list(range(len(S) - 1))
        random.shuffle(pair_indices)
        for i in pair_indices:
            if time.time() > deadline:
                break
            if i >= len(S) - 1:
                continue
            
            # Inline pair removal logic for speed/clarity
            n_local = n
            start_old = max(0, i - n_local + 1)
            end_old = min(i + 1, len(S) - n_local)
            old_windows = [tuple(S[j:j+n_local]) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n_local + 1)
            end_new = min(i - 1, len(S) - 2 - n_local)
            
            new_windows_set = set()
            for k in range(start_new, end_new + 1):
                w = []
                for idx in range(k, k + n_local):
                    if idx < i - 1:
                        w.append(S[idx])
                    elif idx == i - 1:
                        w.append(S[i - 1])
                    else:
                        w.append(S[idx + 2])
                new_windows_set.add(tuple(w))
                
            valid = True
            for p in old_windows:
                if counts.get(p, 0) == 1:
                    if p not in new_windows_set:
                        valid = False
                        break
                        
            if valid:
                for p in old_windows:
                    counts[p] -= 1
                    if counts[p] == 0:
                        del counts[p]
                        
                for k in range(start_new, end_new + 1):
                    w = []
                    for idx in range(k, k + n_local):
                        if idx < i - 1:
                            w.append(S[idx])
                        elif idx == i - 1:
                            w.append(S[i - 1])
                        else:
                            w.append(S[idx + 2])
                    p_new = tuple(w)
                    counts[p_new] = counts.get(p_new, 0) + 1
                    
                S.pop(i)
                S.pop(i)
                
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                break
                
        if made_change:
            continue
            
        # 3. Perturbation: Try valid swaps to escape local minima
        # Try a batch of random swaps
        swap_found = False
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(S) <= n:
                break
            i = random.randint(0, len(S) - 2)
            success, new_S, new_counts = try_swap(S, counts, i)
            if success:
                S = new_S
                counts = new_counts
                swap_found = True
                # Don't break, try to do more swaps? 
                # Or break and restart removal?
                # Breaking and restarting removal is better to see if swap enabled a removal.
                break
        
        if swap_found:
            continue
            
        # 4. If stuck, restart from best known state with new random seed
        S = best_S[:]
        # Rebuild counts for best_S
        counts = {}
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] = counts.get(p, 0) + 1
            
        # If we are stuck even from the best state, we might be at the local minimum.
        # To avoid infinite loop of restarting the same state, we can break if we've tried this many times.
        # But for now, let's just break to be safe.
        break

    return "".join(map(str, best_S))
