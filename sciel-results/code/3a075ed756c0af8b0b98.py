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
    
    # Encode windows as integers for faster hashing
    # Each symbol is 1..n, encode as n-digit base-(n+1) number
    base = n + 1
    base_powers = [base**i for i in range(n)]
    
    def encode_window(S, start):
        """Encode window starting at start as integer"""
        val = 0
        for j in range(n):
            val += S[start + j] * base_powers[j]
        return val
    
    def decode_window(val):
        """Decode integer back to tuple"""
        result = []
        for j in range(n):
            result.append(val % base)
            val //= base
        return tuple(result)
    
    # Initialize counts using integer encoding
    counts = {}
    for i in range(L - n + 1):
        w = encode_window(S, i)
        counts[w] = counts.get(w, 0) + 1
        
    best_S = S[:]
    best_len = L
    
    def get_affected_windows(S, i, n, removal_len=1):
        """
        Returns list of (old_windows, new_windows) for a removal of 'removal_len' chars starting at i.
        Uses integer encoding for speed.
        """
        if removal_len == 1:
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            old_windows = [encode_window(S, j) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n)
            new_windows = []
            for k in range(start_new, end_new + 1):
                # Build the new window by skipping position i
                val = 0
                for j in range(n):
                    idx = k + j
                    if idx >= i:
                        idx += 1  # skip the removed position
                    val += S[idx] * base_powers[j]
                new_windows.append(val)
            return old_windows, new_windows
        
        elif removal_len == 2:
            # Remove i and i+1
            start_old = max(0, i - n + 1)
            end_old = min(i + 1, len(S) - n)
            old_windows = [encode_window(S, j) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n)
            new_windows = []
            for k in range(start_new, end_new + 1):
                val = 0
                for j in range(n):
                    idx = k + j
                    if idx >= i:
                        idx += 2  # skip the two removed positions
                    val += S[idx] * base_powers[j]
                new_windows.append(val)
            return old_windows, new_windows
            
        elif removal_len == 3:
            # Remove i, i+1, i+2
            start_old = max(0, i - n + 1)
            end_old = min(i + 2, len(S) - n)
            old_windows = [encode_window(S, j) for j in range(start_old, end_old + 1)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 3 - n)
            new_windows = []
            for k in range(start_new, end_new + 1):
                val = 0
                for j in range(n):
                    idx = k + j
                    if idx >= i:
                        idx += 3  # skip the three removed positions
                    val += S[idx] * base_powers[j]
                new_windows.append(val)
            return old_windows, new_windows
        else:
            return [], []

    def check_and_apply_removal(S, counts, i, removal_len):
        """
        Checks if removal is valid. If so, applies it to S and counts.
        Returns True if applied.
        """
        if i < 0 or i + removal_len > len(S):
            return False
            
        old_windows, new_windows = get_affected_windows(S, i, n, removal_len)
        
        # Check validity: Any old window with count 1 must be in new_windows
        new_set = set(new_windows)
        for p in old_windows:
            if counts.get(p, 0) == 1:
                if p not in new_set:
                    return False
        
        # Apply removal
        for p in old_windows:
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        for p in new_windows:
            counts[p] = counts.get(p, 0) + 1
            
        # Modify S
        del S[i:i+removal_len]
        
        return True

    # Main Search Loop
    last_progress_time = time.time()
    restart_count = 0
    max_restarts = 50
    
    while time.time() < deadline:
        # 1. Greedy Removal Pass - Linear Scan with Step-Back
        made_change = False
        
        i = 0
        while i < len(S):
            if time.time() > deadline:
                break
            
            # Try removing at i
            if check_and_apply_removal(S, counts, i, 1):
                made_change = True
                # Do not increment i, because S[i] is now the old S[i+1]
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                    last_progress_time = time.time()
            else:
                i += 1
                
        if made_change:
            continue
            
        # 2. Pair Removal Pass
        made_change = False
        i = 0
        while i < len(S) - 1:
            if time.time() > deadline:
                break
                
            if check_and_apply_removal(S, counts, i, 2):
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                    last_progress_time = time.time()
            else:
                i += 1
                
        if made_change:
            continue
            
        # 3. Triple Removal Pass
        made_change = False
        i = 0
        while i < len(S) - 2:
            if time.time() > deadline:
                break
                
            if check_and_apply_removal(S, counts, i, 3):
                made_change = True
                if len(S) < best_len:
                    best_len = len(S)
                    best_S = S[:]
                    last_progress_time = time.time()
            else:
                i += 1
                
        if made_change:
            continue
            
        # 4. Stuck. Try Perturbation
        if len(S) > n:
            # Try multiple perturbation strategies
            perturbation_type = random.choice(['swap', 'slide', '2opt'])
            
            if perturbation_type == 'swap':
                # Random adjacent swap
                i = random.randint(0, len(S) - 2)
                
                S_temp = S[:]
                S_temp[i], S_temp[i+1] = S_temp[i+1], S_temp[i]
                
                start = max(0, i - n + 1)
                end = min(i + 1, len(S) - n)
                
                old_windows = [encode_window(S, j) for j in range(start, end + 1)]
                new_windows = [encode_window(S_temp, j) for j in range(start, end + 1)]
                
                new_set = set(new_windows)
                valid = True
                for p in old_windows:
                    if counts.get(p, 0) == 1:
                        if p not in new_set:
                            valid = False
                            break
                            
                if valid:
                    for p in old_windows:
                        counts[p] -= 1
                        if counts[p] == 0:
                            del counts[p]
                    for p in new_windows:
                        counts[p] = counts.get(p, 0) + 1
                    S[i], S[i+1] = S[i+1], S[i]
                    made_change = True
                    last_progress_time = time.time()
                    
            elif perturbation_type == 'slide':
                # Move element at position i to position i+k
                if len(S) > n + 3:
                    i = random.randint(0, len(S) - 4)
                    k = random.randint(1, 3)
                    j = i + k
                    if j < len(S):
                        S_temp = S[:]
                        # Remove from i and insert at j
                        elem = S_temp[i]
                        del S_temp[i]
                        S_temp.insert(j, elem)
                        
                        # Find affected windows
                        start = max(0, i - n + 1)
                        end = min(j + 1, len(S) - n)
                        
                        old_windows = [encode_window(S, pos) for pos in range(start, end + 1)]
                        new_windows = []
                        for pos in range(start, end + 1):
                            # Need to map old positions to new positions
                            # This is complex; skip if too complicated
                            pass
                        
                        # Simpler: just check if the move preserves all permutations
                        # by rebuilding counts temporarily
                        new_counts = {}
                        for pos in range(len(S_temp) - n + 1):
                            w = encode_window(S_temp, pos)
                            new_counts[w] = new_counts.get(w, 0) + 1
                        
                        # Check all old windows with count 1 are in new_counts
                        valid = True
                        for p in old_windows:
                            if counts.get(p, 0) == 1:
                                if new_counts.get(p, 0) == 0:
                                    valid = False
                                    break
                                    
                        if valid:
                            S = S_temp
                            counts = new_counts
                            made_change = True
                            last_progress_time = time.time()
                            
            elif perturbation_type == '2opt':
                # Swap elements at positions i and j (non-adjacent)
                if len(S) > n + 5:
                    i = random.randint(0, len(S) - 2)
                    j = random.randint(i + 2, len(S) - 1)
                    
                    S_temp = S[:]
                    S_temp[i], S_temp[j] = S_temp[j], S_temp[i]
                    
                    # Find all affected windows
                    start = max(0, min(i, j) - n + 1)
                    end = min(max(i, j) + 1, len(S) - n)
                    
                    old_windows = [encode_window(S, pos) for pos in range(start, end + 1)]
                    new_windows = [encode_window(S_temp, pos) for pos in range(start, end + 1)]
                    
                    new_set = set(new_windows)
                    valid = True
                    for p in old_windows:
                        if counts.get(p, 0) == 1:
                            if p not in new_set:
                                valid = False
                                break
                                
                    if valid:
                        for p in old_windows:
                            counts[p] -= 1
                            if counts[p] == 0:
                                del counts[p]
                        for p in new_windows:
                            counts[p] = counts.get(p, 0) + 1
                        S[i], S[j] = S[j], S[i]
                        made_change = True
                        last_progress_time = time.time()
        
        if not made_change:
            # Stuck completely.
            if time.time() - last_progress_time > 3.0:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                S = best_S[:]
                # Rebuild counts
                counts = {}
                for i in range(len(S) - n + 1):
                    p = encode_window(S, i)
                    counts[p] = counts.get(p, 0) + 1
                last_progress_time = time.time()
            else:
                pass
                
    return "".join(map(str, best_S))
