import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    # Standard recursive construction
    def standard_construct(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    if n < 7:
        return standard_construct(n)
    
    # For n=7, use deletion-based optimization
    if n == 7:
        # Generate standard superpermutation
        S = [int(c) for c in standard_construct(n)]
        best_S = S[:]
        best_len = len(S)
        
        # Initialize permutation counts
        counts = defaultdict(int)
        for i in range(len(S) - n + 1):
            p = tuple(S[i:i+n])
            counts[p] += 1
        
        def try_remove(S, counts, i, n):
            """Try to remove position i. Returns True if successful."""
            # Calculate affected old windows
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            # Check if removal is valid (no count drops to 0)
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    return False
            
            # Perform removal
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # Calculate new straddling windows
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n)
            
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            return True
        
        # Deletion pass: try to remove each position
        i = 0
        while i < len(S):
            if try_remove(S, counts, i, n):
                best_len = len(S)
                best_S = S[:]
                # Don't increment i, try same position again
            else:
                i += 1
        
        # Multiple passes with randomization
        random.seed(42)
        while time.time() < deadline:
            # Random deletion pass
            i = 0
            improved = False
            while i < len(S):
                if try_remove(S, counts, i, n):
                    improved = True
                    best_len = len(S)
                    best_S = S[:]
                else:
                    i += 1
            
            if not improved:
                # Try random swaps to unlock deletions
                if len(S) > n:
                    i = random.randint(0, len(S) - 1)
                    j = random.randint(0, len(S) - 1)
                    if i != j:
                        if i > j:
                            i, j = j, i
                        # Check if swap is valid
                        valid = True
                        # Windows affected by swap
                        for pos in [i, j]:
                            start_w = max(0, pos - n + 1)
                            end_w = min(pos, len(S) - n)
                            for w in range(start_w, end_w + 1):
                                p = tuple(S[w:w+n])
                                if counts.get(p, 0) == 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            # Perform swap
                            for pos in [i, j]:
                                start_w = max(0, pos - n + 1)
                                end_w = min(pos, len(S) - n)
                                for w in range(start_w, end_w + 1):
                                    p = tuple(S[w:w+n])
                                    counts[p] -= 1
                                    if counts[p] == 0:
                                        del counts[p]
                            
                            S[i], S[j] = S[j], S[i]
                            
                            # Update counts for affected windows
                            for pos in [i, j]:
                                start_w = max(0, pos - n + 1)
                                end_w = min(pos, len(S) - n)
                                for w in range(start_w, end_w + 1):
                                    p = tuple(S[w:w+n])
                                    counts[p] = counts.get(p, 0) + 1
                            
                            # Try to delete after swap
                            for del_pos in [i, j, i+1, j-1]:
                                if 0 <= del_pos < len(S):
                                    if try_remove(S, counts, del_pos, n):
                                        best_len = len(S)
                                        best_S = S[:]
        
        return "".join(map(str, best_S))
    
    return standard_construct(n)
