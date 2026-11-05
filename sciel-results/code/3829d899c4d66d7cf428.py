import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 145
    
    # Standard superpermutation construction
    def build_superperm(n):
        if n == 1:
            return [1]
        prev = build_superperm(n - 1)
        # For each permutation of length n-1 in prev, append new symbol and repeat
        # Standard method: for each (n-1)-perm window, add new symbol then repeat the (n-1)-perm
        result = [1]
        for m in range(2, n + 1):
            # Extract all (m-1)-permutations from current word
            perms = []
            seen = set()
            for i in range(len(result) - (m - 1) + 1):
                window = tuple(result[i:i + m - 1])
                if len(set(window)) == m - 1:
                    if window not in seen:
                        seen.add(window)
                        perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find maximum overlap with end of new
                overlap = 0
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            result = new
        return result
    
    word = build_superperm(n)
    
    # Build permutation lookup
    all_perms = set(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def get_counts(w):
        counts = [0] * num_perms
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_idx:
                counts[perm_to_idx[p]] += 1
        return counts
    
    counts = get_counts(word)
    
    def is_valid(counts):
        return all(c > 0 for c in counts)
    
    if not is_valid(counts):
        return "".join(map(str, word))
    
    improved = True
    while improved:
        if time.time() > deadline - 15:
            break
        improved = False
        L = len(word)
        
        # Try deletion from each position
        for i in range(L - n + 1, 0, -1):
            if time.time() > deadline - 10:
                break
            
            # Check if deletion at position i is valid
            # Affected windows: those that include position i
            j_start = max(0, i - n + 1)
            j_end = min(i, L - n)
            
            valid = True
            removed = []
            for j in range(j_start, j_end + 1):
                p = tuple(word[j:j+n])
                idx = perm_to_idx.get(p, -1)
                if idx != -1:
                    if counts[idx] == 1:
                        valid = False
                        break
                    removed.append((j, idx))
            
            if not valid:
                continue
            
            # Perform deletion
            word.pop(i)
            improved = True
            L -= 1
            
            # Update counts for removed windows
            for j, idx in removed:
                counts[idx] -= 1
            
            # Update counts for new windows (those spanning the gap)
            j_start_new = max(0, i - n + 1)
            j_end_new = min(i - 1, L - n)
            
            for j in range(j_start_new, j_end_new + 1):
                new_window = tuple(word[j:i] + word[i:j+n])
                idx = perm_to_idx.get(new_window, -1)
                if idx != -1:
                    counts[idx] += 1
    
    # Try swap optimization
    if time.time() < deadline - 20:
        L = len(word)
        for i in range(L - 1):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, min(i + 4, L)):
                # Try swapping word[i] and word[j]
                word[i], word[j] = word[j], word[i]
                
                # Recalculate counts (expensive but thorough)
                new_counts = get_counts(word)
                
                if is_valid(new_counts):
                    counts = new_counts
                else:
                    word[i], word[j] = word[j], word[i]
    
    return "".join(map(str, word))
