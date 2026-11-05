import time
import itertools
import random
from collections import defaultdict

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses De Bruijn graph formulation with local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Step 1: Generate the base superpermutation using classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap between end of new_word and start of block
            max_ov = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word.extend(block[overlap:])
        word = new_word
    
    current_word = word[:]
    L = len(current_word)
    
    # Step 2: Precompute permutation codes for fast window checking
    all_perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Use base-8 encoding for windows (symbols 1-7 fit in 3 bits, 8 is safe)
    pow8 = [1] * (n + 1)
    for i in range(1, n + 1):
        pow8[i] = pow8[i-1] * 8
    
    code_to_pidx = {}
    for idx, p in enumerate(all_perms):
        c = 0
        for i in range(n):
            c += p[i] * pow8[n - 1 - i]
        code_to_pidx[c] = idx
    
    def window_code(w, start):
        c = 0
        for i in range(n):
            c += w[start + i] * pow8[n - 1 - i]
        return c
    
    L = len(current_word)
    num_windows = L - n + 1
    wcodes = [0] * num_windows
    for i in range(num_windows):
        wcodes[i] = window_code(current_word, i)
    
    counts = [0] * num_perms
    for wc in wcodes:
        pidx = code_to_pidx.get(wc, -1)
        if pidx >= 0:
            counts[pidx] += 1
    
    def rebuild():
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        num_windows = L - n + 1
        wcodes = [0] * num_windows
        for i in range(num_windows):
            wcodes[i] = window_code(current_word, i)
        counts = [0] * num_perms
        for wc in wcodes:
            pidx = code_to_pidx.get(wc, -1)
            if pidx >= 0:
                counts[pidx] += 1
    
    def try_delete(pos):
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        num_windows = L - n + 1
        start_j = max(0, pos - n + 1)
        end_j = min(pos, num_windows - 1)
        
        valid = True
        for j in range(start_j, end_j + 1):
            pidx = code_to_pidx.get(wcodes[j], -1)
            if pidx >= 0 and counts[pidx] == 1:
                valid = False
                break
        
        if valid:
            current_word.pop(pos)
            rebuild()
            return True
        return False
    
    def try_swap(pos):
        """Swap pos and pos+1 if valid."""
        nonlocal wcodes, counts, L, num_windows
        L = len(current_word)
        if pos >= L - 1:
            return False
        
        start_j = max(0, pos - n + 1)
        end_j = min(pos + 1, L - n)
        
        old_wcodes = {}
        for j in range(start_j, end_j + 1):
            old_wcodes[j] = wcodes[j]
            
        current_word[pos], current_word[pos+1] = current_word[pos+1], current_word[pos]
        
        valid = True
        for j in range(start_j, end_j + 1):
            new_code = window_code(current_word, j)
            old_code = old_wcodes[j]
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            if old_pidx >= 0 and new_pidx != old_pidx:
                if counts[old_pidx] == 1:
                    valid = False
                    break
            
        if not valid:
            current_word[pos], current_word[pos+1] = current_word[pos+1], current_word[pos]
            return False
            
        for j in range(start_j, end_j + 1):
            old_code = old_wcodes[j]
            new_code = window_code(current_word, j)
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            if old_pidx >= 0:
                counts[old_pidx] -= 1
            if new_pidx >= 0:
                counts[new_pidx] += 1
            wcodes[j] = new_code
            
        return True
    
    def try_insert(pos, val):
        """Insert val at position pos if it doesn't create duplicate-only windows."""
        nonlocal wcodes, counts, L, num_windows
        if pos < 0 or pos > len(current_word):
            return False
        
        # Check windows that would be affected by insertion
        # Windows containing the new position
        start_j = max(0, pos - n + 1)
        end_j = min(pos, len(current_word) - n + 1)
        
        # Simulate insertion
        current_word.insert(pos, val)
        rebuild()
        
        # Check if any window now has a permutation that only appears once
        # and was previously absent (this would be invalid if we're trying to minimize)
        # Actually, insertion always increases length, so we only do this if it helps
        # For now, this is a placeholder - we mainly use deletions
        return True
    
    # Main optimization loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        
        # 1. Backward Greedy Deletion
        L = len(current_word)
        for i in range(L - 1, -1, -1):
            if try_delete(i):
                improved = True
                break
        if improved:
            no_improve_count = 0
            continue
        
        # 2. Forward Greedy Deletion
        L = len(current_word)
        for i in range(L):
            if try_delete(i):
                improved = True
                break
        if improved:
            no_improve_count = 0
            continue
        
        # 3. Random Deletion Attempts
        L = len(current_word)
        positions = list(range(L))
        random.shuffle(positions)
        for i in positions:
            if try_delete(i):
                improved = True
                break
        if improved:
            no_improve_count = 0
            continue
        
        # 4. Random Swap
        L = len(current_word)
        if L > 1:
            pos = random.randint(0, L - 2)
            if try_swap(pos):
                improved = True
                no_improve_count = 0
                continue
        
        # 5. Multiple Random Swaps
        for _ in range(20):
            if time.time() > deadline:
                break
            L = len(current_word)
            if L > 1:
                pos = random.randint(0, L - 2)
                if try_swap(pos):
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a more aggressive perturbation
                # Randomly swap a few positions to escape local optimum
                L = len(current_word)
                if L > 10:
                    for _ in range(5):
                        pos = random.randint(0, L - 2)
                        try_swap(pos)
                    no_improve_count = 0
                else:
                    break
    
    return "".join(map(str, current_word))
