import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    """
    start_time = time.time()
    deadline = 140.0  # 10 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
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
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
    
    # Pre-compute all permutations and map to integer codes for speed
    all_perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(all_perms)
    
    # Use base-8 encoding for windows
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
        
        # Affected windows: those containing pos or pos+1
        # Window j contains indices j..j+n-1
        # Contains pos if j <= pos <= j+n-1 => pos-n+1 <= j <= pos
        # Contains pos+1 if j <= pos+1 <= j+n-1 => pos+1-n+1 <= j <= pos+1 => pos-n+2 <= j <= pos+1
        # Union: max(0, pos-n+1) to min(pos+1, L-n)
        start_j = max(0, pos - n + 1)
        end_j = min(pos + 1, L - n)
        
        # Check validity
        old_wcodes = {}
        for j in range(start_j, end_j + 1):
            old_wcodes[j] = wcodes[j]
            
        # Simulate swap
        current_word[pos], current_word[pos+1] = current_word[pos+1], current_word[pos]
        
        valid = True
        for j in range(start_j, end_j + 1):
            new_code = window_code(current_word, j)
            old_code = old_wcodes[j]
            
            old_pidx = code_to_pidx.get(old_code, -1)
            new_pidx = code_to_pidx.get(new_code, -1)
            
            # If old was unique and new is different, count drops to 0 -> Invalid
            if old_pidx >= 0 and new_pidx != old_pidx:
                if counts[old_pidx] == 1:
                    valid = False
                    break
            
        # Revert swap if invalid
        if not valid:
            current_word[pos], current_word[pos+1] = current_word[pos+1], current_word[pos]
            return False
            
        # Apply swap
        # Update wcodes and counts
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

    # Main Loop
    no_improve_count = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Backward Greedy
        L = len(current_word)
        for i in range(L - 1, -1, -1):
            if try_delete(i):
                improved = True
                break
        if improved:
            no_improve_count = 0
            continue
        
        # 2. Forward Greedy
        L = len(current_word)
        for i in range(L):
            if try_delete(i):
                improved = True
                break
        if improved:
            no_improve_count = 0
            continue
            
        # 3. Random Deletion
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
        
        # 5. Stuck, try a few more random swaps
        for _ in range(10):
            if time.time() > deadline: break
            L = len(current_word)
            if L > 1:
                pos = random.randint(0, L - 2)
                if try_swap(pos):
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 100:
                # Restart with perturbation? Or just break?
                # For n=7, we might be at 5905.
                break

    return "".join(map(str, current_word))
