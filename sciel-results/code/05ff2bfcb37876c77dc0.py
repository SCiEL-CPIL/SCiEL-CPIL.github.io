"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    """Return shortest superpermutation found for n symbols within time budget."""
    if n <= 2:
        return "121" if n == 2 else "1"
    
    start_time = time.time()
    TIME_LIMIT = 135  # Safety margin before 150s limit
    
    # For small n, use exact known solutions
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321342132431213243214321"
    
    # Build initial superpermutation using recursive construction
    word = build_initial_superperm(n)
    
    # For n >= 5, try to shorten using structural compression
    if n >= 5 and time.time() - start_time < TIME_LIMIT:
        word = compress_superperm(word, n, start_time, TIME_LIMIT)
    
    return "".join(map(str, word))

def build_initial_superperm(n):
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def is_valid_superperm(word, n):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == len(list(permutations(range(1, n + 1))))

def get_permutations_in_word(word, n):
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return seen

def perm_to_code(perm, n):
    """Convert a permutation tuple to an integer code for fast hashing."""
    code = 0
    for p in perm:
        code = code * n + (p - 1)
    return code

def compress_superperm(word, n, start_time, time_limit):
    """Compress a superpermutation using efficient bitmask tracking."""
    current = list(word)
    L = len(current)
    total_perms = n ** n  # Max possible codes, but we only care about valid perms
    
    # Map valid permutations to bit positions
    valid_perm_codes = set()
    for perm in permutations(range(1, n + 1)):
        valid_perm_codes.add(perm_to_code(perm, n))
    
    # Use a set to track which perm codes are covered
    def compute_covered(w):
        """Compute set of perm codes covered by word w."""
        covered = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                code = perm_to_code(window, n)
                if code in valid_perm_codes:
                    covered.add(code)
        return covered
    
    # Check if all perms are covered
    def is_complete(covered):
        return len(covered) == len(valid_perm_codes)
    
    # Initial coverage
    covered = compute_covered(current)
    
    # Phase 1: Greedy single-symbol removal from right to left
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for pos in range(len(current) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            if len(current) - 1 < n:
                continue
            
            # Compute what perms are lost and gained by removing position pos
            # Perms that include position pos: windows starting from max(0, pos-n+1) to min(pos, len-w-n)
            lost = set()
            gained = set()
            
            # Windows that contain position pos
            for i in range(max(0, pos - n + 1), min(pos + 1, len(current) - n + 1)):
                window = tuple(current[i:i + n])
                if len(set(window)) == n:
                    code = perm_to_code(window, n)
                    if code in valid_perm_codes:
                        lost.add(code)
            
            # After removal, new windows form
            trial = current[:pos] + current[pos + 1:]
            for i in range(max(0, pos - n + 1), min(pos, len(trial) - n + 1)):
                window = tuple(trial[i:i + n])
                if len(set(window)) == n:
                    code = perm_to_code(window, n)
                    if code in valid_perm_codes:
                        gained.add(code)
            
            # Check if all lost perms are still covered
            new_covered = (covered - lost) | gained
            if is_complete(new_covered):
                current = trial
                covered = new_covered
                improved = True
                break
    
    # Phase 2: Try to remove chunks between repeated (n-1)-perms
    if time.time() - start_time < time_limit:
        # Build index of (n-1)-perms
        n1_index = {}  # perm code -> list of positions
        for i in range(len(current) - n + 2):
            window = tuple(current[i:i + n - 1])
            if len(set(window)) == n - 1:
                code = perm_to_code(window, n - 1)
                if code not in n1_index:
                    n1_index[code] = []
                n1_index[code].append(i)
        
        # Sort repeated perms by frequency
        repeated_perms = [(code, positions) for code, positions in n1_index.items() if len(positions) > 1]
        repeated_perms.sort(key=lambda x: -len(x[1]))
        
        attempts = 0
        max_attempts = 2000
        
        for code, positions in repeated_perms[:100]:
            if time.time() - start_time > time_limit or attempts >= max_attempts:
                break
            
            for i in range(len(positions) - 1):
                if time.time() - start_time > time_limit or attempts >= max_attempts:
                    break
                
                pos1 = positions[i]
                pos2 = positions[i + 1]
                
                remove_start = pos1 + n - 1
                remove_end = pos2
                
                if remove_end <= remove_start:
                    continue
                
                chunk_len = remove_end - remove_start
                if chunk_len <= 0 or chunk_len > 20:  # Limit chunk size
                    continue
                
                # Try removing this chunk
                trial = current[:remove_start] + current[remove_end:]
                if len(trial) < n:
                    continue
                
                trial_covered = compute_covered(trial)
                if is_complete(trial_covered):
                    current = trial
                    covered = trial_covered
                    attempts += 1
                    # Rebuild index
                    n1_index = {}
                    for i in range(len(current) - n + 2):
                        window = tuple(current[i:i + n - 1])
                        if len(set(window)) == n - 1:
                            c = perm_to_code(window, n - 1)
                            if c not in n1_index:
                                n1_index[c] = []
                            n1_index[c].append(i)
                    repeated_perms = [(c, positions) for c, positions in n1_index.items() if len(positions) > 1]
                    repeated_perms.sort(key=lambda x: -len(x[1]))
                    break
    
    # Phase 3: More aggressive single-symbol removal
    if time.time() - start_time < time_limit:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for pos in range(len(current) - 1, -1, -1):
                if time.time() - start_time > time_limit:
                    break
                if len(current) - 1 < n:
                    continue
                
                lost = set()
                gained = set()
                
                for i in range(max(0, pos - n + 1), min(pos + 1, len(current) - n + 1)):
                    window = tuple(current[i:i + n])
                    if len(set(window)) == n:
                        code = perm_to_code(window, n)
                        if code in valid_perm_codes:
                            lost.add(code)
                
                trial = current[:pos] + current[pos + 1:]
                for i in range(max(0, pos - n + 1), min(pos, len(trial) - n + 1)):
                    window = tuple(trial[i:i + n])
                    if len(set(window)) == n:
                        code = perm_to_code(window, n)
                        if code in valid_perm_codes:
                            gained.add(code)
                
                new_covered = (covered - lost) | gained
                if is_complete(new_covered):
                    current = trial
                    covered = new_covered
                    improved = True
                    break
    
    return current
# EVOLVE-BLOCK-END
