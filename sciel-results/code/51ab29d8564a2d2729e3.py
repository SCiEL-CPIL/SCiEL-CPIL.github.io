import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 110s hard limit for internal logic to allow for return overhead

    # 1. Generate the naive superpermutation using classic recursive construction
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
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_word = word[:]
    n_perms = int(n)
    total_perms = 1
    for i in range(2, n + 1):
        total_perms *= i
        
    # Helper to check validity
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        for i in range(len(w) - n + 1):
            seen.add(tuple(w[i:i+n]))
        return len(seen) == total_perms

    current_len = len(best_word)
    
    # Initial check
    if not is_valid(best_word):
        return "".join(map(str, best_word))

    # Frequency-balanced constructive algorithm
    # Track symbol frequencies in the current word
    def get_freq(w):
        freq = [0] * (n + 1)
        for c in w:
            freq[c] += 1
        return freq
    
    # Greedy construction with frequency balancing
    def greedy_construct():
        """Build superpermutation greedily, prioritizing high-overlap moves with balanced frequencies."""
        # Start with a base permutation
        base = list(range(1, n + 1))
        w = base[:]
        seen_perms = {tuple(base)}
        
        # Track all permutations we need to include
        all_perms = set(itertools.permutations(range(1, n + 1)))
        remaining = all_perms - seen_perms
        
        # Track frequency of each symbol
        freq = get_freq(w)
        
        # We'll greedily extend the word by adding one symbol at a time
        # Choose the symbol that creates the most new permutations while keeping freq balanced
        max_iter = len(remaining) + n  # Upper bound on extensions needed
        for _ in range(max_iter):
            if not remaining:
                break
            
            # Try each possible next symbol
            best_sym = None
            best_score = -1
            best_new_perms = 0
            
            for sym in range(1, n + 1):
                candidate = w + [sym]
                
                # Check if this creates any new permutations
                new_window = tuple(candidate[-n:])
                new_perms_count = 0
                if len(candidate) >= n:
                    if new_window in remaining:
                        new_perms_count = 1
                
                # Score: prefer symbols that create new perms, then balance frequency
                # Penalize symbols that are already over-represented
                freq_penalty = freq[sym] - (sum(freq[1:]) / n)
                score = new_perms_count * 100 - freq_penalty
                
                if score > best_score or (score == best_score and new_perms_count > best_new_perms):
                    best_score = score
                    best_sym = sym
                    best_new_perms = new_perms_count
            
            if best_sym is None:
                break
            
            # Add the symbol
            w.append(best_sym)
            freq[best_sym] += 1
            
            # Update remaining if we created a new permutation
            if len(w) >= n:
                new_window = tuple(w[-n:])
                remaining.discard(new_window)
        
        return w
    
    # Try the greedy construction
    greedy_word = greedy_construct()
    if is_valid(greedy_word):
        if len(greedy_word) < current_len:
            best_word = greedy_word
            current_len = len(best_word)
    
    # Loop with improved local search
    while time.time() < deadline:
        op = random.random()
        candidate = best_word[:]
        
        if op < 0.4:
            # Delete - try multiple positions to find a valid deletion
            if len(candidate) <= n:
                continue
            # Try random deletions, keep the first valid one
            found = False
            for _ in range(5):
                test = best_word[:]
                idx = random.randint(0, len(test) - 1)
                del test[idx]
                if is_valid(test):
                    candidate = test
                    found = True
                    break
            if not found:
                continue
        elif op < 0.65:
            # Swap adjacent
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 2)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
        elif op < 0.8:
            # Move
            if len(candidate) < 2:
                continue
            idx = random.randint(0, len(candidate) - 1)
            new_idx = random.randint(0, len(candidate) - 1)
            if idx == new_idx:
                continue
            val = candidate.pop(idx)
            candidate.insert(new_idx, val)
        else:
            # 3-opt: remove a segment and reinsert elsewhere
            if len(candidate) < 4:
                continue
            i = random.randint(0, len(candidate) - 3)
            j = random.randint(i + 1, len(candidate) - 1)
            segment = candidate[i:j+1]
            del candidate[i:j+1]
            k = random.randint(0, len(candidate))
            candidate[k:k] = segment
            
        if is_valid(candidate):
            if len(candidate) < current_len:
                best_word = candidate
                current_len = len(best_word)
            elif len(candidate) == current_len and random.random() < 0.1:
                # Occasionally accept same-length to escape local optima
                best_word = candidate
    
    return "".join(map(str, best_word))
