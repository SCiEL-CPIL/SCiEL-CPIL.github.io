import time
import random
import itertools

def construct(n):
    # Generate standard superpermutation
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    if n == 1:
        return "1"
        
    if n <= 6:
        return "".join(map(str, word))
        
    start_time = time.time()
    deadline = start_time + 110.0
    
    all_perms = list(itertools.permutations(range(1, n+1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    counts = [0] * num_perms
    
    def get_perm_id(window):
        return perm_to_id.get(tuple(window), -1)
        
    for i in range(len(word) - n + 1):
        pid = get_perm_id(word[i:i+n])
        if pid != -1:
            counts[pid] += 1
            
    best_word = word[:]
    best_len = len(word)
    current_word = word[:]
    current_counts = counts[:]
    current_len = len(word)
    
    while time.time() < deadline:
        improved = False
        # Try to find safe removals
        # Sample indices
        for _ in range(500):
            i = random.randint(0, len(current_word) - 1)
            safe = True
            # Check windows containing i
            for k in range(max(0, i - n + 1), min(i + 1, len(current_word) - n + 1)):
                pid = get_perm_id(current_word[k:k+n])
                if pid != -1 and current_counts[pid] <= 1:
                    safe = False
                    break
            if safe:
                # Perform removal
                # Update counts
                # Decrement old windows
                for k in range(max(0, i - n + 1), min(i + 1, len(current_word) - n + 1)):
                    pid = get_perm_id(current_word[k:k+n])
                    if pid != -1:
                        current_counts[pid] -= 1
                        
                # Remove
                current_word.pop(i)
                current_len -= 1
                
                # Increment new windows
                # New windows at positions k such that k <= i-1 and k >= i-n+1
                # Note: indices shifted.
                for k in range(max(0, i - n + 1), min(i, len(current_word) - n + 1)):
                    pid = get_perm_id(current_word[k:k+n])
                    if pid != -1:
                        current_counts[pid] += 1
                        
                if current_len < best_len:
                    best_len = current_len
                    best_word = current_word[:]
                improved = True
                break # Restart search with new state
        
        if not improved:
            # If no removals found, try a few swaps to escape local minimum
            # Simple random swap and check validity via full scan (slow but safe)
            # Actually, just break to avoid wasting time if stuck.
            # Or try a few more random removals.
            pass

    return "".join(map(str, best_word))
