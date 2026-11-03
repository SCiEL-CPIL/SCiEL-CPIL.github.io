import time
from collections import Counter
import random

def construct(n):
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
        
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # 2. Optimization: Randomized Greedy Deletion
    best_seq = list(word)
    best_len = len(best_seq)
    
    # Helper to compute counts
    def get_counts(seq, n):
        counts = Counter()
        for i in range(len(seq) - n + 1):
            counts[tuple(seq[i:i+n])] += 1
        return counts

    # Helper to check if index i is deletable
    def is_deletable(seq, i, n, counts):
        start_min = max(0, i - n + 1)
        start_max = min(len(seq) - n, i)
        for start in range(start_min, start_max + 1):
            if counts[tuple(seq[start:start+n])] == 1:
                return False
        return True

    # Greedy deletion function with randomization
    def greedy_delete(seq, n, deadline, rng):
        seq = list(seq)
        L = len(seq)
        counts = get_counts(seq, n)
        
        # Initial deletable list
        deletable = [i for i in range(L) if is_deletable(seq, i, n, counts)]
        
        while deletable and time.time() < deadline:
            # Pick a candidate
            idx = rng.choice(deletable)
            
            # Perform deletion
            # Update counts for windows containing idx
            start_min = max(0, idx - n + 1)
            start_max = min(L - n, idx)
            for start in range(start_min, start_max + 1):
                w = tuple(seq[start:start+n])
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
            
            seq.pop(idx)
            L -= 1
            
            # Update counts for new windows formed
            new_start_min = max(0, idx - n + 1)
            new_start_max = min(L - n, idx)
            for start in range(new_start_min, new_start_max + 1):
                w = tuple(seq[start:start+n])
                counts[w] += 1
                
            # Rebuild deletable list (cheap for n=7)
            deletable = [i for i in range(L) if is_deletable(seq, i, n, counts)]
            
        return seq

    # Run multiple randomized passes
    # Use a fixed number of tries or time-based
    tries = 0
    while time.time() - start_time < deadline - 5.0:
        rng = random.Random(tries)
        candidate = greedy_delete(word, n, deadline - 2.0, rng)
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_seq = candidate
            # If we reached the theoretical lower bound or a known good bound, we can stop early?
            # For n=7, 5905 is the target.
            if n == 7 and best_len <= 5905:
                break
        tries += 1
        if tries > 50: # Cap on tries to avoid infinite loop if time check fails
            break

    return "".join(map(str, best_seq))
