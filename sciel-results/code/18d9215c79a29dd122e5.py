import itertools
import random
import time
import math

def construct(n):
    # Step 1: Generate standard superpermutation
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
    
    if n <= 4:
        return "".join(map(str, word))
    
    # Minimization Phase
    def get_permutation_counts(w, n):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    current_word = word[:]
    counts = get_permutation_counts(current_word, n)
    initial_len = len(current_word)
    
    deadline = time.time() + 135  # Leave 15s margin for 150s limit
    
    def try_removal_pass(w, cnts, n, order, deadline):
        """Try to remove characters in the given order. Returns new word and counts."""
        w = w[:]
        cnts = {k: v for k, v in cnts.items()}
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in order:
                if time.time() > deadline:
                    break
                if i >= len(w):
                    continue
                # Windows affected by removing index i
                j_start = max(0, i - n + 1)
                j_end = min(i, len(w) - n)
                
                valid_removal = True
                affected_perms = []
                for j in range(j_start, j_end + 1):
                    p = tuple(w[j:j+n])
                    if cnts[p] == 1:
                        valid_removal = False
                        break
                    affected_perms.append((j, p))
                
                if valid_removal:
                    for j, p in affected_perms:
                        cnts[p] -= 1
                    w.pop(i)
                    improved = True
                    # After pop, indices shift. Continue with same i (next element is now at i)
                    # But if we're iterating a fixed order, we should skip ahead
                    # Actually, since we popped, the next element is at position i
                    # We'll just continue; the next iteration will check index i again
                    # To avoid infinite loops, we break and restart
                    break
        
        return w, cnts

    best_word = current_word[:]
    best_counts = counts
    
    # Multiple strategies for ordering
    strategies = []
    
    # Strategy 1: Left to right
    strategies.append(lambda w: list(range(len(w))))
    # Strategy 2: Right to left
    strategies.append(lambda w: list(range(len(w) - 1, -1, -1)))
    # Strategy 3: Random
    strategies.append(lambda w: (lambda lst: (random.shuffle(lst), lst)[1])(list(range(len(w)))))
    # Strategy 4: Middle out
    strategies.append(lambda w: (lambda lst: (lst.sort(key=lambda x: abs(x - len(w)//2)), lst)[1])(list(range(len(w)))))
    
    # Run multiple passes with different strategies
    for round_num in range(50):
        if time.time() > deadline:
            break
        
        # Pick a strategy (cycle through them)
        strat = strategies[round_num % len(strategies)]
        order = strat(current_word)
        
        new_word, new_counts = try_removal_pass(current_word, counts, n, order, deadline)
        
        if len(new_word) < len(current_word):
            current_word = new_word
            counts = new_counts
            if len(current_word) < len(best_word):
                best_word = current_word[:]
                best_counts = {k: v for k, v in counts.items()}
    
    # If we have time, do more randomized restarts from best
    if time.time() < deadline - 10:
        for _ in range(20):
            if time.time() > deadline:
                break
            # Start from best and try random removals
            w = best_word[:]
            cnts = {k: v for k, v in best_counts.items()}
            # Try random order
            order = list(range(len(w)))
            random.shuffle(order)
            new_w, new_cnts = try_removal_pass(w, cnts, n, order, deadline)
            if len(new_w) < len(best_word):
                best_word = new_w
                best_counts = new_cnts
    
    return "".join(map(str, best_word))
