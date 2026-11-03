import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for overhead
    
    all_perms = list(permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    
    # Precompute suffix maps
    suffix_map = [dict() for _ in range(n)]
    for p in all_perms:
        for k in range(1, n):
            s = p[n-k:]
            if s not in suffix_map[k]:
                suffix_map[k][s] = []
            suffix_map[k][s].append(p)
            
    best_word = None
    best_len = float('inf')
    
    # Run multiple strategies
    # Strategy 1: Lexicographic
    # Strategy 2: Reverse Lexicographic
    # Strategy 3: Randomized (multiple runs)
    
    def run_greedy(randomize=False, reverse=False):
        seen = set()
        current_word = list(all_perms[0])
        seen.add(all_perms[0])
        unvisited = set(all_perms[1:])
        
        while len(seen) < total_perms:
            if time.time() > deadline:
                return None
            found = False
            for k in range(n - 1, 0, -1):
                if len(current_word) < k:
                    continue
                suffix = tuple(current_word[-k:])
                if suffix in suffix_map[k]:
                    candidates = suffix_map[k][suffix]
                    # Filter to unvisited
                    valid_cands = [p for p in candidates if p in unvisited]
                    if valid_cands:
                        if randomize:
                            p = random.choice(valid_cands)
                        elif reverse:
                            p = valid_cands[-1]
                        else:
                            p = valid_cands[0]
                            
                        new_part = list(p[:n-k])
                        current_word.extend(new_part)
                        seen.add(p)
                        unvisited.remove(p)
                        
                        old_len = len(current_word) - len(new_part)
                        for i in range(old_len, len(current_word) - n + 1):
                            if i < 0: continue
                            window = tuple(current_word[i:i+n])
                            if window not in seen:
                                seen.add(window)
                                if window in unvisited:
                                    unvisited.remove(window)
                        
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
        return "".join(map(str, current_word))

    # Run lexicographic
    w = run_greedy()
    if w and len(w) < best_len:
        best_len = len(w)
        best_word = w

    # Run reverse lexicographic
    w = run_greedy(reverse=True)
    if w and len(w) < best_len:
        best_len = len(w)
        best_word = w

    # Run randomized several times if time permits
    attempts = 0
    while time.time() < deadline:
        w = run_greedy(randomize=True)
        if w and len(w) < best_len:
            best_len = len(w)
            best_word = w
        attempts += 1
        # If we've done many attempts and no improvement, maybe stop?
        # Or just keep going.
        
    if best_word is None:
        # Fallback to seed construction if greedy failed
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
        return "".join(map(str, word))
        
    return best_word
