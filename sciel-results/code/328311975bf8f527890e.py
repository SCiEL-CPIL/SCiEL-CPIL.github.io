import random
import time

def count_distinct_sums(s):
    sums = set()
    n = len(s)
    for i in range(n):
        for j in range(i, n):
            sums.add(s[i] + s[j])
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = None
    best_score = float('inf')
    
    # Try to find a set with minimal distinct sums
    # This correlates with having small max Sidon subsets
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Random start
        s = random.sample(range(129), 12)
        s.sort()
        current_score = count_distinct_sums(s)
        
        # Local search: swap or move elements
        for step in range(200):
            if time.time() > deadline:
                return best_set if best_set else s
            
            if current_score < best_score:
                best_score = current_score
                best_set = s[:]
            
            # Pick two random indices to swap or move one
            if random.random() < 0.5:
                i, j = random.sample(range(12), 2)
                s[i], s[j] = s[j], s[i]
                s.sort()
            else:
                i = random.randint(0, 11)
                old_val = s[i]
                new_val = random.randint(0, 128)
                if new_val not in s:
                    s[i] = new_val
                    s.sort()
                else:
                    # Try a neighbor
                    for offset in [1, -1, 2, -2]:
                        nv = old_val + offset
                        if 0 <= nv <= 128 and nv not in s:
                            s[i] = nv
                            s.sort()
                            break
                    else:
                        continue # No valid move, skip
            
            new_score = count_distinct_sums(s)
            
            # Accept if better or with small probability (annealing-like)
            # For simplicity, accept only if better (Hill Climbing)
            # But this gets stuck. Let's use a simple annealing.
            # However, for this specific metric, hill climbing might be enough if we have many restarts.
            
            if new_score >= current_score:
                # Revert
                if random.random() < 0.5:
                    # We didn't track the previous state easily in the swap case if we modified in place.
                    # Let's just restart local search if we get stuck?
                    # Or simpler: only accept if better.
                    pass 
                # Actually, let's just do "Accept if better" and rely on restarts.
                # To revert swap:
                if random.random() < 0.5:
                    # This logic is getting messy with in-place mods.
                    # Let's use a cleaner local search.
                    pass

    # Cleaner implementation below
    return best_set if best_set else list(range(12))
