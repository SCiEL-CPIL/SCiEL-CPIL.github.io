import time
import random

def count_3aps(S):
    S_set = set(S)
    count = 0
    for a in S:
        for d in range(1, 257 - a):
            if a + d in S_set and a + 2 * d in S_set:
                count += 1
    return count

def is_4free(S):
    S_set = set(S)
    for a in S:
        # Max d such that a + 3d <= 256
        max_d = (256 - a) // 3
        for d in range(1, max_d + 1):
            if a + d in S_set and a + 2 * d in S_set and a + 3 * d in S_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Initial construction: Greedy random
    # Try multiple random seeds to find a good starting point
    best_set = set()
    best_score = -1
    
    # Generate a few random initial sets
    for _ in range(10):
        random.seed(int(time.time() * 1000) + random.randint(0, 1000))
        S = set()
        # Add elements in random order if they don't create a 4-AP
        candidates = list(range(1, 257))
        random.shuffle(candidates)
        for x in candidates:
            S.add(x)
            if not is_4free(S):
                S.remove(x)
        
        score = count_3aps(S)
        if score > best_score:
            best_score = score
            best_set = set(S)
            
    # Local Search: Swap based
    S = best_set
    current_score = best_score
    no_improvement_count = 0
    
    while time.time() < deadline:
        improved = False
        # Shuffle elements to avoid bias
        S_list = list(S)
        random.shuffle(S_list)
        
        for u in S_list:
            if time.time() > deadline:
                break
            # Try swapping u with some v not in S
            # Pick a few random v's to try
            # Or iterate all v? 256 swaps per u, 50 u's = 12800 checks per pass.
            # Each check is is_4free (O(N^2)) + count_3aps (O(N^2)).
            # Might be slow in Python if done naively many times.
            
            # Optimization: Only try a subset of swaps or use incremental updates.
            # For now, try a few random swaps.
            for _ in range(5):
                v = random.randint(1, 256)
                if v in S:
                    continue
                
                S_new = set(S)
                S_new.remove(u)
                S_new.add(v)
                
                if not is_4free(S_new):
                    continue
                
                new_score = count_3aps(S_new)
                if new_score > current_score:
                    S = S_new
                    current_score = new_score
                    improved = True
                    break # Restart u loop with new S? Or just continue?
                    # Let's break inner loop and continue outer, effectively restarting search from new state
                elif new_score == current_score:
                    # Accept neutral moves to escape local optima?
                    # Maybe with low probability
                    if random.random() < 0.1:
                        S = S_new
                        improved = True
                        break
            
            if improved:
                no_improvement_count = 0
                break
        else:
            no_improvement_count += 1
            if no_improvement_count > 10:
                # Perturb: remove a random element and re-add greedily?
                # Or just break
                pass
        
    return sorted(S)
