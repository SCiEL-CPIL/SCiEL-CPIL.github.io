import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    N = 32
    MAX_VAL = 512
    
    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m
    
    def mask_to_list(mask):
        return [i for i in range(1, MAX_VAL + 1) if (mask >> i) & 1]
    
    def score_mask(mask):
        """
        Count distinct d > 0 such that there exists a with a, a+d, a+2d all in set.
        Using bitmask: check if (mask & (mask >> d) & (mask >> (2*d))) != 0
        """
        valid_diffs = 0
        # d can be at most (MAX_VAL-1)//2 = 255.5, so up to 255
        for d in range(1, 256):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                valid_diffs += 1
        return valid_diffs

    def score_list(lst):
        return score_mask(list_to_mask(lst))

    # Strategy: Deterministic Multi-Start Local Search
    # 1. Generate diverse structured initial populations
    # 2. For each, perform greedy local search (swap-based) until convergence
    # 3. Keep the best result found across all starts

    def generate_structured_starts(num_starts=20):
        starts = []
        
        # 1. Dense blocks
        starts.append(list(range(1, N + 1)))
        starts.append(list(range(MAX_VAL - N + 1, MAX_VAL + 1)))
        
        # 2. Grids (R x C)
        # Try different grid dimensions
        for r in range(2, 10):
            c = N // r
            if r * c < N:
                continue
            # Place in grid
            base = 1
            grid = []
            for i in range(N):
                row = i // c
                col = i % c
                val = base + row * 64 + col * 16 # Spacing to avoid collisions initially
                if 1 <= val <= MAX_VAL:
                    grid.append(val)
            if len(grid) == N:
                starts.append(grid)
                
        # 3. Modular arithmetic patterns
        # Example: x mod p == r
        for p in range(3, 20):
            for r in range(p):
                candidates = [x for x in range(1, MAX_VAL + 1) if x % p == r]
                if len(candidates) >= N:
                    starts.append(candidates[:N])
                    
        # 4. Random starts
        random.seed(42)
        for _ in range(10):
            starts.append(random.sample(range(1, MAX_VAL + 1), N))
            
        return starts

    def local_search(initial_lst, time_limit):
        """
        Greedy local search: try all single swaps (i, j) and single moves (i -> k).
        Accept if score improves. Stop if no improvement or time limit reached.
        """
        current_lst = initial_lst[:]
        current_mask = list_to_mask(current_lst)
        current_score = score_mask(current_mask)
        
        # Pre-compute set for fast lookup
        current_set = set(current_lst)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_new_score = current_score
            best_new_lst = None
            best_new_mask = None
            best_move = None
            
            # Try all swaps (i, j)
            # We can optimize by only checking moves that might change the score.
            # However, N=32, so 32*31/2 = 496 swaps.
            # For each swap, we recalculate score. 496 * 256 operations ~ 126k ops.
            # This is very fast.
            
            n = len(current_lst)
            for i in range(n):
                vi = current_lst[i]
                for j in range(i + 1, n):
                    vj = current_lst[j]
                    
                    # Swap vi and vj
                    # To speed up, we can just create a new list and mask
                    new_lst = current_lst[:]
                    new_lst[i] = vj
                    new_lst[j] = vi
                    new_mask = list_to_mask(new_lst)
                    new_score = score_mask(new_mask)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_lst = new_lst
                        best_new_mask = new_mask
                        best_move = ('swap', i, j)
                        improved = True
                        
            # Try single moves (replace vi with random k)
            # This is more expensive to enumerate all k (512 options * 32 = 16k).
            # We can do a limited random search for moves if swaps don't improve.
            if not improved:
                # Try 100 random single replacements
                for _ in range(100):
                    i = random.randint(0, n - 1)
                    vi = current_lst[i]
                    # Pick a random new value not in set
                    while True:
                        k = random.randint(1, MAX_VAL)
                        if k not in current_set:
                            break
                    
                    new_lst = current_lst[:]
                    new_lst[i] = k
                    new_mask = list_to_mask(new_lst)
                    new_score = score_mask(new_mask)
                    
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_new_lst = new_lst
                        best_new_mask = new_mask
                        best_move = ('move', i, k)
                        improved = True
            
            if improved:
                current_lst = best_new_lst
                current_mask = best_new_mask
                current_score = best_new_score
                current_set = set(current_lst)
        
        return current_lst, current_score

    # Main loop
    starts = generate_structured_starts(20)
    
    # Sort starts by initial score to prioritize good ones?
    # Or just iterate.
    
    best_overall_score = 0
    best_overall_lst = None
    
    # Allocate time per start. 
    # If we have 20 starts and 115s, we can give ~5s to each.
    # But local search converges fast usually.
    # Let's do a dynamic allocation: give 1s to first 10, 0.5s to next 10.
    
    for idx, start_lst in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
            
        # Give more time to earlier, higher-quality starts?
        # Let's just give a fixed budget per start, decreasing slightly.
        time_per_start = max(0.5, 5.0 - idx * 0.1)
        local_deadline = time.time() + time_per_start
        
        lst, score = local_search(start_lst, local_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_lst = lst[:]
            
        # Check if we have a very high score already
        if best_overall_score >= 50: # Just a heuristic, max possible is ~255
             # We can stop early if we hit a high bar, but let's use time.
             pass

    # If time remains, do a final polish on the best found
    if time.time() < deadline - 2.0 and best_overall_lst:
        final_deadline = deadline - 1.0
        best_overall_lst, best_overall_score = local_search(best_overall_lst, final_deadline)

    return best_overall_lst

# Fallback if construct fails or returns invalid
if __name__ == "__main__":
    res = construct()
    assert len(res) == 32
    assert len(set(res)) == 32
    assert all(1 <= x <= 512 for x in res)
    print("Score:", score_list(res))
