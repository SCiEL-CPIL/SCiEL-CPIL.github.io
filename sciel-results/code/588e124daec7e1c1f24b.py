import time
import random

def _score_bitset(B):
    count = 0
    for d in range(1, 257):
        if (B & (B >> d) & (B >> (2*d))) != 0:
            count += 1
    return count

def _bits_to_set(B):
    return [i for i in range(512) if (B >> i) & 1]

def _greedy_construct(seed):
    random.seed(seed)
    B = 0
    # realized_diffs[d] = True if d is realized
    realized = [False] * 257 
    
    # Helper to check if adding x creates new diffs
    def get_new_diffs(x, current_B, current_realized):
        new_count = 0
        # x can be a, a+d, or a+2d
        # 1. x is a: need x+d, x+2d in B
        # 2. x is a+d: need x-d, x+d in B
        # 3. x is a+2d: need x-d, x-2d in B
        
        # We iterate d from 1 to 255
        # To be fast, we can just check the conditions.
        # But we only care about d NOT in current_realized.
        
        # Actually, it's faster to just compute the score of (B | (1<<x)) and subtract score(B)?
        # No, that's O(256) per candidate. 512 candidates * 256 = 131k ops per step.
        # 32 steps * 131k = 4M ops. This is fast enough.
        
        # Let's just compute the marginal gain directly.
        # A diff d is NEW if it wasn't realized before, but IS realized after adding x.
        # It is realized after adding x if any of the triplets involving x satisfy the condition.
        
        # Triplet 1: x, x+d, x+2d. Requires (B >> (x+d)) & 1 and (B >> (x+2d)) & 1.
        # Triplet 2: x-d, x, x+d. Requires (B >> (x-d)) & 1 and (B >> (x+d)) & 1.
        # Triplet 3: x-2d, x-d, x. Requires (B >> (x-2d)) & 1 and (B >> (x-d)) & 1.
        
        # We check d from 1 to 255.
        for d in range(1, 256):
            if current_realized[d]:
                continue
            
            # Check if d is realized by x
            realized_by_x = False
            
            # Case 1: x is start
            if x + 2*d < 513:
                if ((current_B >> (x+d)) & 1) and ((current_B >> (x+2*d)) & 1):
                    realized_by_x = True
            
            # Case 2: x is middle
            if x - d >= 1:
                if ((current_B >> (x-d)) & 1) and ((current_B >> (x+d)) & 1):
                    realized_by_x = True
            
            # Case 3: x is end
            if x - 2*d >= 1:
                if ((current_B >> (x-2*d)) & 1) and ((current_B >> (x-d)) & 1):
                    realized_by_x = True
            
            if realized_by_x:
                new_count += 1
                
        return new_count

    for _ in range(32):
        best_x = -1
        best_gain = -1
        # Iterate through all possible x
        # To allow randomness in ties, we can collect all max gains
        max_gain = -1
        candidates = []
        
        for x in range(1, 513):
            if (B >> x) & 1:
                continue
            gain = get_new_diffs(x, B, realized)
            if gain > max_gain:
                max_gain = gain
                candidates = [x]
            elif gain == max_gain:
                candidates.append(x)
        
        if not candidates:
            break
            
        # Pick randomly from candidates to break ties
        x = random.choice(candidates)
        B |= (1 << x)
        
        # Update realized diffs
        # We need to re-check which diffs are now realized.
        # Simpler: Just re-run the check for all d? No, that's O(256) per step, which is fine.
        # Or just mark the ones that became true.
        # Since we calculated gain, we know which ones *could* have become true, but not exactly which.
        # It's safer and fast enough (32 * 256 = 8k ops) to just re-evaluate the whole realized array.
        for d in range(1, 256):
            if (B & (B >> d) & (B >> (2*d))) != 0:
                realized[d] = True
                
    return B

def _local_search_fast(B, deadline):
    best_B = B
    best_score = _score_bitset(B)
    
    # Simple hill climbing with random swaps
    # Limit iterations to ensure speed
    max_iters = 5000
    
    while time.time() < deadline and max_iters > 0:
        max_iters -= 1
        bits = _bits_to_set(best_B)
        x = random.choice(bits)
        # Pick a random y not in set
        y = random.randint(1, 512)
        while (best_B >> y) & 1:
            y = random.randint(1, 512)
            
        new_B = (best_B & ~(1 << x)) | (1 << y)
        new_score = _score_bitset(new_B)
        
        if new_score > best_score:
            best_score = new_score
            best_B = new_B
            
    return best_B, best_score

def construct():
    start_time = time.time()
    deadline = time.time() + 100.0 # Safety margin
    
    best_global_B = 0
    best_global_score = 0
    
    # Run multiple greedy constructions with different seeds
    # Greedy is fast, so we can run many.
    # Each greedy is ~0.05s. Local search is ~0.5s if limited.
    
    # 1. Deterministic Greedy (Seed 0)
    B_greedy = _greedy_construct(0)
    score_greedy = _score_bitset(B_greedy)
    
    if score_greedy > best_global_score:
        best_global_score = score_greedy
        best_global_B = B_greedy
        
    # 2. Randomized Greedy + Local Search
    # We have ~90s left.
    # Each iteration: Greedy (0.05s) + Local Search (0.5s) = 0.55s.
    # We can do ~150 iterations.
    
    while time.time() < deadline - 5.0:
        seed = random.randint(1, 100000)
        B_curr = _greedy_construct(seed)
        
        # Quick local search polish
        B_curr, score_curr = _local_search_fast(B_curr, deadline - 1.0)
        
        if score_curr > best_global_score:
            best_global_score = score_curr
            best_global_B = B_curr
            
    # Final check
    result = _bits_to_set(best_global_B)
    return result
