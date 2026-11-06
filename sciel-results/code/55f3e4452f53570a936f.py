"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill
    
    N = 256
    all_nums = list(range(1, N + 1))
    
    # Precompute conflict_mask: conflict_mask[i] is a bitmask of indices j where i and j conflict
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
    
    # All odds are indices 0,2,4,...,254 (numbers 1,3,5,...,255)
    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]
    
    def popcount(x):
        """Fast population count."""
        x = x - ((x >> 1) & 0x55555555555555555555555555555555)
        x = (x & 0x33333333333333333333333333333333) + ((x >> 2) & 0x33333333333333333333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F
        x = x + (x >> 8)
        x = x + (x >> 16)
        return x & 0x000000000000000000000000000000FF
    
    def greedy_add_evens(start_mask, even_order, deadline):
        """Greedily add evens from even_order that don't conflict with current set."""
        current_mask = start_mask
        for e in even_order:
            if time.time() > deadline:
                break
            if (current_mask >> e) & 1:
                continue
            # Check if e conflicts with any element in current
            if (conflict_mask[e] & current_mask) == 0:
                current_mask |= (1 << e)
        return current_mask
    
    def valid_swap(current_mask, remove_idx, add_idx):
        """Check if removing remove_idx and adding add_idx yields a valid set."""
        if (current_mask >> remove_idx) & 1 == 0:
            return False
        if (current_mask >> add_idx) & 1:
            return False
        # add_idx must not conflict with any element in current_mask except remove_idx
        new_rest = current_mask & ~(1 << remove_idx)
        if (conflict_mask[add_idx] & new_rest) == 0:
            return True
        return False
    
    def local_search(current_mask, deadline, max_iters=200000):
        """Local search with 1-add, 1-swap, and 2-swap moves."""
        best_mask = current_mask
        best_size = popcount(current_mask)
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            cur_size = popcount(current_mask)
            
            if move_type < 0.5:
                # 1-add: try to add a single element
                e = random.randint(0, N - 1)
                if (current_mask >> e) & 1:
                    continue
                if (conflict_mask[e] & current_mask) == 0:
                    current_mask |= (1 << e)
                    sz = popcount(current_mask)
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
                        no_improve = 0
                    else:
                        no_improve += 1
            elif move_type < 0.85:
                # 1-swap: remove one, add one
                # Pick a random element to remove
                c = random.randint(0, N - 1)
                if not ((current_mask >> c) & 1):
                    continue
                # Pick a random element to add
                e = random.randint(0, N - 1)
                if (current_mask >> e) & 1:
                    continue
                if valid_swap(current_mask, c, e):
                    current_mask = (current_mask & ~(1 << c)) | (1 << e)
                    sz = popcount(current_mask)
                    if sz > best_size:
                        best_mask = current_mask
                        best_size = sz
                        no_improve = 0
                    else:
                        no_improve += 1
            else:
                # 2-swap: remove two, add two
                c1 = random.randint(0, N - 1)
                c2 = random.randint(0, N - 1)
                if c1 == c2:
                    continue
                if not ((current_mask >> c1) & 1) or not ((current_mask >> c2) & 1):
                    continue
                e1 = random.randint(0, N - 1)
                e2 = random.randint(0, N - 1)
                if e1 == e2:
                    continue
                if (current_mask >> e1) & 1 or (current_mask >> e2) & 1:
                    continue
                if e1 == c1 or e1 == c2 or e2 == c1 or e2 == c2:
                    continue
                # Check validity
                new_rest = current_mask & ~(1 << c1) & ~(1 << c2)
                if (conflict_mask[e1] & new_rest) == 0 and (conflict_mask[e2] & new_rest) == 0:
                    # Also check e1 and e2 don't conflict with each other
                    if (conflict_mask[e1] & (1 << e2)) == 0 and (conflict_mask[e2] & (1 << e1)) == 0:
                        current_mask = new_rest | (1 << e1) | (1 << e2)
                        sz = popcount(current_mask)
                        if sz > best_size:
                            best_mask = current_mask
                            best_size = sz
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if no_improve > max_no_improve:
                # Reset to best and try harder
                current_mask = best_mask
                no_improve = 0
                max_no_improve = min(max_no_improve * 2, 50000)
        
        return best_mask
    
    # Start with all odds
    base_mask = 0
    for i in odds_indices:
        base_mask |= (1 << i)
    
    best_overall = base_mask
    best_size = popcount(base_mask)
    
    # Strategy 1: Greedy add evens in natural order
    current = greedy_add_evens(base_mask, evens_indices, DEADLINE)
    sz = popcount(current)
    if sz > best_size:
        best_overall = current
        best_size = sz
    
    # Strategy 2: Greedy add evens in reverse order
    current = greedy_add_evens(base_mask, list(reversed(evens_indices)), DEADLINE)
    sz = popcount(current)
    if sz > best_size:
        best_overall = current
        best_size = sz
    
    # Strategy 3: Multiple random orderings
    for attempt in range(30):
        if time.time() > DEADLINE - 5:
            break
        random_order = evens_indices[:]
        random.shuffle(random_order)
        current = greedy_add_evens(base_mask, random_order, DEADLINE)
        sz = popcount(current)
        if sz > best_size:
            best_overall = current
            best_size = sz
    
    # Local search from best
    if time.time() < DEADLINE - 10:
        improved = local_search(best_overall, DEADLINE, max_iters=300000)
        sz = popcount(improved)
        if sz > best_size:
            best_overall = improved
            best_size = sz
    
    # Convert back to numbers
    result = sorted([i + 1 for i in range(N) if (best_overall >> i) & 1])
    
    return result
# EVOLVE-BLOCK-END
