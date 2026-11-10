import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 12
    m = 40
    full_mask = (1 << n) - 1
    num_masks = 1 << n  # 4096
    
    # Precompute comparability bitsets: for each mask, which other masks is it comparable to
    # comp_bitsets[a] is a bitset where bit j is set if a is comparable to mask j
    comp_bitsets = [0] * num_masks
    for a in range(num_masks):
        bs = 0
        for b in range(num_masks):
            if a == b:
                continue
            if (a & b) == a or (a & b) == b:
                bs |= (1 << b)
        comp_bitsets[a] = bs
    
    # Precompute popcount for fast counting
    # Use a lookup table for 16-bit chunks
    popcount_table = [0] * 65536
    for i in range(65536):
        popcount_table[i] = bin(i).count('1')
    
    def popcount(x):
        if x == 0:
            return 0
        c = 0
        while x:
            c += 1
            x &= x - 1
        return c
    
    # Better: use the table
    def popcount_fast(x):
        c = 0
        while x:
            c += popcount_table[x & 0xFFFF]
            x >>= 16
        return c
    
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b
    
    def calc_score(masks):
        s = 0
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                if is_comparable(masks[i], masks[j]):
                    s += 1
        return s
    
    def greedy_degree_init(seed_val):
        """Initialize using degree-based greedy: start with empty set, then add sets with highest degree to existing."""
        rng = random.Random(seed_val)
        
        # Start with empty set (comparable to everything)
        current_masks = [0]
        used = set(current_masks)
        
        # Precompute degree for each candidate mask relative to current set
        # degree(mask) = number of masks in current_masks that are comparable to mask
        
        for _ in range(m - 1):
            best_mask = -1
            best_deg = -1
            
            # Compute degrees for all unused masks
            # For each unused mask, count how many current masks it's comparable to
            for mask in range(num_masks):
                if mask in used:
                    continue
                deg = 0
                for cm in current_masks:
                    if is_comparable(mask, cm):
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_mask = mask
            
            if best_mask == -1:
                break
            current_masks.append(best_mask)
            used.add(best_mask)
        
        return current_masks
    
    def simulated_annealing(current_masks, cur_score, used_masks, seed_val, time_limit):
        """Simulated annealing with fast delta calculation using precomputed bitsets."""
        rng = random.Random(seed_val)
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        best_score = cur_score
        best_solution = list(current_masks)
        
        max_steps = 100000
        steps = 0
        
        while steps < max_steps:
            if time.time() > time_limit:
                break
            
            # Pick random index
            idx = rng.randint(0, m - 1)
            old_mask = current_masks[idx]
            
            # Generate neighbor
            new_mask = old_mask
            op = rng.random()
            
            # Get set and unset bits
            ones = [i for i in range(n) if (old_mask >> i) & 1]
            zeros = [i for i in range(n) if not ((old_mask >> i) & 1)]
            
            if op < 0.30:
                # Add bit
                if zeros:
                    bit = rng.choice(zeros)
                    new_mask = old_mask | (1 << bit)
            elif op < 0.60:
                # Remove bit
                if ones:
                    bit = rng.choice(ones)
                    new_mask = old_mask ^ (1 << bit)
            elif op < 0.80:
                # Swap bit
                if ones and zeros:
                    bo = rng.choice(ones)
                    bz = rng.choice(zeros)
                    new_mask = (old_mask ^ (1 << bo)) | (1 << bz)
            elif op < 0.90:
                # Random nearby mask (Hamming distance 1-3)
                    new_mask = old_mask
                    for _ in range(rng.randint(1, 3)):
                        if rng.random() < 0.5 and zeros:
                            bit = rng.choice(zeros)
                            new_mask |= (1 << bit)
                            zeros = [i for i in range(n) if not ((new_mask >> i) & 1)]
                            ones = [i for i in range(n) if (new_mask >> i) & 1]
                        elif ones:
                            bit = rng.choice(ones)
                            new_mask ^= (1 << bit)
                            zeros = [i for i in range(n) if not ((new_mask >> i) & 1)]
                            ones = [i for i in range(n) if (new_mask >> i) & 1]
            else:
                # Random jump
                new_mask = rng.randint(0, full_mask)
            
            if new_mask == old_mask:
                continue
            if new_mask in used_masks:
                continue
            
            # Fast delta calculation using precomputed comparability bitsets
            # Number of comparables for old_mask with other masks
            # We need to count how many of the other m-1 masks are comparable to old_mask vs new_mask
            
            # Build a bitset of which positions (0..m-1) hold which masks
            # Actually, we can just iterate - but let's use the precomputed bitsets
            
            # For old_mask: count how many current masks (excluding idx) are comparable
            # comp_bitsets[old_mask] has bits set for all masks comparable to old_mask
            # We need to count how many of current_masks[j] for j != idx have their bit set in comp_bitsets[old_mask]
            
            # This is still O(m) per evaluation. Let's just do it directly but efficiently.
            
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                mask_j = current_masks[j]
                old_comp = is_comparable(old_mask, mask_j)
                new_comp = is_comparable(new_mask, mask_j)
                if new_comp and not old_comp:
                    delta += 1
                elif old_comp and not new_comp:
                    delta -= 1
            
            # Accept if delta > 0 or with probability exp(delta/T) if delta < 0
            if delta > 0 or (T > T_min and rng.random() < pow(2.718281828, delta / T)):
                used_masks.remove(old_mask)
                used_masks.add(new_mask)
                current_masks[idx] = new_mask
                cur_score += delta
                
                if cur_score > best_score:
                    best_score = cur_score
                    best_solution = list(current_masks)
            
            T *= cooling
            steps += 1
        
        return best_solution, best_score
    
    best_global_score = -1
    best_global_solution = None
    
    random.seed(42)
    
    # Try multiple initializations
    num_restarts = 12
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Greedy degree-based initialization
        current_masks = greedy_degree_init(42 + r)
        used_masks = set(current_masks)
        
        cur_score = calc_score(current_masks)
        
        if cur_score > best_global_score:
            best_global_score = cur_score
            best_global_solution = list(current_masks)
        
        # Simulated annealing polish
        time_limit = deadline - 3.0
        best_solution, best_score = simulated_annealing(
            current_masks, cur_score, used_masks, 1000 + r, time_limit
        )
        
        if best_score > best_global_score:
            best_global_score = best_score
            best_global_solution = best_solution
    
    # Final local search polish on the best solution
    if best_global_solution is not None and time.time() < deadline - 2.0:
        current_masks = list(best_global_solution)
        used_masks = set(current_masks)
        cur_score = best_global_score
        
        # Simple hill climbing
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for idx in range(m):
                old_mask = current_masks[idx]
                # Try all possible single-bit changes
                for bit in range(n):
                    if (old_mask >> bit) & 1:
                        new_mask = old_mask ^ (1 << bit)
                    else:
                        new_mask = old_mask | (1 << bit)
                    
                    if new_mask == old_mask or new_mask in used_masks:
                        continue
                    
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        mask_j = current_masks[j]
                        old_comp = is_comparable(old_mask, mask_j)
                        new_comp = is_comparable(new_mask, mask_j)
                        if new_comp and not old_comp:
                            delta += 1
                        elif old_comp and not new_comp:
                            delta -= 1
                    
                    if delta > 0:
                        used_masks.remove(old_mask)
                        used_masks.add(new_mask)
                        current_masks[idx] = new_mask
                        cur_score += delta
                        improved = True
                        if cur_score > best_global_score:
                            best_global_score = cur_score
                            best_global_solution = list(current_masks)
                        break  # Restart scan after improvement
    
    # Convert best solution to list of lists
    result = []
    for mask in best_global_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
    
    return result
