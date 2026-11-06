import random
import time

def construct():
    """
    Returns a list of 32 distinct integers in 1..512 maximizing the number
    of distinct d > 0 such that a, a+d, a+2d are in the set.
    """
    
    def score_bitset(bitset):
        """
        Calculate the number of distinct differences d > 0 
        such that a, a+d, a+2d are in S for some a, given the bitset.
        """
        count = 0
        # d can be at most 255
        for d in range(1, 256):
            if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
                count += 1
        return count

    def get_bitset(S):
        bitset = 0
        for v in S:
            bitset |= (1 << (v - 1))
        return bitset

    # Initial candidates
    candidates = []
    
    # 1. Dense start
    candidates.append(list(range(1, 33)))
    
    # 2. Spread starts
    for offset in [128, 256, 384]:
        candidates.append(list(range(1, 17)) + list(range(offset + 1, offset + 17)))
        
    # 3. Random starts
    random.seed(42)
    for _ in range(10):
        candidates.append(random.sample(range(1, 513), 32))
        
    # 4. Structured "Golomb-like" or dense clusters
    # Try to place elements to maximize small d and some large d
    c_struct = []
    for i in range(16):
        c_struct.append(i * 3 + 1) # 1, 4, 7...
    for i in range(16):
        c_struct.append(250 + i * 3 + 1)
    # Ensure distinct and in range
    c_struct = list(set(c_struct))
    while len(c_struct) < 32:
        v = random.randint(1, 512)
        if v not in c_struct:
            c_struct.append(v)
    candidates.append(c_struct[:32])

    # Score initial candidates
    scored_candidates = []
    for S in candidates:
        b = get_bitset(S)
        s = score_bitset(b)
        scored_candidates.append((s, S, b))
        
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 5 for refinement
    top_starts = scored_candidates[:5]
    
    best_score_global = top_starts[0][0]
    best_set_global = top_starts[0][1][:]
    
    start_time = time.time()
    total_time_limit = 110.0
    time_per_start = total_time_limit / len(top_starts)
    
    for s_score, s_set, s_bitset in top_starts:
        current_set = s_set[:]
        current_bitset = s_bitset
        current_score = s_score
        
        if time.time() - start_time > total_time_limit:
            break
            
        start_sa_time = time.time()
        time_limit_sa = time_per_start
        
        temperature = 2.0
        cooling = 0.99995
        min_temp = 0.01
        
        while time.time() - start_sa_time < time_limit_sa:
            # Choose move type: 70% replace, 30% swap
            if random.random() < 0.7:
                # Replace
                idx = random.randint(0, 31)
                old_val = current_set[idx]
                new_val = random.randint(1, 512)
                
                if new_val in current_set:
                    continue
                
                # Update bitset
                current_bitset ^= (1 << (old_val - 1)) | (1 << (new_val - 1))
                current_set[idx] = new_val
                
                new_score = score_bitset(current_bitset)
                
                delta = new_score - current_score
                if delta > 0:
                    current_score = new_score
                    if new_score > best_score_global:
                        best_score_global = new_score
                        best_set_global = current_set[:]
                else:
                    if temperature > min_temp:
                        # Accept with probability exp(delta / T)
                        # delta is negative integer
                        prob = 2.71828 ** (delta / temperature)
                        if random.random() < prob:
                            current_score = new_score
                        else:
                            # Revert
                            current_set[idx] = old_val
                            current_bitset ^= (1 << (new_val - 1)) | (1 << (old_val - 1))
                    else:
                        # Revert
                        current_set[idx] = old_val
                        current_bitset ^= (1 << (new_val - 1)) | (1 << (old_val - 1))
                        
            else:
                # Swap
                idx1 = random.randint(0, 31)
                idx2 = random.randint(0, 31)
                if idx1 == idx2:
                    continue
                    
                val1 = current_set[idx1]
                val2 = current_set[idx2]
                
                # Swap in set
                current_set[idx1] = val2
                current_set[idx2] = val1
                
                # Bitset remains the same!
                # So score remains the same?
                # Wait, if we swap values, the set of values is identical.
                # So the score is identical.
                # This move is useless for optimization unless we consider the *order* matters?
                # No, the score depends only on the *set* of integers.
                # So swapping two elements in the list does not change the set, and thus does not change the score.
                # Therefore, Swap moves are redundant for this metric unless we are doing something else.
                # I will remove Swap moves as they are no-ops for the score.
                pass

            temperature *= cooling
            
            if temperature < min_temp:
                # Reheat and perturb
                current_set = best_set_global[:]
                current_bitset = get_bitset(current_set)
                current_score = best_score_global
                temperature = 2.0
                # Perturb
                for _ in range(2):
                    idx = random.randint(0, 31)
                    old_val = current_set[idx]
                    new_val = random.randint(1, 512)
                    while new_val in current_set:
                        new_val = random.randint(1, 512)
                    current_set[idx] = new_val
                    current_bitset ^= (1 << (old_val - 1)) | (1 << (new_val - 1))
                current_score = score_bitset(current_bitset)

    return best_set_global
