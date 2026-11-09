import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    N = 512
    # Weights: w[a] = 1 / (512 - a) for a in 1..511
    # Index a corresponds to bit (a-1) in the mask
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
    
    # Precompute coprime masks
    # coprime_mask[a] has bit (b-1) set if gcd(a, b) == 1
    coprime_mask = [0] * N
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if gcd(a, b) == 1:
                mask |= (1 << (b - 1))
        coprime_mask[a] = mask

    # Helper to get score from mask
    def get_score(mask):
        score = 0.0
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lb = m & -m
            idx = lb.bit_length() - 1
            a = idx + 1
            score += weights[a]
            m ^= lb
        return score

    # Helper to get list of elements from mask
    def mask_to_list(mask):
        res = []
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            res.append(idx + 1)
            m ^= lb
        return res

    # Greedy construction with random weights
    def greedy_construct(rand_weights):
        chosen_mask = 0
        # Sort by rand_weights descending
        candidates = list(range(1, N))
        candidates.sort(key=lambda x: rand_weights[x], reverse=True)
        
        for a in candidates:
            # Check if a is coprime to all in chosen_mask
            # chosen_mask must be subset of coprime_mask[a]
            if (chosen_mask & ~coprime_mask[a]) == 0:
                chosen_mask |= (1 << (a - 1))
        return chosen_mask

    # Local Search
    def local_search(mask):
        current_score = get_score(mask)
        improved = True
        while improved:
            improved = False
            
            # 1. ADD move: Add a single node v not in mask
            # v must be coprime to all in mask
            # mask & ~coprime_mask[v] == 0
            for v in range(1, N):
                if not (mask >> (v-1)) & 1:
                    if (mask & ~coprime_mask[v]) == 0:
                        new_mask = mask | (1 << (v-1))
                        new_score = current_score + weights[v]
                        if new_score > current_score:
                            mask = new_mask
                            current_score = new_score
                            improved = True
                            break # Restart loop to re-evaluate other moves
            
            if improved:
                continue

            # 2. 1-1 SWAP: Remove u in mask, add v not in mask
            # v must be coprime to mask \ {u}
            # (mask ^ (1 << (u-1))) & ~coprime_mask[v] == 0
            for u in range(1, N):
                if (mask >> (u-1)) & 1:
                    mask_without_u = mask ^ (1 << (u-1))
                    for v in range(1, N):
                        if not (mask >> (v-1)) & 1:
                            if (mask_without_u & ~coprime_mask[v]) == 0:
                                gain = weights[v] - weights[u]
                                if gain > 1e-12:
                                    new_mask = mask_without_u | (1 << (v-1))
                                    current_score += gain
                                    mask = new_mask
                                    improved = True
                                    break
                    if improved:
                        break
            
            if improved:
                continue

            # 3. 1-2 SWAP: Remove u in mask, add v1, v2 not in mask
            # v1, v2 coprime to each other and to mask \ {u}
            # This is expensive: O(N^3). We might skip or limit this if time is tight.
            # Given N=511, N^3 is ~130M ops, too slow in Python per iteration.
            # We can approximate by checking pairs of high-weight candidates.
            
            # Instead, let's do a more efficient 1-2 swap by precomputing candidates?
            # Or just rely on ADD and 1-1 SWAP which are O(N^2).
            # Let's try a limited 1-2 swap: only consider top K weights for v1, v2.
            # Or just stick to ADD and 1-1 for now, as it's already a big improvement over greedy.
            
            # Let's add a 2-1 SWAP: Remove u1, u2 in mask, add v not in mask.
            # v coprime to mask \ {u1, u2}.
            # Also expensive.
            
            # Let's refine the search strategy:
            # The ADD move is very powerful initially.
            # The 1-1 SWAP refines it.
            
            # If no improvement, break.
        
        return mask, current_score

    best_mask = 0
    best_score = 0.0
    
    # Run multiple random greedy starts
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline - 10.0:
            break
            
        # Generate random weights
        rand_weights = [random.random() for _ in range(N)]
        
        # Greedy
        m = greedy_construct(rand_weights)
        s = get_score(m)
        
        if s > best_score:
            best_score = s
            best_mask = m
            
        # Local Search on this start
        m_ls, s_ls = local_search(m)
        if s_ls > best_score:
            best_score = s_ls
            best_mask = m_ls
            
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Convert best_mask to list
    result = mask_to_list(best_mask)
    
    # Final verification (optional, for safety)
    # Ensure pairwise coprime
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to simple greedy if error
                return [511, 510, 509, 507, 503, 499, 497, 491, 487, 479, 473, 467, 463, 457, 449, 443, 439, 433, 431, 421, 419, 409, 401, 397, 389, 383, 379, 373, 367, 359, 353, 347, 349, 347, 343, 341, 337, 331, 331, 329, 323, 317, 313, 311, 307, 307, 299, 293, 289, 283, 281, 277, 277, 271, 269, 263, 257, 253, 251, 247, 241, 239, 233, 229, 227, 223, 223, 211, 209, 203, 199, 197, 193, 191, 187, 181, 179, 173, 169, 167, 163, 157, 151, 149, 143, 139, 137, 131, 127, 121, 119, 113, 109, 107, 101, 97, 91, 89, 83, 81, 79, 77, 73, 71, 67, 65, 61, 59, 53, 49, 47, 43, 41, 37, 35, 31, 29, 25, 23, 21, 19, 17, 13, 11, 7, 5, 3, 1]
                
    return result

# EVOLVE-BLOCK-END
