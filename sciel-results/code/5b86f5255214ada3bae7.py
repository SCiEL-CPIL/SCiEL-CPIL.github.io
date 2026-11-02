import itertools
import random
import sys
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    num_blocks = (N + 63) // 64
    balls = np.zeros((N, num_blocks), dtype=np.uint64)
    point_to_balls = [[] for _ in range(N)]
    
    for i in range(N):
        w = words[i]
        # Set bit i
        balls[i, i // 64] |= (1 << (i % 64))
        point_to_balls[i].append(i)
        
        for pos in range(n):
            orig_val = w[pos]
            for val in range(q):
                if val != orig_val:
                    new_w = list(w)
                    new_w[pos] = val
                    j = word_to_idx[tuple(new_w)]
                    balls[i, j // 64] |= (1 << (j % 64))
                    point_to_balls[j].append(i)

    ALL_ONES = np.ones(num_blocks, dtype=np.uint64)
    # Mask out extra bits in last block
    last_block_bits = N % 64
    if last_block_bits:
        ALL_ONES[-1] = (1 << last_block_bits) - 1

    def greedy_construct():
        covered = np.zeros(num_blocks, dtype=np.uint64)
        code = []
        code_set = set()
        
        while not np.array_equal(covered, ALL_ONES):
            max_gain = -1
            best_idx = -1
            candidates = []
            
            # Vectorized gain calculation?
            # gains = (balls & ~covered).sum(axis=1)
            # This creates a temporary array of size N x num_blocks.
            # 6561 * 103 * 8 bytes = 5.4 MB.
            # 500 iterations * 5.4 MB = 2.7 GB memory traffic.
            # This is likely fast in C.
            
            not_covered = ~covered
            # Ensure we don't count bits outside N in the last block?
            # The balls are 0 outside N, so it's fine.
            
            gains = (balls & not_covered).sum(axis=1)
            
            # Find max
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(best_idx)
            code_set.add(best_idx)
            covered |= balls[best_idx]
            
        return code

    def local_search(code):
        code_set = set(code)
        improved = True
        
        while improved:
            improved = False
            
            # 1. Deletion Pass
            new_code = []
            for c in code:
                others = np.zeros(num_blocks, dtype=np.uint64)
                for c2 in code:
                    if c2 != c:
                        others |= balls[c2]
                
                # Check if balls[c] is subset of others
                # (balls[c] & ~others).sum() == 0
                if (balls[c] & ~others).sum() == 0:
                    improved = True
                else:
                    new_code.append(c)
            
            code = new_code
            code_set = set(code)
            
            if not code:
                break

            # 2. Swap Pass
            swaps_made = 0
            max_swaps = 50
            
            for idx in range(len(code)):
                if swaps_made >= max_swaps:
                    break
                    
                c = code[idx]
                others = np.zeros(num_blocks, dtype=np.uint64)
                for c2 in code:
                    if c2 != c:
                        others |= balls[c2]
                
                c_unique = balls[c] & ~others
                if c_unique.sum() == 0:
                    continue
                
                # Find candidates
                candidates = set()
                # Iterate set bits in c_unique
                for block_idx in range(num_blocks):
                    bits = c_unique[block_idx]
                    while bits:
                        lsb = bits & -bits
                        bit_pos = lsb.bit_length() - 1
                        p = block_idx * 64 + bit_pos
                        if p < N:
                            candidates.update(point_to_balls[p])
                        bits ^= lsb
                
                for c_prime in candidates:
                    if c_prime in code_set:
                        continue
                    
                    # Check if c_prime covers c_unique
                    if (balls[c_prime] & c_unique).sum() != c_unique.sum():
                        continue
                    
                    # Check validity
                    lost = balls[c] & ~balls[c_prime]
                    if (lost & ~others).sum() == 0:
                        code[idx] = c_prime
                        code_set.remove(c)
                        code_set.add(c_prime)
                        improved = True
                        swaps_made += 1
                        break
            
            if swaps_made > 0:
                continue
                
        return code

    best_solution = None
    best_len = float('inf')
    
    num_passes = 5
    
    for _ in range(num_passes):
        code = greedy_construct()
        code = local_search(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_solution = code
            
    return [list(words[i]) for i in best_solution]
