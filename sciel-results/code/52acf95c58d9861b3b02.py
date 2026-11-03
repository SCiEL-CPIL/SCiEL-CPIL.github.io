"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search: fast greedy init + swap-based optimization.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # Precompute digit arrays for fast neighbor generation
    # For each word index, precompute its digits
    digits = np.zeros((total_words, n), dtype=np.uint8)
    for i in range(total_words):
        val = i
        for pos in range(n - 1, -1, -1):
            digits[i, pos] = val % q
            val //= q
    
    # Precompute single-position neighbor offsets for radius 1
    # For each position and each digit value, the offset to add/subtract
    # Actually, let's precompute all radius-1 neighbors for each word
    # Number of radius-1 neighbors: n * (q-1)
    num_r1 = n * (q - 1)
    
    # For radius 2, number of neighbors: C(n,2) * (q-1)^2
    from math import comb
    num_r2 = comb(n, 2) * (q - 1) ** 2 if radius >= 2 else 0
    
    # Precompute neighbor lists as arrays
    neighbors = []
    for w in range(total_words):
        word = digits[w]
        neigh_set = {w}
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    # Compute new index
                    new_word = word.copy()
                    new_word[pos] = val
                    new_idx = 0
                    for p in range(n - 1, -1, -1):
                        new_idx = new_idx * q + new_word[p]
                    neigh_set.add(new_idx)
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    o1 = word[pos1]
                    o2 = word[pos2]
                    for v1 in range(q):
                        if v1 == o1:
                            continue
                        for v2 in range(q):
                            if v2 == o2:
                                continue
                            new_word = word.copy()
                            new_word[pos1] = v1
                            new_word[pos2] = v2
                            new_idx = 0
                            for p in range(n - 1, -1, -1):
                                new_idx = new_idx * q + new_word[p]
                            neigh_set.add(new_idx)
        neighbors.append(sorted(neigh_set))
    
    # Convert to numpy arrays for faster iteration
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def greedy_init(seed_offset=0):
        """Fast greedy: pick word covering most uncovered, with randomization."""
        covered = np.zeros(total_words, dtype=np.bool_)
        codewords = []
        uncovered_count = total_words
        
        # For speed, sample from a subset when remaining is large
        while uncovered_count > 0:
            # Sample candidate words
            if uncovered_count > 1000:
                # Sample 500 random words
                candidates = random.sample(range(total_words), min(500, total_words))
            else:
                # Try all uncovered words (small set)
                candidates = [w for w in range(total_words) if not covered[w]]
            
            best_word = -1
            best_new = 0
            
            for w in candidates:
                if covered[w]:
                    continue
                nb = neighbors_np[w]
                new_cov = np.sum(~covered[nb])
                if new_cov > best_new:
                    best_new = new_cov
                    best_word = w
            
            if best_word == -1:
                # Fallback: pick any uncovered
                for w in range(total_words):
                    if not covered[w]:
                        best_word = w
                        break
                if best_word == -1:
                    break
            
            codewords.append(best_word)
            nb = neighbors_np[best_word]
            newly = ~covered[nb]
            covered[nb[newly]] = True
            uncovered_count -= np.sum(newly)
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Local search: remove redundant, then swap to improve coverage."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            nb = neighbors_np[cw]
            coverage_count[nb] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    nb = neighbors_np[cw]
                    # Check if any word is only covered by this codeword
                    if np.any(coverage_count[nb] == 1):
                        continue
                    # Remove it
                    coverage_count[nb] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                    removed_any = True
                    break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
            
            if len(codewords) == 0:
                break
            
            idx = random.randint(0, len(codewords) - 1)
            old_cw = codewords[idx]
            old_nb = neighbors_np[old_cw]
            
            # Words that would become uncovered
            would_uncover = old_nb[coverage_count[old_nb] == 1]
            
            if len(would_uncover) == 0:
                # Safe to remove
                coverage_count[old_nb] -= 1
                codewords.pop(idx)
                cw_set.discard(old_cw)
                continue
            
            # Try to find a replacement that covers all would_uncover
            # Sample from neighbors of the first uncovered word
            first_unc = would_uncover[0]
            candidates = neighbors_np[first_unc]
            random.shuffle(candidates.tolist())
            
            best_new_cw = -1
            best_gain = 0
            
            for new_cw in candidates[:100]:  # Limit candidates for speed
                if new_cw in cw_set or new_cw == old_cw:
                    continue
                new_nb = neighbors_np[new_cw]
                
                # Check if new covers all would_uncover
                # Convert would_uncover to set for fast lookup
                if not all(u in new_nb for u in would_uncover):
                    continue
                
                # Gain: number of words newly covered (coverage_count 0 -> 1)
                gain = np.sum(coverage_count[new_nb] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_new_cw = new_cw
            
            if best_new_cw != -1:
                coverage_count[old_nb] -= 1
                new_nb = neighbors_np[best_new_cw]
                coverage_count[new_nb] += 1
                codewords[idx] = best_new_cw
                cw_set.discard(old_cw)
                cw_set.add(best_new_cw)
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts with time budgeting
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    best_len = float('inf')
    
    restart_count = 0
    while time.time() - start_time < time_limit:
        restart_count += 1
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed
        if remaining_time < 10:
            break
        
        # Time allocation: greedy ~3s, optimize rest
        greedy_time = min(3, remaining_time * 0.15)
        opt_time = remaining_time - greedy_time - 2  # 2s safety
        
        if opt_time < 5:
            break
        
        # Use different random seeds for diversity
        random.seed(int(time.time() * 1000) + restart_count * 7919)
        
        cw = greedy_init()
        cw = optimize(cw, opt_time)
        
        if len(cw) < best_len:
            best_len = len(cw)
            best_codewords = cw
        
        # If we've found a good solution, spend less time per restart
        if best_len <= 500:
            if remaining_time < 30:
                break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
