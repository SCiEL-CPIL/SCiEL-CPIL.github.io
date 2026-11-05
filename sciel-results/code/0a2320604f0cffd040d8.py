import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = 145.0  # 5s margin before the 150s kill
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    word = [1]
    
    for k in range(2, n + 1):
        # Check time
        if time.time() > deadline - 5.0:
            # Not enough time to do optimization for this k or larger.
            # We must finish the construction for n.
            # Use standard order for this k and subsequent.
            # Standard order is just the permutations in lex order?
            # Or just append k to each permutation block?
            # The seed program's fallback was incomplete.
            # Let's just use a simple valid construction for the remaining steps.
            # For k, we have a list of (k-1)-perms.
            # We can just iterate through them in the order they appear in the current word.
            # This guarantees validity but might be long.
            # Actually, if we are out of time, we just have to produce *some* valid word.
            # The standard recursive construction SP(k) from SP(k-1) is valid.
            # The order of perms in `perms` list is determined by their appearance in `word`.
            # If we just use that order, is it the standard construction?
            # The standard construction usually uses a specific order (e.g., lex).
            # But any order of the (k-1)-perms produces a valid superpermutation for k,
            # provided we connect them correctly?
            # Wait. The construction is:
            # For each perm p in order: append p, k, p.
            # Overlaps are handled by the string concatenation logic.
            # Does ANY order of the (k-1)-perms produce a valid superpermutation?
            # Yes, because each block p, k, p contains all permutations of p with k inserted?
            # No. The block p, k, p contains:
            # 1. Permutations of p (length k-1) - already in word.
            # 2. Permutations of p with k inserted?
            # Actually, the standard proof is that the set of windows of length k
            # in the constructed word covers all k! permutations.
            # If we change the order of blocks, do we still cover all?
            # The windows of length k are:
            # - Those entirely within a block (same as before, cover perms of k-1 with k in middle? No.)
            # - Those spanning block boundaries.
            # The standard construction relies on the specific order to ensure all
            # permutations are covered.
            # Specifically, the "new" permutations introduced by block i are those
            # that start with the suffix of block i-1 and end with the prefix of block i+1?
            # Actually, it's known that ANY order of the (k-1)-perms works to produce
            # a valid superpermutation, BUT the length depends on the order.
            # Wait, is that true?
            # Let's check n=3.
            # Perms of 1,2: 12, 21.
            # Order 1: 12, 21.
            # Word: 1 2 1 (from 12) -> 1 2 1 2 1 (from 21, overlap 21? No, 12 ends in 2, 21 starts in 2. Overlap 1? 2. 12 ends in 2, 21 starts in 2. Overlap 1. So 121 + 21 = 12121? No.
            # Block 12: 1 2 1.
            # Block 21: 2 1 2.
            # Concat: 1 2 1 (ends in 1). Block 2 1 2 (starts in 2). No overlap.
            # Word: 1 2 1 2 1 2. Length 6.
            # Perms of 1,2,3: 123, 132, 213, 231, 312, 321.
            # Windows of 121212: 121, 212, 121, 212. Only 121 and 212. INVALID.
            
            # So ANY order does NOT work. The standard construction has a specific property.
            # The standard construction SP(k) is defined recursively.
            # If we break the recursion, we might lose validity.
            
            # However, the seed program's "fallback" was just `pass` and then continuing.
            # This implies the seed program might have been invalid if it hit the deadline.
            # But the seed scored 1.0, so it likely finished in time or the deadline wasn't hit.
            # For n=7, the standard construction length is 5905.
            # The standard construction takes very little time to compute if we just follow the recipe.
            # The optimization (TSP/SA) is what takes time.
            
            # So if we run out of time, we should just finish the standard construction.
            # How to do standard construction quickly?
            # We have the list of perms. We need to order them in the "standard" way.
            # The standard way is usually the order they appear in the previous superpermutation?
            # Or lex order?
            # Actually, the recursive definition:
            # SP(k) = SP(k-1) with k inserted? No.
            # SP(k) is formed by taking SP(k-1) and inserting k?
            # No, the common construction is:
            # Let P_{k-1} be the list of permutations of 1..k-1.
            # For each p in P_{k-1} (in some order), append p, k, p.
            # The order of P_{k-1} is the order of the (k-1)-windows in SP(k-1).
            # So if we just use the order they appear in `word`, that IS the standard construction.
            
            # So fallback: use the order in `perms` (which is appearance order in `word`).
            best_order = list(range(len(perms)))
            
        else:
            # Extract unique (k-1)-permutations from current word
            perms = []
            seen = set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            num_perms = len(perms)
            
            if num_perms == 0:
                return "".join(map(str, word))
            
            # Determine strategy based on time and size
            time_remaining = deadline - time.time()
            
            best_order = list(range(num_perms))
            best_len = float('inf')
            
            # If we have enough time and num_perms is manageable, try optimization
            if time_remaining > 20.0 and num_perms > 10:
                # Precompute overlap matrix
                # overlap[i][j] = length of longest suffix of perms[i] that is prefix of perms[j]
                overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
                
                # Optimization: Compute in C-like speed? No, Python.
                # But for n=7, k=7, num_perms=5040.
                # 5040^2 = 25M.
                # Each check is O(k).
                # 25M * 5 = 125M ops. Too slow in Python?
                # Maybe.
                # Let's try to be smarter.
                # We can use a dictionary for prefixes/suffixes.
                # But max overlap is small (k-2).
                
                # Let's just do it, but with a time check.
                t0 = time.time()
                for i in range(num_perms):
                    pi = perms[i]
                    for j in range(num_perms):
                        if i == j:
                            continue
                        pj = perms[j]
                        max_ov = k - 2
                        ov = 0
                        # Check from max_ov down to 0
                        for o in range(max_ov, 0, -1):
                            if pi[-o:] == pj[:o]:
                                ov = o
                                break
                        overlap_matrix[i][j] = ov
                    
                    if time.time() - t0 > 5.0:
                        # Too slow, break and use fallback
                        break
                
                # If we broke early, overlap_matrix is incomplete.
                # Check if we finished.
                if time.time() - t0 > 5.0 and num_perms > 1000:
                    # Fallback to greedy without full matrix?
                    # Or just use standard order.
                    pass
                else:
                    # Run Simulated Annealing
                    # Energy = total length
                    # Length = sum(block_len - overlap)
                    # block_len = 2*(k-1) + 1
                    # E(order) = block_len + sum_{i=1}^{N-1} (block_len - overlap[order[i-1]][order[i]])
                    # Minimize E.
                    
                    block_len = 2 * (k - 1) + 1
                    
                    # Initial order: standard (identity)
                    current_order = list(range(num_perms))
                    current_energy = block_len
                    for i in range(1, num_perms):
                        current_energy += block_len - overlap_matrix[current_order[i-1]][current_order[i]]
                    
                    best_order = current_order
                    best_energy = current_energy
                    
                    # SA Parameters
                    T = 10.0
                    T_min = 0.1
                    alpha = 0.999
                    # Number of steps
                    # Budget: 50s
                    sa_deadline = time.time() + 50.0
                    
                    # Random seed for reproducibility? No, random.
                    random.seed(42)
                    
                    while time.time() < sa_deadline and T > T_min:
                        # Swap two random elements
                        i = random.randint(0, num_perms - 1)
                        j = random.randint(0, num_perms - 1)
                        if i == j:
                            continue
                        if abs(i - j) == 1:
                            continue # Swapping adjacent is a 2-opt move, but let's do random swaps first.
                        
                        # Calculate delta E
                        # Remove i, j from their positions and swap.
                        # This is complex for arbitrary i, j.
                        # Let's do adjacent swaps for simplicity, or 2-opt.
                        # Adjacent swap:
                        # i, j = i, i+1
                        # E changes only for links (i-1, i), (i, i+1), (i+1, i+2)
                        
                        i = random.randint(0, num_perms - 2)
                        j = i + 1
                        
                        a = current_order[i]
                        b = current_order[j]
                        
                        # Links affected:
                        # (i-1, a), (a, b), (b, i+2)
                        # New:
                        # (i-1, b), (b, a), (a, i+2)
                        
                        delta_e = 0
                        
                        if i > 0:
                            prev = current_order[i-1]
                            delta_e += (block_len - overlap_matrix[prev][b]) - (block_len - overlap_matrix[prev][a])
                        if i + 2 < num_perms:
                            next_ = current_order[i+2]
                            delta_e += (block_len - overlap_matrix[a][next_]) - (block_len - overlap_matrix[b][next_])
                        # Middle link:
                        delta_e += (block_len - overlap_matrix[b][a]) - (block_len - overlap_matrix[a][b])
                        
                        # Acceptance
                        if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                            # Swap
                            current_order[i], current_order[j] = b, a
                            current_energy += delta_e
                            
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_order = current_order[:]
                        
                        T *= alpha
                        
        # Construct new word using best_order
        new_word = []
        for idx, p_idx in enumerate(best_order):
            p = perms[p_idx]
            block = list(p) + [k] + list(p)
            
            if idx == 0:
                new_word = block
            else:
                # Recalculate overlap for safety
                max_overlap = min(len(new_word), len(block))
                overlap = 0
                for o in range(min(k, max_overlap), -1, -1):
                    if o == 0:
                        break
                    if new_word[-o:] == block[:o]:
                        overlap = o
                        break
                new_word += block[overlap:]
                
        word = new_word

    return "".join(map(str, word))
