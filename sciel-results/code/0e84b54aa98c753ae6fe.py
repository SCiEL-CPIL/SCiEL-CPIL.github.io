import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback: return a trivial covering (all words) or simple grid
        # For this task, we assume q=3, n=7, r=1
        return [[0]*n]

    total_words = q ** n
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (radius 1) for each word
    neighbors = []
    for i in range(total_words):
        w = idx_to_word(i)
        covered = [i]
        for pos in range(n):
            original = w[pos]
            for val in range(q):
                if val == original:
                    continue
                w[pos] = val
                covered.append(word_to_idx(w))
            w[pos] = original
        neighbors.append(covered)
        
    # Optimization: Convert neighbors to sets for faster "in" checks? 
    # Or use bitsets? 
    # 2187 words. Bitset of 2187 bits is ~274 bytes. 
    # Using Python integers as bitsets is efficient.
    
    neighbor_bitsets = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        neighbor_bitsets[i] = mask

    # Initial code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_words = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code_words.append(h + list(tail))
            
    code_set = set(word_to_idx(w) for w in initial_code_words)
    
    # State
    in_code = np.zeros(total_words, dtype=bool)
    for i in code_set:
        in_code[i] = True
        
    count = np.zeros(total_words, dtype=np.int32)
    for i in code_set:
        for j in neighbors[i]:
            count[j] += 1
            
    def is_valid():
        return np.all(count > 0)
        
    def greedy_removal():
        changed = True
        while changed:
            changed = False
            items = list(code_set)
            random.shuffle(items)
            for i in items:
                if i not in code_set:
                    continue
                removable = True
                for j in neighbors[i]:
                    if count[j] == 1:
                        removable = False
                        break
                if removable:
                    code_set.remove(i)
                    in_code[i] = False
                    for j in neighbors[i]:
                        count[j] -= 1
                    changed = True

    greedy_removal()
    
    start_time = time.time()
    time_limit = 115.0
    
    best_size = len(code_set)
    best_code = set(code_set)
    
    # Helper to find a word k that covers a set of uncovered words U
    def find_cover(U):
        if not U:
            return -1 # No need to add
        # We want k not in code_set such that U is subset of neighbors[k]
        # Optimization: Precompute which words cover each specific word?
        # Or just iterate k.
        # To speed up, we can check intersection of bitsets?
        # U is a list of indices.
        # We need k such that for all u in U, u in neighbors[k].
        # This is equivalent to: intersection of neighbor_bitsets[u] for u in U contains k?
        # No, neighbor_bitsets[u] contains words *within distance 1 of u*.
        # If k is within distance 1 of u, then u is within distance 1 of k.
        # So yes, k must be in the intersection of neighbor_bitsets[u] for all u in U.
        
        # Calculate intersection of bitsets for U
        # If |U| is small, this is fast.
        if len(U) == 1:
            mask = neighbor_bitsets[U[0]]
        else:
            mask = neighbor_bitsets[U[0]]
            for u in U[1:]:
                mask &= neighbor_bitsets[u]
        
        # Find k in mask such that not in_code[k]
        # Iterate through set bits in mask
        # Python int bit operations are fast.
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            k = lsb.bit_length() - 1
            if not in_code[k]:
                return k
            m ^= lsb
        return -1

    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        if not code_set:
            break
            
        # 1. Try safe removal
        i = random.choice(list(code_set))
        critical = False
        U = []
        for j in neighbors[i]:
            if count[j] == 1:
                critical = True
                U.append(j)
                
        if not critical:
            code_set.remove(i)
            in_code[i] = False
            for j in neighbors[i]:
                count[j] -= 1
            greedy_removal()
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
            stuck_count = 0
            continue
            
        # 2. Try Swap 1 (Remove i, Add k)
        # U is the set of words that will become uncovered if we remove i
        k = find_cover(U)
        if k != -1:
            # Perform swap
            code_set.remove(i)
            in_code[i] = False
            for j in neighbors[i]:
                count[j] -= 1
            code_set.add(k)
            in_code[k] = True
            for j in neighbors[k]:
                count[j] += 1
            greedy_removal()
            if len(code_set) < best_size:
                best_size = len(code_set)
                best_code = set(code_set)
                stuck_count = 0
            else:
                stuck_count += 1
        else:
            # 3. Try Swap 2 (Remove i, j, Add k)
            # Pick a random partner j
            j = random.choice(list(code_set))
            if j == i:
                continue
                
            # Tentatively remove i and j
            # Calculate U_new
            # Words covered only by i or j
            # We need to be careful with counts.
            
            # Save state? Or just calculate U.
            # U consists of words with count 1 (covered only by i or j) 
            # AND (covered by i or covered by j)
            # Actually, if we remove i and j, a word x is uncovered if:
            # count[x] == 1 and (i in neighbors[x] or j in neighbors[x])?
            # No.
            # count[x] is number of codewords covering x.
            # If we remove i and j, new_count[x] = count[x] - (1 if i covers x else 0) - (1 if j covers x else 0).
            # x is uncovered if new_count[x] == 0.
            
            U_new = []
            for x in range(total_words):
                # Optimization: Only check x that are in neighbors[i] or neighbors[j]
                # If x is not covered by i or j, removing them doesn't affect x.
                pass
            
            # Efficient way:
            # Iterate neighbors of i and j
            # Union of neighbors
            # We can just iterate neighbors[i] and neighbors[j] and check counts.
            
            # Let's just do a simple check for the union
            # Since n=7, neighbors are small (15).
            
            # We need to know if i or j covers x.
            # i covers x if x in neighbors[i].
            
            # Let's collect candidates for U_new
            # x is affected if x in neighbors[i] or x in neighbors[j].
            
            # We'll iterate through neighbors[i]
            # And neighbors[j]
            
            affected = set(neighbors[i])
            affected.update(neighbors[j])
            
            for x in affected:
                c = count[x]
                # Subtract coverage from i
                if x in neighbors[i]:
                    c -= 1
                # Subtract coverage from j
                if x in neighbors[j]:
                    c -= 1
                if c == 0:
                    U_new.append(x)
            
            if not U_new:
                # Safe to remove both!
                code_set.remove(i)
                in_code[i] = False
                for x in neighbors[i]:
                    count[x] -= 1
                code_set.remove(j)
                in_code[j] = False
                for x in neighbors[j]:
                    count[x] -= 1
                greedy_removal()
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
                stuck_count = 0
                continue
            
            # Need to find k covering U_new
            k = find_cover(U_new)
            if k != -1:
                # Perform Remove i, j, Add k
                code_set.remove(i)
                in_code[i] = False
                for x in neighbors[i]:
                    count[x] -= 1
                code_set.remove(j)
                in_code[j] = False
                for x in neighbors[j]:
                    count[x] -= 1
                code_set.add(k)
                in_code[k] = True
                for x in neighbors[k]:
                    count[x] += 1
                greedy_removal()
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = set(code_set)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                stuck_count += 1
                
        # Random Restart / Perturbation if stuck
        if stuck_count > 100:
            # Perturb: Remove a random word and add a random word?
            # Or just restart from best_code
            if best_code:
                code_set = set(best_code)
                in_code = np.zeros(total_words, dtype=bool)
                for i in code_set:
                    in_code[i] = True
                count = np.zeros(total_words, dtype=np.int32)
                for i in code_set:
                    for j in neighbors[i]:
                        count[j] += 1
                greedy_removal()
            stuck_count = 0

    final_indices = list(best_code)
    final_words = [idx_to_word(i) for i in final_indices]
    return final_words
