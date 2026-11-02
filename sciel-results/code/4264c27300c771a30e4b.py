import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Specific case: q=3, n=8, radius=1
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other cases
        total = q ** n
        if total > 100000:
            return []
        # Simple greedy for small general cases
        all_words = np.array(list(np.ndindex(*([q] * n))), dtype=np.int32)
        uncovered = set(all_words)
        codewords = []
        while uncovered and time.time() < deadline:
            best = None
            best_count = -1
            for w in uncovered:
                count = 0
                for i in range(n):
                    for v in range(q):
                        nw = list(w)
                        if nw[i] != v:
                            nw[i] = v
                            if tuple(nw) in uncovered:
                                count += 1
                if count > best_count:
                    best_count = count
                    best = w
            if best is None:
                break
            codewords.append(list(best))
            to_remove = set()
            for i in range(n):
                for v in range(q):
                    nw = list(best)
                    nw[i] = v
                    to_remove.add(tuple(nw))
            uncovered -= to_remove
        return codewords

    total_words = 3 ** 8
    num_coords = 8
    alphabet = 3

    # Precompute neighbors
    # Index is base-3 representation.
    # Neighbors are index itself + 2 * 3^i for each position i where digit is not 2,
    # and index - 2 * 3^i for each position i where digit is not 0.
    # Actually, simpler:
    # For index i, digit at pos j is d.
    # If d > 0, neighbor is i - 3^j.
    # If d < 2, neighbor is i + 3^j.
    
    powers = [3**i for i in range(num_coords)]
    
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        temp = i
        nbs = [i]
        for j in range(num_coords):
            d = temp % 3
            temp //= 3
            if d > 0:
                nbs.append(i - powers[j])
            if d < 2:
                nbs.append(i + powers[j])
        neighbors[i] = np.array(nbs, dtype=np.int32)

    # Greedy Construction
    def greedy_construct():
        uncovered = np.ones(total_words, dtype=np.bool_)
        counts = np.full(total_words, 17, dtype=np.int32)
        codewords = []
        
        # Randomize initial choice slightly or just pick max
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Find max count
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_cw = candidates[random.randint(0, len(candidates) - 1)]
            
            if counts[best_cw] == 0:
                break
            
            codewords.append(best_cw)
            
            for nb in neighbors[best_cw]:
                if uncovered[nb]:
                    uncovered[nb] = False
                    for cw in neighbors[nb]:
                        counts[cw] -= 1

        return set(codewords)

    # Local Search
    def local_search(code_set):
        improved = True
        # Precompute neighbor sets for faster lookup? 
        # Using list of lists is okay for 6561 size.
        
        while improved and time.time() < deadline:
            improved = False
            c_list = list(code_set)
            random.shuffle(c_list)
            
            # Compute coverage counts
            cov_count = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                cov_count[neighbors[c]] += 1
            
            # 1. Try Removals
            for c1 in c_list:
                if time.time() > deadline:
                    return code_set
                if c1 not in code_set:
                    continue
                
                # Check if c1 is removable
                # A word w in N(c1) is uncovered if cov_count[w] == 1
                removable = True
                for w in neighbors[c1]:
                    if cov_count[w] == 1:
                        removable = False
                        break
                
                if removable:
                    code_set.remove(c1)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Try 1-Swaps
            # For each c1, find u_words (words only covered by c1)
            # Find d such that N(d) covers u_words
            # And N(c1) \ N(d) is covered by others
            for c1 in c_list:
                if time.time() > deadline:
                    return code_set
                if c1 not in code_set:
                    continue
                
                u_words = []
                for w in neighbors[c1]:
                    if cov_count[w] == 1:
                        u_words.append(w)
                
                if not u_words:
                    # Should have been caught by removal, but just in case
                    code_set.remove(c1)
                    improved = True
                    break
                
                # Candidates for d: neighbors of u_words[0]
                # d must be in N(w) for all w in u_words
                # Intersection of N(w) for w in u_words
                # Since |u_words| is usually small, we can just check neighbors of u_words[0]
                candidates = set(neighbors[u_words[0]].tolist())
                
                found_swap = False
                for d in candidates:
                    if d in code_set:
                        continue
                    
                    # Check if d covers all u_words
                    valid = True
                    for w in u_words:
                        if w not in neighbors[d]: # This check is O(17) if we use set, but neighbors is list.
                            # Better to check if w is in the list
                            # Since d is in N(w), w must be in N(d) by symmetry
                            # So if d is in N(w), w is in N(d).
                            # We already know d is in N(u_words[0]).
                            # We need to check d in N(w) for other w in u_words.
                            # d in N(w) <=> w in N(d).
                            pass 
                            
                    # Actually, simpler:
                    # d must cover all u_words.
                    # This means for all w in u_words, dist(d,w) <= 1.
                    # Which means d is in N(w).
                    # We selected d from N(u_words[0]).
                    # So we just need to check if d is in N(w) for all other w in u_words.
                    
                    d_nbs_set = set(neighbors[d].tolist())
                    if not all(w in d_nbs_set for w in u_words):
                        continue
                        
                    # Check if removing c1 and adding d is valid
                    # Words in N(c1) \ N(d) must have cov_count >= 2 (so after removing c1, they still have >= 1)
                    c1_nbs_set = set(neighbors[c1].tolist())
                    
                    valid = True
                    for w in c1_nbs_set - d_nbs_set:
                        if cov_count[w] <= 1:
                            valid = False
                            break
                    
                    if valid:
                        code_set.remove(c1)
                        code_set.add(d)
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if improved:
                continue

            # 3. Try 2-Swaps (Remove c1, c2; Add d1, d2)
            # This is expensive. Only do if time permits and we are stuck.
            if time.time() < deadline - 20:
                # Limit to small subsets to keep it fast
                # Pick random pairs
                for _ in range(10): # Limit attempts
                    if time.time() > deadline:
                        return code_set
                    if len(c_list) < 2:
                        break
                    i = random.randint(0, len(c_list) - 1)
                    j = random.randint(0, len(c_list) - 1)
                    if i == j:
                        continue
                    c1 = c_list[i]
                    c2 = c_list[j]
                    if c1 not in code_set or c2 not in code_set:
                        continue
                    
                    # Words that would be uncovered if c1, c2 removed
                    # cov_count[w] - (1 if w in N(c1)) - (1 if w in N(c2)) <= 0
                    u_words = []
                    c1_nbs = neighbors[c1]
                    c2_nbs = neighbors[c2]
                    c1_set = set(c1_nbs.tolist())
                    c2_set = set(c2_nbs.tolist())
                    
                    for w in c1_set | c2_set:
                        rem = cov_count[w]
                        if w in c1_set:
                            rem -= 1
                        if w in c2_set:
                            rem -= 1
                        if rem <= 0:
                            u_words.append(w)
                    
                    if not u_words:
                        # Both redundant
                        code_set.remove(c1)
                        code_set.remove(c2)
                        improved = True
                        break
                    
                    # Find d1, d2 covering u_words
                    # d1 must cover some, d2 must cover the rest
                    # This is complex. Simplify: 
                    # Try to find d1 such that N(d1) covers a large part of u_words
                    # Then find d2 covering the rest.
                    
                    # Heuristic: 
                    # Pick u_words[0], candidates for d1 are N(u_words[0])
                    # For each d1, check how many u_words it covers.
                    # If it covers all, we just need to check if c1,c2 removal is valid with d1 added (and maybe d2=c1 or c2? No, we add new ones).
                    # Actually, if we add d1, we might still need d2.
                    
                    # Let's try a simpler 2-swap: Remove c1, add d1. (This is 1-swap, already done).
                    # True 2-swap: Remove c1, c2. Add d1, d2.
                    
                    # This is too expensive to do generally.
                    # Let's skip complex 2-swaps for now and rely on multiple restarts of 1-swaps/removals.
                    # The seed program also skipped complex 2-swaps.
                    pass

        return code_set

    # Main
    best_code = None
    best_size = float('inf')
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        code_set = greedy_construct()
        code_set = local_search(code_set)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = code_set

    if best_code is None:
        best_code = greedy_construct()

    # Convert indices to digit lists
    def idx_to_digits(idx):
        digits = [0] * num_coords
        for i in range(num_coords - 1, -1, -1):
            digits[i] = idx % 3
            idx //= 3
        return digits

    result = [idx_to_digits(c) for c in best_code]
    # Ensure distinctness (should be distinct as it's a set)
    return result

