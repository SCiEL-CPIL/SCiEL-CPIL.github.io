import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 108  # 108s for computation, 12s buffer for return/eval overhead
    
    N = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute neighbors using numpy for efficiency
    neighbors_list = []
    for c_idx in range(N):
        word = np.zeros(n, dtype=np.int32)
        tmp = c_idx
        for i in range(n - 1, -1, -1):
            word[i] = tmp % q
            tmp //= q
        
        # Generate all neighbors within radius 1
        neighbors = [c_idx]
        for i in range(n):
            orig = word[i]
            for new_val in range(q):
                if new_val != orig:
                    new_word = word.copy()
                    new_word[i] = new_val
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_word[j]
                    neighbors.append(idx)
        
        neighbors_list.append(np.array(neighbors, dtype=np.int32))
    
    # Precompute ball_sets as frozensets for O(1) subset checks
    ball_sets = [frozenset(neighbors_list[i]) for i in range(N)]
    
    # Precompute all_words for final conversion
    all_words = []
    for i in range(N):
        word = [0] * n
        tmp = i
        for j in range(n - 1, -1, -1):
            word[j] = tmp % q
            tmp //= q
        all_words.append(word)
    
    def greedy_cover():
        """Greedy set cover with numpy-accelerated counting."""
        uncovered = np.ones(N, dtype=np.bool_)
        code = []
        code_set = set()
        
        while uncovered.any():
            if time.time() > deadline:
                break
            
            # Find the codeword covering the most uncovered words
            best_code = None
            best_count = -1
            
            for c_idx in range(N):
                if c_idx in code_set:
                    continue
                count = np.sum(uncovered[neighbors_list[c_idx]])
                if count > best_count:
                    best_count = count
                    best_code = c_idx
            
            if best_code is None:
                break
            
            code.append(best_code)
            code_set.add(best_code)
            uncovered[neighbors_list[best_code]] = False
        
        return code
    
    def local_search(code):
        """Remove redundant codewords using coverage counts."""
        code_set = set(code)
        
        # Build coverage count array
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors_list[c]] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c_idx = code[i]
                # Check if c_idx is the only cover for any word
                is_redundant = True
                for w in neighbors_list[c_idx]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    cover_count[neighbors_list[c_idx]] -= 1
                    code.pop(i)
                    code_set.discard(c_idx)
                    improved = True
                    break
        
        return code
    
    def branch_and_bound_mwis(code, time_limit=20):
        """
        MWIS-style local search: try to replace codewords to reduce total count.
        Uses a conflict graph perspective where we want to maximize independent set size
        (minimize code size) by swapping codewords.
        """
        bnb_start = time.time()
        bnb_deadline = min(bnb_start + time_limit, deadline)
        
        code = list(code)
        code_set = set(code)
        best_code = list(code)
        best_len = len(code)
        
        # Coverage count array
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors_list[c]] += 1
        
        # Precompute for each codeword, which words it uniquely covers
        # This helps in deciding which codewords to try replacing
        for iteration in range(50):
            if time.time() > bnb_deadline:
                break
            
            # Try to remove or swap each codeword
            for i in range(len(code)):
                if time.time() > bnb_deadline:
                    break
                
                c_idx = code[i]
                # Check if c_idx is redundant
                is_redundant = True
                for w in neighbors_list[c_idx]:
                    if cover_count[w] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    cover_count[neighbors_list[c_idx]] -= 1
                    code.pop(i)
                    code_set.discard(c_idx)
                    if len(code) < best_len:
                        best_code = list(code)
                        best_len = len(code)
                    continue
                
                # Try to replace c_idx with a better codeword
                # Find words that c_idx uniquely covers
                unique_words = []
                for w in neighbors_list[c_idx]:
                    if cover_count[w] == 1:
                        unique_words.append(w)
                
                if not unique_words:
                    continue
                
                # Find candidates that cover these unique words
                best_candidate = None
                best_score = -1
                
                for cand in range(N):
                    if cand in code_set or cand == c_idx:
                        continue
                    
                    # Count how many unique words this candidate covers
                    cand_ball = neighbors_list[cand]
                    covered_unique = sum(1 for w in unique_words if cover_count[w] > 1 or w in cand_ball)
                    
                    # Also count how many total words it covers that are currently covered
                    total_covered = np.sum(cover_count[cand_ball] > 0)
                    
                    score = covered_unique * 2 + total_covered
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                
                if best_candidate is not None and best_score > len(unique_words):
                    # Swap
                    cover_count[neighbors_list[c_idx]] -= 1
                    cover_count[neighbors_list[best_candidate]] += 1
                    code[i] = best_candidate
                    code_set.discard(c_idx)
                    code_set.add(best_candidate)
                    
                    # Try to remove redundant codewords after swap
                    for k in range(len(code)):
                        c_k = code[k]
                        is_redundant_k = True
                        for w in neighbors_list[c_k]:
                            if cover_count[w] == 1:
                                is_redundant_k = False
                                break
                        if is_redundant_k:
                            cover_count[neighbors_list[c_k]] -= 1
                            code.pop(k)
                            code_set.discard(c_k)
                            break
                    
                    if len(code) < best_len:
                        best_code = list(code)
                        best_len = len(code)
        
        return best_code
    
    def simulated_annealing(code, max_time=30):
        """SA with pair swaps using precomputed ball_sets for fast validity checks."""
        sa_start = time.time()
        sa_deadline = min(sa_start + max_time, deadline)
        
        code = list(code)
        code_set = set(code)
        
        # Coverage count array
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code:
            cover_count[neighbors_list[c]] += 1
        
        best_code = list(code)
        best_len = len(code)
        
        T = 5.0
        cooling_rate = 0.995
        min_T = 0.05
        
        while time.time() < sa_deadline and T > min_T:
            if len(code) < 2:
                break
            
            # Pick two random codewords to swap
            i, j = random.sample(range(len(code)), 2)
            c_i, c_j = code[i], code[j]
            
            # Quick validity check using ball_sets
            only_i = ball_sets[c_i] - ball_sets[c_j]
            only_j = ball_sets[c_j] - ball_sets[c_i]
            
            # Generate candidates that cover the critical words
            candidates = []
            for _ in range(30):
                new_c = random.randint(0, N - 1)
                if new_c in code_set or new_c == c_i or new_c == c_j:
                    continue
                b_new = ball_sets[new_c]
                if len(only_i - b_new) < len(only_i) or len(only_j - b_new) < len(only_j):
                    candidates.append(new_c)
            
            if len(candidates) < 2:
                T *= cooling_rate
                continue
            
            new_c1, new_c2 = random.sample(candidates, 2)
            
            # Full validity check
            new_cover = ball_sets[new_c1] | ball_sets[new_c2]
            if len(only_i - new_cover) == 0 and len(only_j - new_cover) == 0:
                # Update coverage
                cover_count[neighbors_list[c_i]] -= 1
                cover_count[neighbors_list[c_j]] -= 1
                cover_count[neighbors_list[new_c1]] += 1
                cover_count[neighbors_list[new_c2]] += 1
                
                code[i] = new_c1
                code[j] = new_c2
                code_set.discard(c_i)
                code_set.discard(c_j)
                code_set.add(new_c1)
                code_set.add(new_c2)
                
                # Try to remove redundant codewords
                for k in range(len(code)):
                    c_k = code[k]
                    is_redundant = True
                    for w in neighbors_list[c_k]:
                        if cover_count[w] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        cover_count[neighbors_list[c_k]] -= 1
                        code.pop(k)
                        code_set.discard(c_k)
                        break
                
                if len(code) < best_len:
                    best_code = list(code)
                    best_len = len(code)
            
            T *= cooling_rate
        
        return best_code
    
    # Multi-start approach
    best_overall = None
    best_len = float('inf')
    
    num_starts = 4
    start_time_multi = time.time()
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        random.seed(42 + start_idx * 1000)
        
        # Greedy cover
        code = greedy_cover()
        
        if time.time() > deadline - 10:
            if len(code) < best_len:
                best_overall = code
                best_len = len(code)
            break
        
        # Local search: remove redundant
        code = local_search(code)
        
        if time.time() > deadline - 25:
            if len(code) < best_len:
                best_overall = code
                best_len = len(code)
            break
        
        # MWIS-style branch and bound
        bnb_time = min(15, deadline - time.time() - 20)
        if bnb_time > 3:
            code = branch_and_bound_mwis(code, time_limit=bnb_time)
        
        if time.time() > deadline - 20:
            if len(code) < best_len:
                best_overall = code
                best_len = len(code)
            break
        
        # Simulated annealing
        sa_time = min(20, deadline - time.time() - 10)
        if sa_time > 5:
            code = simulated_annealing(code, max_time=sa_time)
        
        # Final local search
        code = local_search(code)
        
        if len(code) < best_len:
            best_overall = code
            best_len = len(code)
    
    # Final local search on best result
    if best_overall is not None and time.time() < deadline - 5:
        best_overall = local_search(best_overall)
    
    if best_overall is None:
        # Fallback: simple greedy
        best_overall = greedy_cover()
    
    # Verify and fix if needed
    cover_count = np.zeros(N, dtype=np.int32)
    for c in best_overall:
        cover_count[neighbors_list[c]] += 1
    
    if np.any(cover_count == 0):
        # Add missing coverage
        while np.any(cover_count == 0) and time.time() < deadline:
            uncovered_words = np.where(cover_count == 0)[0]
            best_c = None
            best_count = -1
            for c_idx in range(N):
                if c_idx in best_overall:
                    continue
                count = np.sum(cover_count[neighbors_list[c_idx]] == 0)
                if count > best_count:
                    best_count = count
                    best_c = c_idx
            if best_c is None:
                break
            best_overall.append(best_c)
            cover_count[neighbors_list[best_c]] += 1
    
    # Convert to words
    result = [all_words[i] for i in best_overall]
    return result
