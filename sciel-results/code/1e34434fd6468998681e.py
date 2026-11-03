import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
    
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    
    q_powers = [q ** i for i in range(n)]
    
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets as Python sets for fast membership testing
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]
    
    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words
        
        while remaining > 0:
            if time.time() > deadline - 30.0:
                break
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]
            
            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])
            
            codewords.append(best_word)
            
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1
        
        return codewords
    
    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)
        
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)
        
        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1
        
        # Precompute: for each word, the set of words that cover it (i.e., its neighbors)
        # We already have neighbor_sets for this purpose.
        
        # For faster swap evaluation, precompute for each word w, the list of codeword indices that cover w
        # This is expensive, so we'll use a different approach: 
        # For each word w, count[w] tells us how many codewords cover it.
        # When we remove codeword ci, words with count==1 that are neighbors of ci become uncovered.
        
        while time.time() < time_limit:
            improved = False
            
            # Pass 1: Remove redundant codewords (vectorized min check)
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                
                # Check if all neighbors have count > 1
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]
            
            # Pass 2: Swaps - find codeword to replace with another that covers more unique words
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)
                
                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]
                    
                    # Words uniquely covered by ci (count == 1 and neighbor of ci)
                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]
                    
                    if len(critical) == 0:
                        # Can simply remove ci
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        break
                    
                    # Need to find a candidate that covers all critical words
                    # Start with intersection of neighbor sets of all critical words
                    if len(critical) == 1:
                        candidate_set = neighbor_sets[critical[0]].copy()
                    else:
                        candidate_set = neighbor_sets[critical[0]].copy()
                        for u in critical[1:]:
                            candidate_set &= neighbor_sets[u]
                            if not candidate_set:
                                break
                    
                    if not candidate_set:
                        continue
                    
                    # Filter out ci and already-in-code candidates
                    valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                    
                    if not valid_cands:
                        continue
                    
                    # Choose the candidate that maximizes additional coverage
                    # For each candidate, compute how many new words it covers (not covered by any other codeword)
                    best_cand = None
                    best_gain = -1
                    
                    for cand in valid_cands:
                        # New coverage: neighbors of cand with count == 0 (after removing ci)
                        # But we need to simulate: remove ci, add cand
                        # Words newly covered: neighbors of cand that are NOT neighbors of ci (or have count > 1)
                        # Actually, simpler: gain = number of neighbors of cand with count == 1 that are NOT critical
                        # Plus neighbors with count == 0 (but those don't exist since we have a valid covering)
                        
                        # Let's compute: after removing ci, count' = count - 1 for neighbors of ci
                        # Then adding cand: count'' = count' + 1 for neighbors of cand
                        # Net gain in covered words: neighbors of cand that have count' == 0
                        # i.e., neighbors of cand that are NOT neighbors of ci (since count[u] >= 1 for all u)
                        
                        # Simpler: gain = |neighbors(cand) \ neighbors(ci)|
                        # But we need to be more precise.
                        
                        # For each word u in neighbors(cand):
                        #   If u in neighbors(ci): count'[u] = count[u] - 1, then count''[u] = count[u]
                        #   If u not in neighbors(ci): count'[u] = count[u], then count''[u] = count[u] + 1
                        # 
                        # New words covered: u in neighbors(cand) where count'[u] == 0
                        # Since count[u] >= 1 for all u (valid covering), count'[u] == 0 iff u in neighbors(ci) and count[u] == 1
                        # So new words covered = neighbors(cand) ∩ critical
                        # But we already require that candidate covers all critical words, so this is |critical|
                        # 
                        # Actually, the gain is the number of words that become covered that weren't before.
                        # Since we maintain a valid covering, no word becomes uncovered.
                        # The question is: can we reduce the code size? No, we're swapping, not removing.
                        # 
                        # Wait, the goal is to minimize the number of codewords. Swapping doesn't reduce size.
                        # The reduction happens in Pass 1 (removal).
                        # 
                        # So for swaps, we want to find a swap that enables future removals.
                        # A good swap is one that creates more redundancy (higher minimum count).
                        
                        # Let's compute the minimum count after the swap.
                        # min_count_after = min over all words of count_after[u]
                        # count_after[u] = count[u] - (1 if u in neighbors(ci)) + (1 if u in neighbors(cand))
                        
                        # This is expensive to compute for all words. Let's just pick a random valid candidate.
                        best_cand = random.choice(valid_cands)
                        break
                    
                    if best_cand is not None:
                        for u in neighbors[ci]:
                            count[u] -= 1
                        for u in neighbors[best_cand]:
                            count[u] += 1
                        
                        in_code[ci] = False
                        in_code[best_cand] = True
                        cw[i] = best_cand
                        improved = True
                        break
            
            if not improved:
                # Try a perturbation: randomly swap a few codewords to escape local minimum
                if len(cw) > 2 and random.random() < 0.3:
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]
                    
                    # Check if swapping maintains coverage
                    temp_count = count.copy()
                    for u in neighbors[c1]:
                        temp_count[u] -= 1
                    for u in neighbors[c2]:
                        temp_count[u] += 1
                    
                    if np.min(temp_count) >= 1:
                        # Perform the swap
                        for u in neighbors[c1]:
                            count[u] -= 1
                        for u in neighbors[c2]:
                            count[u] += 1
                        in_code[c1] = False
                        in_code[c2] = True
                        cw[idx1] = c2
                        improved = True
                        
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
        
        return best_cw

    # Try multiple greedy initializations with different seeds
    best_overall = None
    best_overall_size = float('inf')
    
    # Allocate time for multiple attempts
    num_attempts = 3
    time_per_attempt = (deadline - 10.0 - start_time) / num_attempts
    
    for attempt in range(num_attempts):
        attempt_start = time.time()
        remaining_time = deadline - 10.0 - attempt_start
        if remaining_time < 15.0:
            break
        
        # Greedy init with different randomization
        if attempt == 0:
            cw = greedy_init(0)
        else:
            # Add some random perturbation to greedy
            cw = greedy_init(attempt * 1000)
        
        # Local search with remaining time
        ls_time = max(5.0, remaining_time * 0.7)
        result = local_search(cw, start_time + ls_time)
        
        if len(result) < best_overall_size:
            best_overall_size = len(result)
            best_overall = result
        
        if time.time() > deadline - 15.0:
            break
    
    if best_overall is None:
        # Fallback: single greedy
        best_overall = greedy_init(0)
    
    # Final polish: one more local search pass on the best result with remaining time
    final_time_limit = deadline - 3.0
    if time.time() < final_time_limit - 5.0:
        best_overall = local_search(best_overall, final_time_limit)
    
    # Convert back to list of lists
    result = []
    for ci in best_overall:
        word = []
        val = ci
        for pos in range(n - 1, -1, -1):
            word.append(val % q)
            val //= q
        word.reverse()
        result.append(word)
    
    return result
