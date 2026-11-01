import time
import random
import itertools
import numpy as np
import heapq

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Specific case for q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        total_words = q ** n
        
        # Generate all words as tuples
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        # Precompute spheres: list of indices for each word within distance 1
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
        
        # Precompute reverse spheres for faster update
        reverse_spheres = [[] for _ in range(total_words)]
        for i in range(total_words):
            for j in spheres[i]:
                reverse_spheres[j].append(i)
        
        # Precompute symmetry score for tie-breaking
        # Symmetry score: how "regular" a word is.
        # We want to favor words with balanced symbol frequencies.
        # Score = -sum(count[s]^2) for s in 0..2. Higher score = more balanced.
        symmetry_scores = np.zeros(total_words)
        for i, w in enumerate(all_words):
            counts = [0, 0, 0]
            for val in w:
                counts[val] += 1
            score = -(counts[0]**2 + counts[1]**2 + counts[2]**2)
            symmetry_scores[i] = score
        
        best_code = None
        best_size = total_words
        
        # Number of restarts depends on time
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Greedy Construction
            covered = np.zeros(total_words, dtype=bool)
            uncovered_count = total_words
            code = []
            picked = set()
            
            # Gain array: number of uncovered words covered by each candidate
            gain = np.zeros(total_words, dtype=int)
            for i in range(total_words):
                gain[i] = len(spheres[i])
                
            # Use heap for efficient max-gain selection
            # Tie-breaking: combine symmetry score with a small random perturbation
            heap = []
            noise = np.random.rand(total_words)
            
            for i in range(total_words):
                # Key = (-gain[i], -symmetry_scores[i], noise[i])
                heap.append((-gain[i], -symmetry_scores[i], noise[i], i))
            
            heapq.heapify(heap)
            
            while uncovered_count > 0:
                if time.time() > deadline:
                    break
                
                best_candidate = -1
                best_gain = 0
                
                # Pop until we find a valid candidate
                while heap:
                    neg_gain, neg_sym, noise_val, cand_idx = heapq.heappop(heap)
                    
                    # Check if stale
                    if -neg_gain != gain[cand_idx]:
                        continue
                    
                    # Check if picked
                    if cand_idx in picked:
                        continue
                    
                    best_candidate = cand_idx
                    best_gain = -neg_gain
                    break
                
                if best_candidate == -1:
                    break
                
                # Select candidate
                picked.add(best_candidate)
                code.append(list(all_words[best_candidate]))
                
                # Update coverage and gains
                for idx in spheres[best_candidate]:
                    if not covered[idx]:
                        covered[idx] = True
                        uncovered_count -= 1
                        
                        # Update gain for all candidates that cover idx
                        for cand in reverse_spheres[idx]:
                            if cand not in picked:
                                gain[cand] -= 1
                                heapq.heappush(heap, (-gain[cand], -symmetry_scores[cand], noise[cand], cand))
            
            # Local Search: 
            # 1. Remove redundant words
            # 2. Try to swap words to improve structure
            # Loop until no improvement
            
            improved = True
            while improved:
                improved = False
                if time.time() > deadline:
                    break
                
                # Step 1: Remove redundant words
                for i in range(len(code) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    cw_idx = word_to_idx[tuple(code[i])]
                    is_redundant = True
                    for w in spheres[cw_idx]:
                        # Is w covered by any other codeword?
                        covered_by_other = False
                        for j in range(len(code)):
                            if i == j:
                                continue
                            other_cw_idx = word_to_idx[tuple(code[j])]
                            if w in spheres[other_cw_idx]:
                                covered_by_other = True
                                break
                        if not covered_by_other:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        code.pop(i)
                        improved = True
                        # Restart scan from beginning after removal
                        break # Break inner loop, restart outer while
        
                # Step 2: Try swaps (only if no redundancy found in this pass, or always try a few)
                # To save time, only try swaps if we are stuck (no redundancy found)
                # Or just try a limited number of swaps.
                if time.time() > deadline:
                    break
                
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    if not code:
                        break
                    
                    # Pick a random word to try to replace
                    i = random.randint(0, len(code) - 1)
                    old_cw_idx = word_to_idx[tuple(code[i])]
                    
                    # Tentatively remove
                    temp_code = code[:]
                    temp_code.pop(i)
                    
                    # Find uncovered words
                    covered_temp = np.zeros(total_words, dtype=bool)
                    for cw in temp_code:
                        idx = word_to_idx(tuple(cw))
                        for nb in spheres[idx]:
                            covered_temp[nb] = True
                    
                    uncovered = np.where(~covered_temp)[0]
                    
                    if len(uncovered) == 0:
                        # It was redundant
                        code = temp_code
                        improved = True
                        break # Restart while loop
                    
                    if len(uncovered) > 13:
                        continue
                    
                    # Find a word that covers all uncovered words
                    u0 = uncovered[0]
                    found_swap = False
                    for c in reverse_spheres[u0]:
                        if c in [word_to_idx(tuple(cw)) for cw in temp_code]:
                            continue
                    
                        # Check if c covers all uncovered
                        valid = True
                        for u in uncovered:
                            if u not in spheres[c]:
                                valid = False
                                break
                        
                        if valid:
                            # Perform swap
                            code[i] = list(all_words[c])
                            found_swap = True
                            improved = True
                            break
                    
                    if found_swap:
                        break # Restart while loop to check for new redundancies
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = [c[:] for c in code]
                
                # If we found 73, we can stop early?
                if best_size <= 73:
                    break

        if best_code is not None:
            return best_code
        else:
            # Fallback: Return the last code generated (should be valid)
            # If no restart finished, return a valid code
            return [list(t) for t in all_words]
            
    else:
        # Fallback: Simple greedy for other params
        all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        
        spheres = []
        for w in all_words:
            sphere = [word_to_idx[w]]
            for i in range(n):
                for val in range(q):
                    if val == w[i]:
                        continue
                    new_w = list(w)
                    new_w[i] = val
                    sphere.append(word_to_idx[tuple(new_w)])
            spheres.append(sphere)
            
        reverse_spheres = [[] for _ in range(len(all_words))]
        for i in range(len(all_words)):
            for j in spheres[i]:
                reverse_spheres[j].append(i)
                
        covered = [False] * len(all_words)
        uncovered_count = len(all_words)
        code = []
        picked = set()
        
        gain = [0] * len(all_words)
        for i in range(len(all_words)):
            gain[i] = len(spheres[i])
            
        heap = []
        for i in range(len(all_words)):
            heap.append((-gain[i], random.random(), i))
        heapq.heapify(heap)
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            best_candidate = -1
            while heap:
                neg_gain, tb, cand_idx = heapq.heappop(heap)
                if -neg_gain != gain[cand_idx]:
                    continue
                if cand_idx in picked:
                    continue
                best_candidate = cand_idx
                break
            if best_candidate == -1:
                break
            picked.add(best_candidate)
            code.append(list(all_words[best_candidate]))
            for idx in spheres[best_candidate]:
                if not covered[idx]:
                    covered[idx] = True
                    uncovered_count -= 1
                    for cand in reverse_spheres[idx]:
                        if cand not in picked:
                            gain[cand] -= 1
                            heapq.heappush(heap, (-gain[cand], random.random(), cand))
                            
        return code
