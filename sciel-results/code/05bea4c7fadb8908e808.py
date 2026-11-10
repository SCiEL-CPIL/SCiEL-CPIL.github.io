import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    # Parameters
    total_words = q ** n
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute neighbors for each word
    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    neighbors = [[] for _ in range(total_words)]
    # Convert index to base-q tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[i] = idx % q
            idx //= q
        return t

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Generate all tuples
    all_tuples = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    all_indices = list(range(total_words))
    
    # Map tuple to index for fast lookup
    tuple_to_idx_map = {t: i for i, t in enumerate(all_tuples)}

    # Precompute neighbors
    for i, t1 in enumerate(all_tuples):
        for t2 in all_tuples:
            # Hamming distance
            dist = sum(1 for a, b in zip(t1, t2) if a != b)
            if dist <= radius:
                neighbors[i].append(tuple_to_idx_map[t2])
                
    # Optimization: Use numpy for coverage counts
    # covered[i] = 1 if word i is covered, 0 otherwise
    # counts[i] = number of codewords covering word i
    
    def greedy_construct(randomize_ties=False):
        covered = np.zeros(total_words, dtype=np.int8)
        counts = np.zeros(total_words, dtype=np.int16)
        code = []
        
        while np.sum(covered) < total_words:
            best_gain = -1
            best_candidates = []
            
            # We only need to consider words not already in the code?
            # Actually, adding a word already in the code is useless.
            # So candidates are all_indices minus current code.
            # For speed, just iterate all and skip if in code?
            # Or maintain a set of available candidates.
            
            # Since n=6, total_words=729, iterating all is fine.
            in_code_set = set(code)
            
            for c in all_indices:
                if c in in_code_set:
                    continue
                
                # Calculate gain: number of neighbors of c that are NOT covered
                gain = 0
                for nb in neighbors[c]:
                    if covered[nb] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain and randomize_ties:
                    best_candidates.append(c)
            
            if not best_candidates:
                break # Should not happen if not fully covered
                
            if randomize_ties and len(best_candidates) > 1:
                c = random.choice(best_candidates)
            else:
                c = best_candidates[0]
                
            code.append(c)
            in_code_set.add(c)
            
            # Update coverage
            for nb in neighbors[c]:
                counts[nb] += 1
                if counts[nb] == 1:
                    covered[nb] = 1
                    
        return code

    def local_search(code):
        """
        Tries to reduce the size of the code by removing redundant codewords
        or swapping one codeword to allow removal of another.
        Returns the improved code.
        """
        code = list(code)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to remove any redundant codeword
            for i in range(len(code)):
                c = code[i]
                # Check if all neighbors of c are covered by OTHER codewords
                # i.e., counts[nb] > 1 for all nb in neighbors[c]
                # We need to compute counts dynamically or check against the set.
                
                # Efficient check:
                # For each nb in neighbors[c], is there another codeword c' != c such that nb in neighbors[c']?
                # This is equivalent to: counts[nb] > 1, where counts is computed from the current code.
                
                # To avoid full recount, we can just check:
                redundant = True
                for nb in neighbors[c]:
                    # Check if nb is covered by any other codeword
                    is_covered_by_other = False
                    for j in range(len(code)):
                        if j == i: continue
                        if nb in neighbors[code[j]]:
                            is_covered_by_other = True
                            break
                    if not is_covered_by_other:
                        redundant = False
                        break
                
                if redundant:
                    code.pop(i)
                    improved = True
                    break # Restart loop as indices changed
            
            if improved:
                continue
                
            # 2. If no single removal, try a swap
            # Replace code[i] with new_word such that we can then remove code[j]
            # This is complex. Let's try a simpler heuristic:
            # Randomly swap one codeword with another random word, then try to remove.
            # Or, specifically:
            # For each i, remove code[i]. Let U be uncovered words.
            # Find a word w that covers U.
            # If found, replace code[i] with w.
            # Then try to remove any redundant word.
            
            swapped = False
            for i in range(len(code)):
                if time.time() > deadline: break
                c_old = code[i]
                
                # Temporarily remove c_old
                # Find uncovered words U
                # We can simulate this by checking which words are ONLY covered by c_old.
                # Let's just compute the coverage without c_old.
                
                # For speed, let's just iterate candidates for replacement.
                # A good candidate w must cover all words that were ONLY covered by c_old.
                # Let's find the set of words W_only_c_old.
                words_only_c_old = []
                for nb in neighbors[c_old]:
                    # Check if nb is covered by any other codeword
                    covered_by_other = False
                    for j in range(len(code)):
                        if j == i: continue
                        if nb in neighbors[code[j]]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        words_only_c_old.append(nb)
                
                if not words_only_c_old:
                    # c_old is redundant, but we already checked for that in step 1.
                    # So this case shouldn't happen if step 1 failed.
                    continue
                
                # Find a word w (not in code) that covers all words in words_only_c_old
                # And ideally, w should cover some other words to be useful, but primarily it must cover W_only.
                # Also, w should not be in the current code.
                
                found_swap = False
                # To speed up, restrict candidates to neighbors of words in W_only_c_old?
                # Or just all words not in code.
                in_code_set = set(code)
                
                for w in all_indices:
                    if w in in_code_set:
                        continue
                    
                    # Check if w covers all words in words_only_c_old
                    covers_all = True
                    for u in words_only_c_old:
                        if u not in neighbors[w]:
                            covers_all = False
                            break
                    
                    if covers_all:
                        # Perform swap
                        code[i] = w
                        swapped = True
                        # Now try to remove redundant words
                        # We break out of the swap loop and go back to the main 'improved' loop
                        # which will check for removals.
                        break
                
                if swapped:
                    improved = True
                    break # Restart main loop
            
        return code

    best_code = None
    best_size = float('inf')
    
    # Run multiple random restarts
    while time.time() < deadline:
        # Greedy with random tie-breaking
        code = greedy_construct(randomize_ties=True)
        
        # Local search
        code = local_search(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
            
            # If we hit the theoretical lower bound or known optimum, we can stop early?
            # Known optimum is 73. Lower bound 71.
            if best_size <= 73:
                # Try a few more to see if we can get 72 or 71? 
                # But 73 is the best known. 71 is lower bound.
                # If we find 73, it's a strong candidate.
                # Let's run a few more iterations to be sure, but 73 is likely the target.
                pass
        else:
            # If we haven't improved in a while, maybe increase randomness?
            pass

    if best_code is None:
        # Fallback to deterministic greedy
        best_code = greedy_construct(randomize_ties=False)
        best_code = local_search(best_code)

    return [list(t) for t in [tuple(t) for t in [all_tuples[i] for i in best_code]]]
