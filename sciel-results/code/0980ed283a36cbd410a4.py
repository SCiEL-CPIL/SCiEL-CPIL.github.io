import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Precompute neighbors
    # Words are integers 0 to q**n - 1
    # Convert int to list of digits and back
    def int_to_digits(x, n, q):
        digits = []
        for _ in range(n):
            digits.append(x % q)
            x //= q
        return digits
        
    def digits_to_int(digits, q):
        x = 0
        for d in reversed(digits):
            x = x * q + d
        return x

    total_words = q ** n
    neighbors = [[] for _ in range(total_words)]
    
    # Precompute neighbors for all words
    for i in range(total_words):
        digits = int_to_digits(i, n, q)
        # Distance 0
        neighbors[i].append(i)
        # Distance 1
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val == orig: continue
                new_digits = digits[:]
                new_digits[pos] = val
                neighbors[i].append(digits_to_int(new_digits, q))
                
    # 2. Greedy Construction
    covered = [0] * total_words  # Count of codewords covering this word
    code = []
    
    # To speed up greedy, we can just pick the word with max new coverage
    # But checking max new coverage for all 729 words every step is O(729 * 13 * steps)
    # Steps ~ 73. 729 * 13 * 73 ~ 690k ops. Very fast.
    
    uncovered_count = total_words
    while uncovered_count > 0:
        best_cand = -1
        best_gain = -1
        # We can restrict candidates to words not in code, or just all words.
        # Actually, adding a word already in code is useless.
        for c in range(total_words):
            # Optimization: if c is in code, skip? 
            # Using a set for code membership check is fast.
            if c in code: continue
            
            gain = 0
            for nb in neighbors[c]:
                if covered[nb] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cand = c
            # Early break if gain is max possible? Max gain is 13.
            # If gain == 13, break.
            if gain == 13:
                break
        
        if best_cand == -1:
            break # Should not happen if uncovered_count > 0
            
        code.append(best_cand)
        for nb in neighbors[best_cand]:
            if covered[nb] == 0:
                uncovered_count -= 1
            covered[nb] += 1
            
    # 3. Local Search / Refinement
    # We want to minimize len(code).
    # Strategy: 
    # 1. Try to remove any codeword that is redundant.
    # 2. If no redundant codeword, try to swap a "critical" codeword with a better one 
    #    that might allow future removals.
    
    code_set = set(code)
    
    def is_valid(code_list):
        # Check if code_list covers all words
        # This is expensive if done naively. 
        # We maintain 'covered' array.
        pass
        
    # We will maintain 'covered' array incrementally.
    # Current state: 'code' is a list, 'code_set' is a set, 'covered' is an array.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: Try to remove redundant codewords
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if removing c leaves any word with covered count 0
            # We need to check if any neighbor of c has covered count 1.
            # If yes, removing c makes it 0 -> invalid.
            # If no, removing c is safe.
            can_remove = True
            for nb in neighbors[c]:
                if covered[nb] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove c
                for nb in neighbors[c]:
                    covered[nb] -= 1
                code.pop(i)
                code_set.remove(c)
                improved = True
                # Restart the pass? Or continue? 
                # Restarting is safer to find other redundancies created.
                break
        
        if time.time() > deadline:
            break
            
        if not improved:
            # Pass 2: Try swaps
            # Find a codeword c such that removing it creates uncovered words U.
            # Then find c' that covers U.
            # We try to swap c with c' if c' is not in code and the new code is valid.
            # Validity check for swap:
            # Remove c -> covered' = covered - 1 for nb in N(c)
            # Add c' -> covered'' = covered' + 1 for nb in N(c')
            # We need covered''[w] > 0 for all w.
            
            # To avoid O(N^2) swaps, we pick a random critical c.
            # Or iterate through all c.
            
            swap_done = False
            for c in code:
                # Check if c is critical
                # Find U = {w | covered[w] == 1 and w in N(c)}
                U = []
                for nb in neighbors[c]:
                    if covered[nb] == 1:
                        U.append(nb)
                        
                if not U:
                    continue # c is redundant, should have been caught in Pass 1
                
                # We need a c' such that for all w in U, c' is in N(w).
                # i.e., c' is in intersection of N(w) for w in U.
                # Also, we must ensure that adding c' doesn't break coverage of words 
                # that were covered only by c' (but c' is new, so no) 
                # AND that removing c doesn't break coverage of words 
                # that are NOT in U but were covered by c?
                # Wait. If w is covered by c and covered[w] > 1, then w is not in U.
                # Removing c reduces covered[w] by 1, so it becomes > 0. Safe.
                # So the only risk is U.
                
                # So we just need c' such that c' covers all w in U.
                # And c' is not in code_set.
                
                # Find candidates c'
                # Candidates must be in N(w) for all w in U.
                # Start with N(U[0]) and intersect with N(w) for other w in U.
                
                if len(U) == 1:
                    candidates = set(neighbors[U[0]])
                else:
                    candidates = set(neighbors[U[0]])
                    for w in U[1:]:
                        candidates.intersection_update(neighbors[w])
                        if not candidates:
                            break
                
                if not candidates:
                    continue
                    
                # Remove c' candidates that are already in code
                candidates -= code_set
                
                if not candidates:
                    continue
                    
                # Pick a candidate. 
                # Heuristic: Pick the one that covers the most other "critical" words 
                # or just pick random to explore.
                # Let's pick the one that maximizes "potential" - e.g., covers most words with count 1?
                # Or just pick the first one.
                
                # Let's try a few random candidates from the set
                cand_list = list(candidates)
                random.shuffle(cand_list)
                
                for c_prime in cand_list:
                    # Check if swap c -> c_prime is valid.
                    # We know c_prime covers U.
                    # We need to ensure that for all w NOT in U, covered[w] - 1 (if w in N(c)) + 1 (if w in N(c_prime)) > 0.
                    # Actually, simpler:
                    # 1. Temporarily remove c.
                    # 2. Temporarily add c_prime.
                    # 3. Check min covered.
                    # 4. Revert if invalid.
                    
                    # Since we know U is covered by c_prime, and U was the only set with count 1 
                    # affected by c's removal...
                    # Wait. Are there other words with count 1 that are in N(c) but NOT in U?
                    # No, U is defined as {w in N(c) | covered[w] == 1}.
                    # So any w in N(c) with covered[w] > 1 is safe after removal.
                    # Any w not in N(c) is unaffected by removal.
                    # So the only words at risk are U.
                    # And c_prime covers U.
                    # So the swap is ALWAYS valid?
                    # Let's double check.
                    # covered_new[w] = covered_old[w] - 1_{w in N(c)} + 1_{w in N(c_prime)}
                    # For w in U: covered_old[w] = 1. w in N(c). w in N(c_prime).
                    # covered_new[w] = 1 - 1 + 1 = 1. OK.
                    # For w not in U:
                    #   Case 1: w in N(c). Then covered_old[w] > 1.
                    #     If w in N(c_prime): covered_new[w] = covered_old[w] - 1 + 1 = covered_old[w] > 1. OK.
                    #     If w not in N(c_prime): covered_new[w] = covered_old[w] - 1 >= 1. OK.
                    #   Case 2: w not in N(c).
                    #     If w in N(c_prime): covered_new[w] = covered_old[w] + 1 > 0. OK.
                    #     If w not in N(c_prime): covered_new[w] = covered_old[w] > 0. OK.
                    
                    # So YES, any c' that covers U is a valid swap!
                    
                    # Perform swap
                    code[code.index(c)] = c_prime
                    code_set.remove(c)
                    code_set.add(c_prime)
                    
                    # Update covered array
                    for nb in neighbors[c]:
                        covered[nb] -= 1
                    for nb in neighbors[c_prime]:
                        covered[nb] += 1
                        
                    improved = True
                    swap_done = True
                    break
                    
                if swap_done:
                    break
                    
            if time.time() > deadline:
                break
                
    return [int_to_digits(c, n, q) for c in code]
