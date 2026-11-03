import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # 1. Generate initial seed: Direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Convert to base-3 integers for efficiency
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * 3 + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n):
            word[i] = val % 3
            val //= 3
        return word

    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
    
    code = set(initial_codewords)
    
    # 2. Precompute balls for all 3^n words?
    # 3^6 = 729 words. 729 * 13 neighbors is small.
    # We can precompute the ball of radius 1 for every possible word.
    all_words = list(range(3**n))
    balls = {}
    for w in all_words:
        b = [w]
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.append(list_to_int(new_word))
        balls[w] = set(b)
    
    # 3. Greedy Removal
    # Count how many codewords cover each word
    # coverage[word] = set of codeword indices covering it
    # To save memory/time, we can just count, and if count==1, identify the unique codeword.
    
    def check_and_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Calculate coverage counts
        coverage = {w: 0 for w in all_words}
        unique_coverer = {w: None for w in all_words}
        
        # Map codeword to index for easy removal? 
        # Let's just iterate over code.
        # To make removal fast, we need to know which codeword covers a word if count==1.
        
        # Reset
        for w in all_words:
            coverage[w] = 0
            unique_coverer[w] = None
            
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                elif coverage[w] > 1:
                    unique_coverer[w] = None # No longer unique
        
        # Find removable codewords
        # A codeword c is removable if there is no word w such that unique_coverer[w] == c
        # i.e., c does not uniquely cover any word.
        
        essential = set()
        for w in all_words:
            if unique_coverer[w] is not None:
                essential.add(unique_coverer[w])
                
        removable = code - essential
        
        if not removable:
            return False
            
        # Remove one at a time, prioritizing those that cover "less" unique volume? 
        # Actually, if it's removable, it covers 0 unique words.
        # Let's just pick one randomly or the first one.
        # To maximize speed, we can remove all removable ones?
        # No, removing one might make another essential or non-essential.
        # But if c1 and c2 are both removable, removing c1 doesn't make any word have coverage 0.
        # It might make some words have coverage 1, where the unique coverer is c2?
        # If w was covered by c1, c2, c3. c1 removed -> w covered by c2, c3.
        # If w was covered by c1, c2. c1 removed -> w covered by c2.
        # So c2 might become essential.
        # So we must remove one, update, then re-evaluate.
        
        # Pick a random removable codeword
        c_to_remove = random.choice(list(removable))
        code.remove(c_to_remove)
        return True

    # 4. Local Search (1-swap)
    def try_swap():
        nonlocal code
        if time.time() > deadline:
            return False
            
        # Calculate coverage
        coverage = {w: 0 for w in all_words}
        unique_coverer = {w: None for w in all_words}
        
        for w in all_words:
            coverage[w] = 0
            unique_coverer[w] = None
            
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                elif coverage[w] > 1:
                    unique_coverer[w] = None
        
        # Find codewords that are "almost" removable
        # i.e., their unique coverage set U_c is small.
        # If U_c is empty, it's removable (handled by check_and_remove).
        # So we look for c where U_c is not empty.
        
        # Group words by unique coverer
        unique_groups = {}
        for w in all_words:
            uc = unique_coverer[w]
            if uc is not None:
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        
        # We want to remove c if we can add w_new such that U_c subset B(w_new, 1)
        # And w_new is not in code (or is in code, but then we just removed c and added existing? No, must be distinct).
        # Actually, if w_new is in code, we are just swapping c for w_new.
        
        # Iterate through codewords with small unique groups
        # Sort by size of unique group to find easy swaps first
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if len(U_c) == 0:
                continue # Should be handled by removal
            
            # Try to find w_new that covers U_c
            # w_new must be within distance 1 of all words in U_c.
            # This implies U_c must be "close".
            
            # Candidate w_new's: 
            # 1. Words in U_c
            # 2. Neighbors of words in U_c
            
            candidate_ws = set()
            for w in U_c:
                candidate_ws.add(w)
                for nb in balls[w]:
                    candidate_ws.add(nb)
            
            # Filter candidate_ws to those not in code? 
            # If w_new is in code, we are effectively doing a swap.
            # If w_new is not in code, we are adding a new codeword.
            # The problem asks for a list of codewords. Distinctness is required.
            # So w_new can be anything, as long as the final set is distinct.
            
            # Check each candidate
            for w_new in candidate_ws:
                # Check if B(w_new, 1) covers U_c
                if all(w in balls[w_new] for w in U_c):
                    # Check if adding w_new and removing c keeps all words covered.
                    # Words covered by c: B(c, 1)
                    # Words that lose coverage: B(c, 1) - (B(w_new, 1) union B(other codewords, 1))
                    # We know U_c are the words ONLY covered by c.
                    # If U_c subset B(w_new, 1), then those words are now covered by w_new.
                    # What about words covered by c AND others?
                    # They remain covered by others.
                    # So the only risk is if w_new is already in code?
                    # If w_new in code, we remove c and add w_new (which is already there). 
                    # This means we just remove c. But if c was not removable, it means U_c was not empty.
                    # If U_c subset B(w_new, 1) and w_new is in code, then U_c is covered by w_new (which is in code).
                    # But w_new is in code, so U_c should have been covered by w_new already?
                    # Wait. unique_coverer[w] == c means NO other codeword covers w.
                    # If w_new is in code and w in B(w_new, 1), then w_new covers w.
                    # So unique_coverer[w] would be w_new (or another), not c.
                    # Contradiction. 
                    # So if w_new is in code, it cannot cover any w in U_c (by definition of U_c).
                    # Therefore, the condition "U_c subset B(w_new, 1)" can only be satisfied if w_new is NOT in code?
                    # Or if my logic on unique_coverer is flawed?
                    # unique_coverer[w] = c implies coverage[w] == 1 and the one is c.
                    # If w_new is in code and w in B(w_new, 1), then coverage[w] >= 2 (c and w_new).
                    # So unique_coverer[w] would be None.
                    # So U_c would not contain w.
                    # Thus, if w_new is in code, B(w_new, 1) and U_c are disjoint.
                    # So we cannot use a w_new that is already in code to cover U_c.
                    # So w_new must be a NEW codeword.
                    
                    if w_new in code:
                        continue
                    
                    # Perform swap
                    code.remove(c)
                    code.add(w_new)
                    return True
                    
        return False

    # Main Loop
    while time.time() < deadline:
        if check_and_remove():
            continue
        if try_swap():
            continue
        
        # If stuck, try random perturbation?
        # Or break if no progress?
        # For now, just break to return best found.
        # Or try a few random restarts of the swap logic?
        # Given time constraints, simple break is safer.
        break

    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result

# EVOLVE-BLOCK-END
