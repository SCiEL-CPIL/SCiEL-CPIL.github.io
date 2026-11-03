import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112  # 112s safety margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    # Convert to base-q integers for efficiency
    def list_to_int(word):
        val = 0
        for digit in word:
            val = val * q + digit
        return val

    def int_to_list(val, n):
        word = [0] * n
        for i in range(n):
            word[i] = val % q
            val //= q
        return word

    # Precompute balls (radius-1 neighborhoods) for all q^n words
    all_words = list(range(q**n))
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
    
    # Also precompute: for each word, which codewords cover it (dual)
    # This is the same as balls, just viewed from the other side
    
    # 1. Generate initial seed: Direct sum of tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
    
    code = set(initial_codewords)
    
    # Helper: compute coverage and unique coverers
    def compute_coverage():
        coverage = {w: 0 for w in all_words}
        unique_coverer = {w: None for w in all_words}
        
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                elif coverage[w] > 1:
                    unique_coverer[w] = None
        
        return coverage, unique_coverer
    
    # 2. Batch removal of non-essential codewords
    def batch_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        coverage, unique_coverer = compute_coverage()
        
        # Find essential codewords
        essential = set()
        for w in all_words:
            if unique_coverer[w] is not None:
                essential.add(unique_coverer[w])
        
        removable = code - essential
        
        if not removable:
            return False
            
        # Remove all removable codewords at once
        code -= removable
        return True
    
    # 3. WalkSAT-style local search for minimum hitting set
    # Each space word is a clause requiring at least one incident codeword
    # We want to minimize the number of selected codewords while satisfying all clauses
    
    def walksat_step():
        nonlocal code
        if time.time() > deadline:
            return False
        
        coverage, unique_coverer = compute_coverage()
        
        # Group words by unique coverer
        unique_groups = {}
        for w in all_words:
            uc = unique_coverer[w]
            if uc is not None:
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        
        if not unique_groups:
            return False
        
        # Sort by size of unique group to find easy swaps first
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if len(U_c) == 0:
                continue
            
            # Try to find w_new that covers U_c
            candidate_ws = set()
            for w in U_c:
                candidate_ws.add(w)
                for nb in balls[w]:
                    candidate_ws.add(nb)
            
            # Filter to words not in code
            candidate_ws -= code
            
            # Check each candidate
            for w_new in candidate_ws:
                # Check if B(w_new, 1) covers U_c
                if all(w in balls[w_new] for w in U_c):
                    # Perform swap: remove c, add w_new
                    code.remove(c)
                    code.add(w_new)
                    return True
                    
        return False
    
    # 4. More aggressive: try to remove a codeword and see if coverage is maintained
    # by adding a different codeword (this is a 1-move in the space of codeword sets)
    def try_remove_and_repair():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # For each codeword, check if removing it leaves some words uncovered
        # and if those can be covered by adding one new codeword
        for c in list(code):
            # Temporarily remove c
            code.remove(c)
            
            # Find uncovered words
            covered = set()
            for c2 in code:
                covered |= balls[c2]
            
            uncovered = set(all_words) - covered
            
            if not uncovered:
                # c was not needed, keep it removed
                return True
            
            # Try to find a single codeword that covers all uncovered words
            best_w = None
            best_count = 0
            for w in all_words:
                if w in code:
                    continue
                count = len(balls[w] & uncovered)
                if count == len(uncovered):
                    best_w = w
                    break
                if count > best_count:
                    best_count = count
                    best_w = w
            
            if best_w is not None and best_count == len(uncovered):
                code.add(best_w)
                return True
            
            # Couldn't repair, put c back
            code.add(c)
        
        return False
    
    # 5. Perturbation: random restart with smaller initial code
    def perturb():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Remove a random subset of codewords and repair
        if len(code) <= 10:
            return False
        
        n_remove = random.randint(1, min(5, len(code) // 4))
        to_remove = random.sample(list(code), n_remove)
        for c in to_remove:
            code.remove(c)
        
        # Repair by adding best codewords
        for _ in range(n_remove + 2):
            if time.time() > deadline:
                break
            covered = set()
            for c in code:
                covered |= balls[c]
            uncovered = set(all_words) - covered
            if not uncovered:
                break
            best_c = None
            best_count = 0
            for w in all_words:
                if w in code:
                    continue
                count = len(balls[w] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = w
            if best_c is not None and best_count > 0:
                code.add(best_c)
            else:
                break
        
        return True
    
    # Main Loop
    while time.time() < deadline:
        made_progress = False
        
        # Try batch removal
        if batch_remove():
            made_progress = True
            continue
        
        # Try WalkSAT-style swap
        if walksat_step():
            made_progress = True
            continue
        
        # Try remove and repair
        if try_remove_and_repair():
            made_progress = True
            continue
        
        # If stuck, try random perturbation
        if len(code) > 0 and random.random() < 0.15:
            perturb()
            continue
        
        # No progress, break
        break
    
    # Final polish: try batch removal one more time
    batch_remove()
    walksat_step()
    batch_remove()
    
    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result
