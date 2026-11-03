import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit
    
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
    total = q ** n
    all_words = list(range(total))
    balls = {}
    ball_lists = {}  # list version for faster iteration
    for w in all_words:
        w_list = int_to_list(w, n)
        b = [w]
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.append(list_to_int(new_word))
        balls[w] = set(b)
        ball_lists[w] = b
    
    # Precompute: for each word, its neighbors (words at distance <= 1)
    # This is the same as balls, used for coverage checks
    
    # 1. Generate initial seed: Direct sum of tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
    
    code = set(initial_codewords)
    
    # Fast coverage computation using numpy for speed
    def compute_coverage_fast():
        """Returns (coverage_counts, unique_coverer) arrays of size total."""
        coverage = np.zeros(total, dtype=np.int32)
        unique_coverer = np.full(total, -1, dtype=np.int32)
        
        for c in code:
            for w in ball_lists[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                else:
                    unique_coverer[w] = -1
        
        return coverage, unique_coverer
    
    # 2. Batch removal of non-essential codewords
    def batch_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        coverage, unique_coverer = compute_coverage_fast()
        
        # Find essential codewords
        essential = set()
        for w in range(total):
            if unique_coverer[w] >= 0:
                essential.add(int(unique_coverer[w]))
        
        removable = code - essential
        
        if not removable:
            return False
            
        code -= removable
        return True
    
    # 3. Try to remove codeword c and replace with best single codeword
    def try_swap_remove(c):
        """Try removing c and adding one codeword that covers all words c uniquely covers."""
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Find words uniquely covered by c
        coverage, unique_coverer = compute_coverage_fast()
        U_c = [w for w in range(total) if unique_coverer[w] == c]
        
        if not U_c:
            # c is not uniquely covering anything, just remove it
            code.remove(c)
            return True
        
        # Find best replacement: codeword whose ball covers all of U_c
        # Candidate replacements are in the union of balls of words in U_c
        candidate_set = set()
        for w in U_c:
            for nb in ball_lists[w]:
                if nb not in code:
                    candidate_set.add(nb)
        
        # Also include words at distance 1 from U_c words
        for w in U_c:
            candidate_set.add(w)
        
        candidate_set -= code
        
        best_candidate = None
        best_covered = 0
        
        for w_new in candidate_set:
            covered = 0
            for u in U_c:
                if u in balls[w_new]:
                    covered += 1
            if covered == len(U_c):
                best_candidate = w_new
                break
            if covered > best_covered:
                best_covered = covered
                best_candidate = w_new
        
        if best_candidate is not None and best_covered == len(U_c):
            code.remove(c)
            code.add(best_candidate)
            return True
        
        return False
    
    # 4. Try removing 2 codewords and adding 1 (2-for-1 swap)
    def try_two_for_one():
        nonlocal code
        if time.time() > deadline:
            return False
        
        code_list = list(code)
        if len(code_list) < 2:
            return False
        
        # Sample pairs to try (not all pairs for speed)
        n_pairs_to_try = min(50, len(code_list) * (len(code_list) - 1) // 2)
        
        # Prefer pairs that are close (Hamming distance <= 2)
        random.shuffle(code_list)
        tried = set()
        
        for _ in range(n_pairs_to_try):
            idx1 = random.randint(0, len(code_list) - 1)
            idx2 = random.randint(0, len(code_list) - 1)
            if idx1 == idx2:
                continue
            c1, c2 = code_list[idx1], code_list[idx2]
            pair_key = (min(c1, c2), max(c1, c2))
            if pair_key in tried:
                continue
            tried.add(pair_key)
            
            # Find words uniquely covered by c1 or c2
            coverage, unique_coverer = compute_coverage_fast()
            U = set()
            for w in range(total):
                if unique_coverer[w] == c1 or unique_coverer[w] == c2:
                    U.add(w)
            
            if not U:
                # Neither is essential, remove both
                code.discard(c1)
                code.discard(c2)
                return True
            
            # Find best single replacement
            candidate_set = set()
            for w in U:
                for nb in ball_lists[w]:
                    if nb not in code:
                        candidate_set.add(nb)
            
            best_candidate = None
            best_covered = 0
            for w_new in candidate_set:
                covered = 0
                for u in U:
                    if u in balls[w_new]:
                        covered += 1
                if covered == len(U):
                    best_candidate = w_new
                    break
                if covered > best_covered:
                    best_covered = covered
                    best_candidate = w_new
            
            if best_candidate is not None and best_covered == len(U):
                code.discard(c1)
                code.discard(c2)
                code.add(best_candidate)
                return True
        
        return False
    
    # 5. Greedy repair: add codewords to cover uncovered words
    def greedy_repair():
        nonlocal code
        if time.time() > deadline:
            return False
        
        # Find uncovered words
        covered = np.zeros(total, dtype=bool)
        for c in code:
            for w in ball_lists[c]:
                covered[w] = True
        
        uncovered = [w for w in range(total) if not covered[w]]
        
        if not uncovered:
            return False
        
        added = 0
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find best codeword to add
            best_c = None
            best_count = 0
            for w in all_words:
                if w in code:
                    continue
                count = sum(1 for u in uncovered if u in balls[w])
                if count > best_count:
                    best_count = count
                    best_c = w
            
            if best_c is None or best_count == 0:
                break
            
            code.add(best_c)
            added += 1
            
            # Update uncovered
            new_uncovered = []
            for u in uncovered:
                if u not in balls[best_c]:
                    new_uncovered.append(u)
            uncovered = new_uncovered
        
        return added > 0
    
    # 6. Perturbation: remove random codewords and repair
    def perturb():
        nonlocal code
        if time.time() > deadline:
            return False
        
        if len(code) <= 10:
            return False
        
        n_remove = random.randint(2, min(6, len(code) // 3))
        to_remove = random.sample(list(code), n_remove)
        for c in to_remove:
            code.remove(c)
        
        greedy_repair()
        return True
    
    # Main search loop with multiple strategies
    best_size = len(code)
    best_code = set(code)
    
    stall_count = 0
    max_stall = 20
    
    while time.time() < deadline:
        # Track best
        if len(code) < best_size:
            best_size = len(code)
            best_code = set(code)
            stall_count = 0
        else:
            stall_count += 1
        
        if stall_count > max_stall:
            # Try perturbation
            if random.random() < 0.3:
                perturb()
                stall_count = 0
            else:
                break
        
        # Try batch removal
        if batch_remove():
            continue
        
        # Try single swap
        if time.time() > deadline:
            break
        code_list = list(code)
        random.shuffle(code_list)
        swap_done = False
        for c in code_list:
            if time.time() > deadline:
                break
            if try_swap_remove(c):
                swap_done = True
                break
        if swap_done:
            continue
        
        # Try 2-for-1
        if time.time() > deadline:
            break
        if try_two_for_one():
            continue
        
        # Try perturbation
        if random.random() < 0.1:
            perturb()
            continue
        
        # No progress
        break
    
    # Restore best if current is worse
    if len(code) > best_size:
        code = best_code
    
    # Final polish
    batch_remove()
    code_list = list(code)
    random.shuffle(code_list)
    for c in code_list:
        if time.time() > deadline - 5:
            break
        try_swap_remove(c)
    batch_remove()
    
    # Convert back to list of lists
    result = [int_to_list(c, n) for c in code]
    return result
