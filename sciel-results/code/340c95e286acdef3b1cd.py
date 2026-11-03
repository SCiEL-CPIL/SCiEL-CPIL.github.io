import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
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

    total_words = q ** n
    all_words = list(range(total_words))
    
    # Precompute balls as frozensets for fast intersection
    balls = {}
    for w in all_words:
        b = {w}
        w_list = int_to_list(w, n)
        for i in range(n):
            for d in range(q):
                if d != w_list[i]:
                    new_word = w_list.copy()
                    new_word[i] = d
                    b.add(list_to_int(new_word))
        balls[w] = frozenset(b)

    # Precompute reverse mapping: for each word, which codewords cover it?
    # This is the same as balls, but we might need it for "unique coverer" logic.
    # Actually, for efficiency, we'll maintain a coverage count array dynamically.
    
    # 2. Initial Construction: Direct Sum of Tetracode
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            word = h + list(tail)
            initial_codewords.append(list_to_int(word))
            
    code = set(initial_codewords)
    
    # Helper: Compute coverage counts and unique coverers
    def compute_coverage():
        coverage = np.zeros(total_words, dtype=np.int32)
        unique_coverer = np.full(total_words, -1, dtype=np.int32)
        
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
                if coverage[w] == 1:
                    unique_coverer[w] = c
                else:
                    unique_coverer[w] = -1
        
        return coverage, unique_coverer

    # Helper: Check if code covers all words
    def is_valid():
        covered = set()
        for c in code:
            covered |= balls[c]
        return len(covered) == total_words

    # 3. Greedy Removal Phase
    def greedy_remove():
        nonlocal code
        if time.time() > deadline:
            return False
        
        coverage, unique_coverer = compute_coverage()
        
        essential = set()
        for w in range(total_words):
            if unique_coverer[w] >= 0:
                essential.add(int(unique_coverer[w]))
                
        removable = code - essential
        if removable:
            code -= removable
            return True
        return False

    # Run greedy removal until stable
    while greedy_remove():
        pass

    # 4. Local Search: Swap and Repair
    # We want to find a codeword c that can be removed, and potentially replaced by c'
    # such that coverage is maintained and size decreases or stays same.
    
    def try_swap_or_remove():
        nonlocal code
        if time.time() > deadline:
            return False
            
        coverage, unique_coverer = compute_coverage()
        
        unique_groups = {}
        for w in range(total_words):
            uc = unique_coverer[w]
            if uc >= 0:
                uc = int(uc)
                if uc not in unique_groups:
                    unique_groups[uc] = []
                unique_groups[uc].append(w)
        
        candidates = sorted(unique_groups.items(), key=lambda x: len(x[1]))
        
        for c, U_c in candidates:
            if not U_c:
                continue
            
            valid_cs = set(balls[U_c[0]])
            for w in U_c[1:]:
                valid_cs &= balls[w]
                
            candidate_new = valid_cs - code
            
            if not candidate_new:
                continue
            
            # Score each candidate by how many other unique words it covers
            # (this helps enable future removals)
            best_c_new = None
            best_score = -1
            for c_new in candidate_new:
                score = 0
                for w in balls[c_new]:
                    if unique_coverer[w] >= 0 and unique_coverer[w] != c:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_c_new = c_new
            
            if best_c_new is None:
                continue
            
            code.remove(c)
            code.add(best_c_new)
            
            greedy_remove()
            
            return True
            
        return False

    def try_remove_and_repair():
        nonlocal code
        if time.time() > deadline:
            return False
        
        if not code:
            return False
        
        c = random.choice(list(code))
        code.remove(c)
        
        # Calculate uncovered using numpy for speed
        covered_mask = np.zeros(total_words, dtype=bool)
        for c2 in code:
            for w in balls[c2]:
                covered_mask[w] = True
        uncovered = np.where(~covered_mask)[0].tolist()
        
        if not uncovered:
            return True
            
        valid_cs = set(balls[uncovered[0]])
        for w in uncovered[1:]:
            valid_cs &= balls[w]
            
        candidate_new = valid_cs - code
        
        if candidate_new:
            # Pick candidate that covers most uncovered words (should be all, but verify)
            best_c_new = None
            best_count = -1
            for c_new in candidate_new:
                count = sum(1 for w in uncovered if w in balls[c_new])
                if count > best_count:
                    best_count = count
                    best_c_new = c_new
            code.add(best_c_new)
            greedy_remove()
            return True
        else:
            code.add(c)
            return False

    # Main Loop
    best_size = len(code)
    best_code = set(code)
    
    greedy_remove()
    best_size = len(code)
    best_code = set(code)
    
    stuck_count = 0
    
    while time.time() < deadline:
        made_progress = False
        
        if try_remove_and_repair():
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)
                stuck_count = 0
            made_progress = True
            continue
        
        if try_swap_or_remove():
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)
                stuck_count = 0
            else:
                stuck_count += 1
            made_progress = True
            continue
        
        # Perturb: remove random codewords and repair greedily
        if random.random() < 0.15 or stuck_count > 20:
            n_remove = random.randint(1, 3)
            to_remove = random.sample(list(code), min(n_remove, len(code)))
            for c in to_remove:
                code.remove(c)
                
            # Greedy repair with full candidate search
            for _ in range(10):
                if time.time() > deadline:
                    break
                covered_mask = np.zeros(total_words, dtype=bool)
                for c in code:
                    for w in balls[c]:
                        covered_mask[w] = True
                uncovered = np.where(~covered_mask)[0].tolist()
                if not uncovered:
                    break
                    
                best_c = None
                best_count = 0
                for w in all_words:
                    if w in code:
                        continue
                    count = sum(1 for u in uncovered if u in balls[w])
                    if count > best_count:
                        best_count = count
                        best_c = w
                        
                if best_c is not None and best_count > 0:
                    code.add(best_c)
                else:
                    break
            
            greedy_remove()
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = set(code)
                stuck_count = 0
            made_progress = True
            continue
            
        if not made_progress:
            stuck_count += 1
            if stuck_count > 50:
                break

    # Restore best code found
    code = best_code
    greedy_remove()
    
    result = [int_to_list(c, n) for c in sorted(code)]
    return result
