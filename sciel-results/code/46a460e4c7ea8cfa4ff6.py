import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    total_words = q ** n
    
    # Helper functions
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors for all words
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
    if radius >= 2:
        for i in range(total_words):
            first_order = set(neighbors[i])
            for j in first_order:
                for k in neighbors[j]:
                    if k not in neighbors[i]:
                        neighbors[i].append(k)
            neighbors[i] = list(set(neighbors[i]))

    # For the specific case q=3, n=6, radius=1, use algebraic structure
    if q == 3 and n == 6 and radius == 1:
        # Tetracode: 9 words of length 4 that form a perfect code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        # Direct sum gives 81 words: tetracode × {0,1,2}^2
        # We want to reduce this to ~73 words
        
        # Strategy: Partition by last 2 coordinates. For each pair (x,y) in {0,1,2}^2,
        # we have 81 words with that tail. But we can share codewords across tails.
        
        # Better approach: Use the structure that each codeword covers 13 words.
        # Total coverage needed: 729. With 73 codewords: 73*13 = 949 > 729.
        
        # Initialize with direct sum (81 words)
        initial = set()
        for h in tetracode:
            for tail in product(range(3), repeat=2):
                word = h + list(tail)
                initial.add(word_to_int(word, 3))
        
        # Now use local search to reduce
        # Coverage counting approach (much faster than bitmask for 729 elements)
        
        def get_coverage(code_set):
            """Return array of coverage counts for all 729 words."""
            cov = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                for nb in neighbors[c]:
                    cov[nb] += 1
            return cov
        
        def is_valid(code_set):
            """Check if code covers all words."""
            cov = np.zeros(total_words, dtype=np.int32)
            for c in code_set:
                for nb in neighbors[c]:
                    cov[nb] += 1
            return np.all(cov > 0)
        
        def try_remove(code_set, cov):
            """Try to remove a redundant codeword. Returns (new_code_set, new_cov, removed) or None."""
            # Find codewords where all their neighbors have coverage >= 2
            candidates = []
            for c in code_set:
                # Check if all neighbors of c have coverage > 1
                can_remove = True
                for nb in neighbors[c]:
                    if cov[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    candidates.append(c)
            
            if not candidates:
                return None
            
            # Remove one candidate (random for diversity)
            c = random.choice(candidates)
            new_code = set(code_set)
            new_code.remove(c)
            new_cov = cov.copy()
            for nb in neighbors[c]:
                new_cov[nb] -= 1
            return new_code, new_cov, c
        
        def try_swap(code_set, cov):
            """Try to swap: remove one codeword, add another to maintain coverage."""
            # First, find a codeword that can be removed (all neighbors covered >= 2)
            removable = []
            for c in code_set:
                can_remove = True
                for nb in neighbors[c]:
                    if cov[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    removable.append(c)
            
            if not removable:
                return None
            
            c_remove = random.choice(removable)
            
            # After removing c_remove, find uncovered words
            temp_cov = cov.copy()
            for nb in neighbors[c_remove]:
                temp_cov[nb] -= 1
            
            uncovered = np.where(temp_cov == 0)[0]
            if len(uncovered) == 0:
                # Just remove, no need to add
                new_code = set(code_set)
                new_code.remove(c_remove)
                return new_code, temp_cov, c_remove, None
            
            # Find a single codeword that covers all uncovered words
            best_add = -1
            best_count = 0
            for candidate in range(total_words):
                if candidate in code_set:
                    continue
                count = 0
                for nb in neighbors[candidate]:
                    if temp_cov[nb] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_add = candidate
                if count == len(uncovered):
                    break
            
            if best_add != -1 and best_count == len(uncovered):
                new_code = set(code_set)
                new_code.remove(c_remove)
                new_code.add(best_add)
                new_cov = temp_cov.copy()
                for nb in neighbors[best_add]:
                    new_cov[nb] += 1
                return new_code, new_cov, c_remove, best_add
            
            return None
        
        def try_two_swap(code_set, cov):
            """Try to remove 2 codewords and add 1 to maintain coverage."""
            # Find pairs of removable codewords
            removable = []
            for c in code_set:
                can_remove = True
                for nb in neighbors[c]:
                    if cov[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    removable.append(c)
            
            if len(removable) < 2:
                return None
            
            # Try pairs from removable
            random.shuffle(removable)
            for i in range(min(len(removable), 10)):
                for j in range(i+1, min(len(removable), 10)):
                    c1, c2 = removable[i], removable[j]
                    
                    temp_cov = cov.copy()
                    for nb in neighbors[c1]:
                        temp_cov[nb] -= 1
                    for nb in neighbors[c2]:
                        temp_cov[nb] -= 1
                    
                    uncovered = np.where(temp_cov == 0)[0]
                    if len(uncovered) == 0:
                        new_code = set(code_set)
                        new_code.remove(c1)
                        new_code.remove(c2)
                        return new_code, temp_cov, (c1, c2), None
                    
                    if len(uncovered) > 26:  # Can't cover more than 13*2=26 with one word
                        continue
                    
                    # Find a single codeword that covers all uncovered
                    best_add = -1
                    best_count = 0
                    for candidate in range(total_words):
                        if candidate in code_set:
                            continue
                        count = 0
                        for nb in neighbors[candidate]:
                            if temp_cov[nb] == 0:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_add = candidate
                        if count == len(uncovered):
                            break
                    
                    if best_add != -1 and best_count == len(uncovered):
                        new_code = set(code_set)
                        new_code.remove(c1)
                        new_code.remove(c2)
                        new_code.add(best_add)
                        new_cov = temp_cov.copy()
                        for nb in neighbors[best_add]:
                            new_cov[nb] += 1
                        return new_code, new_cov, (c1, c2), best_add
            
            return None
        
        # Main search loop
        best_code = set(initial)
        best_cov = get_coverage(best_code)
        best_size = len(best_code)
        
        current_code = set(best_code)
        current_cov = get_coverage(current_code)
        
        no_improvement_count = 0
        max_no_improvement = 50
        
        while time.time() < deadline:
            improved = False
            
            # Try removals
            for _ in range(100):
                result = try_remove(current_code, current_cov)
                if result is None:
                    break
                current_code, current_cov, _ = result
                improved = True
            
            if improved and len(current_code) < best_size:
                best_code = set(current_code)
                best_cov = get_coverage(best_code)
                best_size = len(current_code)
                no_improvement_count = 0
            elif improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Try 1-swaps
            for _ in range(50):
                result = try_swap(current_code, current_cov)
                if result is None:
                    break
                current_code, current_cov, _, _ = result
                improved = True
            
            if improved and len(current_code) < best_size:
                best_code = set(current_code)
                best_cov = get_coverage(best_code)
                best_size = len(current_code)
                no_improvement_count = 0
            
            # Try 2-swaps
            for _ in range(20):
                result = try_two_swap(current_code, current_cov)
                if result is None:
                    break
                current_code, current_cov, _, _ = result
                improved = True
            
            if improved and len(current_code) < best_size:
                best_code = set(current_code)
                best_cov = get_coverage(best_code)
                best_size = len(current_code)
                no_improvement_count = 0
            
            # Perturbation if stuck
            if no_improvement_count > max_no_improvement:
                # Random perturbation: remove 1-3 random codewords and repair
                k = random.randint(1, 3)
                if len(current_code) > k + 5:
                    to_remove = random.sample(list(current_code), k)
                    for c in to_remove:
                        current_code.remove(c)
                    current_cov = get_coverage(current_code)
                    
                    # Greedy repair
                    while True:
                        uncovered = np.where(current_cov == 0)[0]
                        if len(uncovered) == 0:
                            break
                        if time.time() > deadline - 5:
                            break
                        
                        best_add = -1
                        best_count = 0
                        for candidate in range(total_words):
                            if candidate in current_code:
                                continue
                            count = 0
                            for nb in neighbors[candidate]:
                                if current_cov[nb] == 0:
                                    count += 1
                            if count > best_count:
                                best_count = count
                                best_add = candidate
                            if best_count >= len(uncovered):
                                break
                        
                        if best_add == -1:
                            break
                        
                        current_code.add(best_add)
                        for nb in neighbors[best_add]:
                            current_cov[nb] += 1
                    
                    if len(current_code) < best_size:
                        best_code = set(current_code)
                        best_cov = get_coverage(best_code)
                        best_size = len(current_code)
                
                no_improvement_count = 0
                max_no_improvement = min(max_no_improvement + 10, 200)
        
        # Final polish on best solution
        final_code = set(best_code)
        final_cov = get_coverage(final_code)
        
        # Try more aggressive reduction
        while time.time() < deadline - 2:
            changed = False
            # Try removing any redundant codeword
            for c in list(final_code):
                can_remove = True
                for nb in neighbors[c]:
                    if final_cov[nb] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    final_code.remove(c)
                    for nb in neighbors[c]:
                        final_cov[nb] -= 1
                    changed = True
                    if len(final_code) < best_size:
                        best_code = set(final_code)
                        best_cov = get_coverage(best_code)
                        best_size = len(final_code)
            if not changed:
                break
        
        result = [int_to_word(c, n, q) for c in best_code]
        return result
    
    else:
        # General case: greedy construction
        uncovered = set(range(total_words))
        code = set()
        
        while uncovered and time.time() < deadline:
            best_center = -1
            best_count = 0
            for candidate in range(total_words):
                if candidate in code:
                    continue
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            
            if best_center == -1 or best_count == 0:
                break
            
            code.add(best_center)
            for nb in neighbors[best_center]:
                uncovered.discard(nb)
        
        # Try to reduce
        while time.time() < deadline - 5:
            improved = False
            for c in list(code):
                # Check if c is redundant
                redundant = True
                for nb in neighbors[c]:
                    covered_by_other = False
                    for other in code:
                        if other != c and nb in neighbors[other]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        redundant = False
                        break
                if redundant:
                    code.remove(c)
                    improved = True
                    break
            if not improved:
                break
        
        result = [int_to_word(c, n, q) for c in code]
        return result
