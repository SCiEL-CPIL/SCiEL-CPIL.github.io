import numpy as np
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 8 and radius == 1:
        total_words = 3 ** n
        # Precompute neighbors
        neighbors = [[] for _ in range(total_words)]
        for i in range(total_words):
            # Convert i to base 3 digits
            digits = [0] * n
            val = i
            for j in range(n):
                digits[j] = val % 3
                val //= 3
            
            neigh = [i]
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(3):
                    if new_val == orig:
                        continue
                    digits[pos] = new_val
                    ni = 0
                    for k in range(n-1, -1, -1):
                        ni = ni * 3 + digits[k]
                    neigh.append(ni)
                digits[pos] = orig
            neighbors[i] = neigh

        # 1. Initial Greedy Construction
        uncovered = np.ones(total_words, dtype=bool)
        counts = np.full(total_words, 17, dtype=np.int32)
        code = []
        
        while uncovered.any():
            best_idx = np.argmax(counts)
            code.append(best_idx)
            for v in neighbors[best_idx]:
                if uncovered[v]:
                    uncovered[v] = False
                    for u in neighbors[v]:
                        counts[u] -= 1
                        
        # 2. Local Search: Iterative Deletion
        # Maintain coverage_count
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for v in neighbors[c]:
                coverage_count[v] += 1
                
        code_set = set(code)
        
        start_time = time.time()
        max_time = 115 # Leave 5s buffer
        
        # Try multiple passes
        # We can also try to add random codewords to escape local minima
        # But pure deletion is a good start.
        
        improved = True
        while improved and (time.time() - start_time) < max_time:
            improved = False
            # Shuffle code to avoid bias
            random.shuffle(code)
            new_code = []
            for c in code:
                # Check if c is removable
                # c is removable if for all v in neighbors[c], coverage_count[v] > 1
                # We can check this quickly
                removable = True
                for v in neighbors[c]:
                    if coverage_count[v] == 1:
                        removable = False
                        break
                
                if removable:
                    # Remove c
                    for v in neighbors[c]:
                        coverage_count[v] -= 1
                    improved = True
                else:
                    new_code.append(c)
            
            if improved:
                code = new_code
            else:
                # If no improvement in a pass, try a "kick"
                # Add a random codeword that might help, then delete again?
                # Or just stop if we've done many passes.
                # Let's try a simple kick: add a random word, then run one deletion pass.
                if len(code) < 600: # Don't grow too much
                    random_c = random.randint(0, total_words - 1)
                    if random_c not in code_set:
                        code.append(random_c)
                        code_set.add(random_c)
                        for v in neighbors[random_c]:
                            coverage_count[v] += 1
                        # Run one deletion pass immediately
                        random.shuffle(code)
                        new_code = []
                        for c in code:
                            removable = True
                            for v in neighbors[c]:
                                if coverage_count[v] == 1:
                                    removable = False
                                    break
                            if removable:
                                for v in neighbors[c]:
                                    coverage_count[v] -= 1
                                improved = True
                            else:
                                new_code.append(c)
                        code = new_code
                        # Remove from set if removed? 
                        # The set code_set is getting out of sync with code list if we remove.
                        # Let's rebuild code_set from code if we do kicks.
                        # Actually, for the kick, we just append and then delete.
                        # We need to ensure code_set is accurate if we use it.
                        # Let's simplify: don't use code_set for the kick check, just check list membership (slow) or assume random is likely not in code.
                        # Given 6561 space and ~500 code, collision is rare.
                        
                # If still no improvement after a kick, we might be stuck.
                # To prevent infinite loop of kicks, we can count kicks.
                # But the time limit handles it.
        
        # Convert back to list of lists
        def int_to_digits(val, base, length):
            res = [0] * length
            for i in range(length):
                res[i] = val % base
                val //= base
            return res

        return [int_to_digits(c, q, n) for c in code]
    else:
        # Fallback
        total_words = q ** n
        def int_to_list(val, base, length):
            res = [0] * length
            for i in range(length):
                res[i] = val % base
                val //= base
            return res

        def list_to_int(lst, base):
            val = 0
            for i in range(len(lst)-1, -1, -1):
                val = val * base + lst[i]
            return val

        neighbors = []
        for i in range(total_words):
            w = int_to_list(i, q, n)
            neigh = {i}
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    ni = list_to_int(w, q)
                    neigh.add(ni)
                w[pos] = orig
            neighbors.append(neigh)

        uncovered = set(range(total_words))
        code = []
        while uncovered:
            best_c = -1
            best_count = -1
            for c in uncovered:
                count = len(neighbors[c] & uncovered)
                if count > best_count:
                    best_count = count
                    best_c = c
            code.append(best_c)
            uncovered = uncovered - neighbors[best_c]
        return [int_to_list(c, q, n) for c in code]
