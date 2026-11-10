import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    def get_neighbors(val):
        neighbors = [val]
        for pos in range(n):
            digit = (val // (q ** pos)) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_val = val - digit * (q ** pos) + new_digit * (q ** pos)
                neighbors.append(new_val)
        return neighbors

    all_neighbors = [get_neighbors(i) for i in range(total_words)]
    
    def int_to_list(val, base, length):
        res = [0] * length
        for i in range(length):
            res[i] = val % base
            val //= base
        return res
        
    best_code = None
    best_size = float('inf')
    
    # Try multiple random seeds to find a better solution
    for seed in range(10):
        random.seed(seed)
        uncovered = set(range(total_words))
        code = []
        
        word_order = list(range(total_words))
        random.shuffle(word_order)
        
        for u in word_order:
            if u not in uncovered:
                continue
                
            candidates = all_neighbors[u]
            
            best_c = -1
            best_count = -1
            
            for c in candidates:
                count = 0
                for w in all_neighbors[c]:
                    if w in uncovered:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_c = c
            
            code.append(best_c)
            for w in all_neighbors[best_c]:
                if w in uncovered:
                    uncovered.remove(w)
                    
        # Local Search Phase: Removal
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            for w in all_neighbors[c]:
                cover_count[w] += 1
                
        changed = True
        while changed:
            changed = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                can_remove = True
                for w in all_neighbors[c]:
                    if cover_count[w] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    code.pop(i)
                    for w in all_neighbors[c]:
                        cover_count[w] -= 1
                    changed = True

        if len(code) < best_size:
            best_size = len(code)
            best_code = code.copy()

    return [int_to_list(c, q, n) for c in best_code]
