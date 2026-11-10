import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    total_words = q ** n
    
    # Precompute neighbors as sets for fast lookup
    digits = [[0]*n for _ in range(total_words)]
    for i in range(total_words):
        x = i
        for j in range(n):
            digits[i][j] = x % q
            x //= q
            
    neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        neighbors[i].add(i)
        for pos in range(n):
            orig_val = digits[i][pos]
            for new_val in range(q):
                if new_val == orig_val: continue
                delta = (new_val - orig_val) * (q ** (n - 1 - pos))
                neighbors[i].add(i + delta)
                
    # Initial Code: Tetracode Product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    def list_to_idx(lst):
        idx = 0
        for val in lst:
            idx = idx * q + val
        return idx
        
    code_idx = set()
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n-4):
            full_word = h + list(tail)
            code_idx.add(list_to_idx(full_word))
            
    cover_count = [0] * total_words
    for c in code_idx:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def remove_redundancies():
        changed = True
        while changed:
            changed = False
            if time.time() > deadline - 5:
                return
            for c in list(code_idx):
                redundant = True
                for nb in neighbors[c]:
                    if cover_count[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    code_idx.remove(c)
                    for nb in neighbors[c]:
                        cover_count[nb] -= 1
                    changed = True
                    break
                    
    remove_redundancies()
    
    # Local Search
    while time.time() < deadline - 2:
        remove_redundancies()
        
        if not code_idx:
            break
            
        c = random.choice(list(code_idx))
        
        unique_words = [nb for nb in neighbors[c] if cover_count[nb] == 1]
        
        if not unique_words:
            continue # Should be removed, but just in case
            
        best_cand = -1
        best_score = -1
        
        # Iterate through potential new positions
        # We can restrict to words that cover at least one unique word?
        # No, must cover ALL unique words.
        
        for c_new in range(total_words):
            if c_new in code_idx:
                continue
                
            # Check if c_new covers all unique_words
            covers_all = True
            for w in unique_words:
                if w not in neighbors[c_new]:
                    covers_all = False
                    break
                    
            if not covers_all:
                continue
                
            # Score: redundancy created
            score = 0
            for nb in neighbors[c_new]:
                if cover_count[nb] > 1:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_cand = c_new
                
        if best_cand != -1:
            code_idx.remove(c)
            for nb in neighbors[c]:
                cover_count[nb] -= 1
                
            code_idx.add(best_cand)
            for nb in neighbors[best_cand]:
                cover_count[nb] += 1
                
            remove_redundancies()
            
    result = []
    for idx in code_idx:
        word = [0]*n
        x = idx
        for j in range(n-1, -1, -1):
            word[j] = x % q
            x //= q
        result.append(word)
        
    return result
