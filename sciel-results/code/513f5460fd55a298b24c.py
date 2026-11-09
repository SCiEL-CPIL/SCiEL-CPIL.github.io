import time
import random

def construct(q, n, radius):
    # Check if we are in the specific known case
    if q == 3 and n == 7 and radius == 1:
        start_time = time.time()
        time_limit = 145  # 5s margin for 150s limit
        
        # Precomputation
        total_words = q ** n
        words = []
        for i in range(total_words):
            w = [0] * n
            val = i
            for j in range(n):
                w[j] = val % q
                val //= q
            words.append(tuple(w))
        
        word_to_idx = {w: i for i, w in enumerate(words)}
        
        neighbors = [[] for _ in range(total_words)]
        for i, w in enumerate(words):
            neighbors[i].append(i)
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_w = list(w)
                    new_w[pos] = new_val
                    new_idx = word_to_idx[tuple(new_w)]
                    neighbors[i].append(new_idx)
        
        # Greedy Cover
        def greedy_cover():
            uncovered = set(range(total_words))
            codewords = []
            while uncovered:
                # Pick a random uncovered word
                u = random.choice(list(uncovered))
                # Pick a random neighbor to cover it
                c = random.choice(neighbors[u])
                codewords.append(c)
                for w in neighbors[c]:
                    if w in uncovered:
                        uncovered.remove(w)
            return codewords

        # Check redundancy
        def is_redundant(c, codewords):
            # c is redundant if for all w in neighbors[c], there exists c' in codewords \ {c} such that w in neighbors[c']
            others = [c_ for c_ in codewords if c_ != c]
            for w in neighbors[c]:
                covered = False
                for c_ in others:
                    if w in neighbors[c_]:
                        covered = True
                        break
                if not covered:
                    return False
            return True

        def remove_redundant(codewords):
            changed = True
            while changed:
                changed = False
                for i in range(len(codewords)):
                    if is_redundant(codewords[i], codewords):
                        codewords.pop(i)
                        changed = True
                        break
            return codewords

        best_code = None
        best_size = float('inf')
        
        while time.time() - start_time < time_limit:
            code = greedy_cover()
            code = remove_redundant(code)
            
            # Local Search
            improved = True
            while improved and time.time() - start_time < time_limit:
                improved = False
                random.shuffle(code)
                for i in range(len(code)):
                    c = code[i]
                    critical = []
                    for w in neighbors[c]:
                        covered = False
                        for j in range(len(code)):
                            if j == i:
                                continue
                            if w in neighbors[code[j]]:
                                covered = True
                                break
                        if not covered:
                            critical.append(w)
                    
                    if not critical:
                        code.pop(i)
                        improved = True
                        break
                    
                    if len(critical) > 3:
                        continue # Too hard to cover with one swap likely, skip to save time?
                        
                    # Find candidates
                    candidates = set(neighbors[critical[0]])
                    for w in critical[1:]:
                        candidates &= set(neighbors[w])
                        if not candidates:
                            break
                    
                    if candidates:
                        c_new = random.choice(list(candidates))
                        if c_new not in code or c_new == c:
                            code[i] = c_new
                            new_code = remove_redundant(code)
                            if len(new_code) < len(code):
                                code = new_code
                                improved = True
                                break
                            else:
                                code[i] = c # revert
            
            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                
            if best_size <= 156:
                break

        if best_code is None:
             # Fallback to seed-like if something went wrong
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             import itertools
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

        return [list(words[i]) for i in best_code]

    else:
        # Fallback for other parameters (should not happen based on task)
        # Return a trivial solution: all words
        # Or seed logic if applicable
        import itertools
        if q == 3 and n >= 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Naive: all words
             return [list(w) for w in itertools.product(range(q), repeat=n)]

