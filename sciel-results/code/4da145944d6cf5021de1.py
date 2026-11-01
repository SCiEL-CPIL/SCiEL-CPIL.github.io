import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 0:
        return [[]]
    
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Generic fallback
    return _generic_construct(q, n, radius, deadline)


def _solve_q3_n6_r1(deadline):
    q, n, r = 3, 6, 1
    total = q ** n  # 729
    
    # Precompute all words as indices
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor lists (word + all words at Hamming distance <= 1)
    neighbors = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        neighbors[i].add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                neighbors[i].add(idx_map[tuple(nw)])
    
    # Convert to lists for speed
    neighbors_list = [list(s) for s in neighbors]
    
    # Start with tetracode direct sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = set()
    for head in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            word = head + list(tail)
            code.add(idx_map[tuple(word)])
    
    # Compute cover count
    cover_count = [0] * total
    for c in code:
        for nb in neighbors_list[c]:
            cover_count[nb] += 1
    
    def is_valid():
        return all(cover_count[i] > 0 for i in range(total))
    
    def try_remove(c, code_set, cc):
        """Try removing codeword c. Return True if successful (code still valid)."""
        # Check if any word would become uncovered
        for nb in neighbors_list[c]:
            if cc[nb] == 1:
                return False
        code_set.remove(c)
        for nb in neighbors_list[c]:
            cc[nb] -= 1
        return True
    
    def try_swap(c, new_c, code_set, cc):
        """Try replacing c with new_c. Return True if successful."""
        if new_c in code_set:
            return False
        # Words uniquely covered by c
        unique_to_c = []
        for nb in neighbors_list[c]:
            if cc[nb] == 1:
                unique_to_c.append(nb)
        
        # Check if new_c covers all unique_to_c
        for u in unique_to_c:
            if u != new_c and u not in neighbors[new_c]:
                return False
        
        # Do the swap
        code_set.remove(c)
        for nb in neighbors_list[c]:
            cc[nb] -= 1
        code_set.add(new_c)
        for nb in neighbors_list[new_c]:
            cc[nb] += 1
        return True
    
    # Local search: iteratively try to reduce code size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Pass 1: try removing each codeword
        for c in list(code):
            if time.time() > deadline:
                break
            if try_remove(c, code, cover_count):
                improved = True
        
        if time.time() > deadline:
            break
        
        # Pass 2: try swaps
        code_list = list(code)
        random.shuffle(code_list)
        for c in code_list:
            if time.time() > deadline:
                break
            # Find words uniquely covered by c
            unique_to_c = []
            for nb in neighbors_list[c]:
                if cover_count[nb] == 1:
                    unique_to_c.append(nb)
            
            if not unique_to_c:
                continue
            
            # Try to find a replacement
            # Candidate: words that cover all unique_to_c
            # Heuristic: look at neighbors of unique_to_c words
            candidates = set()
            for u in unique_to_c:
                candidates.update(neighbors_list[u])
            candidates -= code
            
            for new_c in candidates:
                if time.time() > deadline:
                    break
                if try_swap(c, new_c, code, cover_count):
                    improved = True
                    break
    
    # Final greedy repair if needed
    while not is_valid() and time.time() < deadline:
        # Find uncovered words
        uncovered = [i for i in range(total) if cover_count[i] == 0]
        if not uncovered:
            break
        
        # Find best codeword to add
        best_c = -1
        best_cnt = -1
        for c in range(total):
            if c in code:
                continue
            cnt = sum(1 for u in uncovered if u in neighbors[c])
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        
        if best_c == -1 or best_cnt == 0:
            break
        
        code.add(best_c)
        for nb in neighbors_list[best_c]:
            cover_count[nb] += 1
    
    return [list(all_words[c]) for c in sorted(code)]


def _generic_construct(q, n, radius, deadline):
    """Simple greedy for other cases."""
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    idx_map = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors
    neighbors = [set() for _ in range(total)]
    for i, w in enumerate(all_words):
        for r2 in range(radius + 1):
            for pos in itertools.combinations(range(n), r2):
                for vals in itertools.product(range(q), repeat=r2):
                    nw = list(w)
                    for p, v in zip(pos, vals):
                        nw[p] = v
                    neighbors[i].add(idx_map[tuple(nw)])
    
    neighbors_list = [list(s) for s in neighbors]
    
    code = set()
    cover_count = [0] * total
    
    while time.time() < deadline:
        uncovered = [i for i in range(total) if cover_count[i] == 0]
        if not uncovered:
            break
        
        best_c = -1
        best_cnt = -1
        for c in range(total):
            if c in code:
                continue
            cnt = sum(1 for u in uncovered if u in neighbors[c])
            if cnt > best_cnt:
                best_cnt = cnt
                best_c = c
        
        if best_c == -1 or best_cnt == 0:
            break
        
        code.add(best_c)
        for nb in neighbors_list[best_c]:
            cover_count[nb] += 1
    
    return [list(all_words[c]) for c in sorted(code)]
