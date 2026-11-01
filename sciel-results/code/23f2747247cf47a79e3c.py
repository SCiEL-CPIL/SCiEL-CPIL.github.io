import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Only handle the specific case q=3, n=6, radius=1 efficiently
    # For other cases, fallback to seed-like or simple greedy
    if q != 3 or n != 6 or radius != 1:
        # Fallback: simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        # Map word to index for fast lookup
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        idx_to_word = {i: w for i, w in enumerate(all_words)}
        
        # Precompute neighbors
        neighbors = []
        for w in all_words:
            nb = [w]
            for i in range(n):
                for v in range(q):
                    if v != w[i]:
                        nw = list(w)
                        nw[i] = v
                        nb.append(tuple(nw))
            neighbors.append(nb)
            
        uncovered = set(range(len(all_words)))
        code = []
        
        while uncovered:
            best_idx = -1
            best_cover = 0
            for i in range(len(all_words)):
                cover = 0
                for nb in neighbors[i]:
                    if word_to_idx[nb] in uncovered:
                        cover += 1
                if cover > best_cover:
                    best_cover = cover
                    best_idx = i
            code.append(list(idx_to_word[best_idx]))
            for nb in neighbors[best_idx]:
                uncovered.discard(word_to_idx[nb])
            if time.time() > deadline:
                break
        return code

    # Specific optimized path for q=3, n=6, radius=1
    # Total words: 729
    # Represent words as integers 0..728 (base 3)
    
    def decode(x):
        return [int(x // (3**i) % 3) for i in range(6)]
        
    def encode(w):
        x = 0
        for i in range(6):
            x += w[i] * (3**i)
        return x

    all_words = list(range(729))
    
    # Precompute coverage sets for each word
    # cover[i] is a set of words covered by codeword i
    covers = [None] * 729
    for i in all_words:
        w = decode(i)
        c_set = {i}
        for pos in range(6):
            for val in range(3):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                c_set.add(encode(nw))
        covers[i] = c_set

    # Greedy Set Cover
    uncovered = set(all_words)
    code = []
    
    # To speed up greedy, we can iterate candidates.
    # 729 candidates * 729 words is small.
    
    while uncovered:
        if time.time() > deadline:
            break
        best_idx = -1
        best_count = 0
        for i in all_words:
            # Count how many words in covers[i] are in uncovered
            cnt = 0
            for u in covers[i]:
                if u in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                if best_count == len(uncovered):
                    break # Perfect cover found for remaining
        if best_idx == -1:
            break
        code.append(best_idx)
        uncovered.difference_update(covers[best_idx])
        
    # Local Search: Remove redundant codewords
    # A codeword c is redundant if its coverage is fully covered by others.
    changed = True
    while changed:
        changed = False
        new_code = []
        for c in code:
            # Check if c is redundant
            # Union of covers of all other codewords in code
            # This is O(|code|^2 * 13) which is fine for |code| < 100
            union_others = set()
            for c2 in code:
                if c2 != c:
                    union_others.update(covers[c2])
            
            if covers[c].issubset(union_others):
                changed = True
            else:
                new_code.append(c)
        code = new_code
        
    # Try to improve by swapping?
    # For now, return the greedy+reduced solution.
    # This should be well below 81.
    
    # Convert indices back to lists
    result = [decode(c) for c in code]
    
    return result
