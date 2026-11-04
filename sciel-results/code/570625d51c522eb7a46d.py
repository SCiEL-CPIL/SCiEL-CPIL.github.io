import time
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit
    
    # Fallback for larger n or if q != 3 (though task implies ternary)
    # Direct sum of tetracode
    if q != 3 or n > 8:
        if n < 4:
            # Small n, brute force or simple
            # For n=1, K_3(1,1) = 3 (all words)
            # For n=2, K_3(2,1) = 3? No, ball size 1+2*2=5. 9/5 ~ 1.8. 
            # Actually for small n, we can just return all words if 3**n is small
            if 3**n <= 1000:
                return [list(word) for word in itertools.product(range(q), repeat=n)]
        # Direct sum construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Brute force for very small n
             return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific optimized path for q=3, small n (e.g., n=6)
    total_words = 3 ** n
    if total_words > 10000:
        # Fallback to direct sum if too big for bitset
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Precompute neighbors (ball of radius 1) for each word
    # word_to_int: digits [d0...dn-1] -> int
    # int_to_word: int -> digits
    def int_to_word(x):
        w = [0] * n
        for i in range(n):
            w[n - 1 - i] = x % 3
            x //= 3
        return w

    def word_to_int(w):
        x = 0
        for d in w:
            x = x * 3 + d
        return x

    # Precompute ball masks and ball lists
    ball_masks = [0] * total_words
    ball_lists = [[] for _ in range(total_words)]
    
    for x in range(total_words):
        w = int_to_word(x)
        mask = 0
        blist = []
        # Center
        blist.append(x)
        mask |= (1 << x)
        
        # Neighbors
        for i in range(n):
            orig = w[i]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[i] = new_val
                y = word_to_int(w)
                blist.append(y)
                mask |= (1 << y)
            w[i] = orig
        
        ball_masks[x] = mask
        ball_lists[x] = blist

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1
    code = []
    
    while uncovered:
        best_c = 0
        max_overlap = 0
        # Optimization: We can break early if max_overlap == popcount(uncovered)
        # But we need to find the best one.
        for c in range(total_words):
            overlap = (ball_masks[c] & uncovered).bit_count()
            if overlap > max_overlap:
                max_overlap = overlap
                best_c = c
                if max_overlap == uncovered.bit_count():
                    # Perfect coverage for remaining, can't do better
                    # But we still need to pick one. This one is as good as any.
                    # Actually, if we cover all remaining, we are done after this pick.
                    pass
        
        code.append(best_c)
        uncovered &= ~ball_masks[best_c]
        
        if time.time() > deadline:
             break

    # Pruning Phase
    # Count coverage for each point
    counts = [0] * total_words
    for c in code:
        for v in ball_lists[c]:
            counts[v] += 1

    final_code = []
    for c in code:
        is_essential = False
        # Tentatively remove c
        for v in ball_lists[c]:
            counts[v] -= 1
            if counts[v] == 0:
                is_essential = True
                break
        
        if is_essential:
            # Restore counts
            for v in ball_lists[c]:
                counts[v] += 1
            final_code.append(c)
        # else: c is redundant, counts remain decremented

    # Convert back to list of lists
    result = [int_to_word(c) for c in final_code]
    return result
