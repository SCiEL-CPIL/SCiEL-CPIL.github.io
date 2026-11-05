import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    total_words = q ** n
    # Generate all words and precompute powers
    words = []
    powers = [1] * n
    for j in range(1, n):
        powers[j] = powers[j-1] * q
    for i in range(total_words):
        w = []
        val = i
        for _ in range(n):
            w.append(val % q)
            val //= q
        words.append(tuple(w))
        
    # Precompute balls using index arithmetic
    balls = [[] for _ in range(total_words)]
    for i in range(total_words):
        balls[i].append(i)
        w = words[i]
        for pos in range(n):
            for delta in range(1, q):
                idx = i + delta * powers[pos]
                # Handle wraparound: if w[pos] + delta >= q, subtract q * powers[pos]
                if w[pos] + delta >= q:
                    idx -= q * powers[pos]
                balls[i].append(idx)
                
    # Convert balls to bitmasks (Python ints)
    masks = [0] * total_words
    for i in range(total_words):
        m = 0
        for idx in balls[i]:
            m |= (1 << idx)
        masks[i] = m
        
    full_mask = (1 << total_words) - 1
    
    # Helper: Greedy Set Cover
    def greedy_cover(initial_code=None):
        if initial_code is None:
            code = []
            covered = 0
        else:
            code = list(initial_code)
            covered = 0
            for c in code:
                covered |= masks[c]
                
        uncovered = full_mask ^ covered
        
        while uncovered != 0:
            best_gain = -1
            best_idx = -1
            for i in range(total_words):
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            code.append(best_idx)
            covered |= masks[best_idx]
            uncovered = full_mask ^ covered
            if time.time() > deadline:
                break
        return code

    # Helper: Local Search (Remove Redundant) - only check ball words
    def local_search(code):
        cover_count = [0] * total_words
        for c in code:
            for idx in balls[c]:
                cover_count[idx] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Only need to check words in balls[c]
                min_count = min(cover_count[idx] for idx in balls[c])
                if min_count >= 2:
                    for idx in balls[c]:
                        cover_count[idx] -= 1
                    code.pop(i)
                    improved = True
                    break
        
        return code

    # Initial solutions
    # 1. Tetracode construction (81 words)
    tetracode_words = []
    for a in range(3):
        for b in range(3):
            w = [a, b, (a+b)%3, (a+2*b)%3]
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        for f in range(3):
                            full_w = w + [c, d, e, f]
                            idx = full_w[0] + full_w[1]*3 + full_w[2]*9 + full_w[3]*27 + full_w[4]*81 + full_w[5]*243
                            tetracode_words.append(idx)
                
    best_code = greedy_cover(tetracode_words)
    best_code = local_search(best_code)
    best_size = len(best_code)
    
    # 2. Pure Greedy
    code2 = greedy_cover()
    code2 = local_search(code2)
    if len(code2) < best_size:
        best_code = code2
        best_size = len(code2)
        
    # 3. Add-Remove Search with proper revert
    code = list(best_code)
    cover_count = [0] * total_words
    for c in code:
        for idx in balls[c]:
            cover_count[idx] += 1
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # 1. Try to remove redundant
        removed = False
        for i in range(len(code)):
            c = code[i]
            min_count = min(cover_count[idx] for idx in balls[c])
            if min_count >= 2:
                for idx in balls[c]:
                    cover_count[idx] -= 1
                code.pop(i)
                removed = True
                no_improve_count = 0
                break
        
        if not removed:
            # 2. Try Add-Remove with proper state save/revert
            added_improved = False
            for _ in range(20):
                if time.time() > deadline:
                    break
                c_add = random.randint(0, total_words - 1)
                if c_add in code:
                    continue
                    
                # Save state
                saved_code = list(code)
                saved_cover = list(cover_count)
                
                # Add c_add
                code.append(c_add)
                for idx in balls[c_add]:
                    cover_count[idx] += 1
                    
                # Try to remove redundant words
                size_before = len(code)
                to_remove = []
                for i in range(len(code)):
                    c = code[i]
                    min_count = min(cover_count[idx] for idx in balls[c])
                    if min_count >= 2:
                        to_remove.append(i)
                        
                if to_remove:
                    to_remove.sort(reverse=True)
                    for i in to_remove:
                        c = code[i]
                        for idx in balls[c]:
                            cover_count[idx] -= 1
                        code.pop(i)
                        
                    if len(code) < size_before:
                        added_improved = True
                        no_improve_count = 0
                        break
                    else:
                        # Revert
                        code = saved_code
                        cover_count = saved_cover
                        
                if not added_improved:
                    # Revert
                    code = saved_code
                    cover_count = saved_cover
        
            if not added_improved:
                no_improve_count += 1
            else:
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)

    # Final polish
    best_code = local_search(best_code)
    
    # Convert indices back to words
    result = []
    for idx in best_code:
        result.append(list(words[idx]))
        
    return result
