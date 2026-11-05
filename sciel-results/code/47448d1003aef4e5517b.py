import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # This solution is specifically optimized for q=3, n=8, radius=1
    # But we must handle the general interface.
    if q != 3 or n != 8 or radius != 1:
        # Fallback to a generic greedy for other parameters
        # (Though the task implies this specific problem)
        total_words = q ** n
        digits_of = [None] * total_words
        for i in range(total_words):
            d = []
            x = i
            for _ in range(n):
                d.append(x % q)
                x //= q
            d.reverse()
            digits_of[i] = tuple(d)
        
        neighbors = [[] for _ in range(total_words)]
        for i in range(total_words):
            d = digits_of[i]
            neigh = [i]
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * (q ** (n - 1 - pos))
                    idx = i + delta
                    neigh.append(idx)
            neighbors[i] = neigh
            
        uncovered = [1] * total_words
        code = []
        while any(uncovered) and time.time() < deadline - 5.0:
            best_idx = -1
            max_cover = 0
            for i in range(total_words):
                gain = 0
                for idx in neighbors[i]:
                    if uncovered[idx]:
                        gain += 1
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
            if max_cover == 0:
                break
            code.append(best_idx)
            for idx in neighbors[best_idx]:
                uncovered[idx] = 0
                
        return [list(digits_of[i]) for i in code]

    # Specific optimization for q=3, n=8, r=1
    total_words = 3 ** 8
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_digit = d[pos]
            for new_digit in range(q):
                if new_digit == orig_digit:
                    continue
                delta = (new_digit - orig_digit) * (q ** (n - 1 - pos))
                idx = i + delta
                neigh.append(idx)
        neighbors[i] = neigh

    # 1. Algebraic Initialization: Tetracode Product
    # The Tetracode (length 4, q=3, r=1) has 9 codewords.
    # We hardcode a known valid Tetracode.
    # Verification:
    # 0000, 1111, 2222
    # 0121, 1202, 2010
    # 0212, 2100, 1021
    tetracode_digits = [
        (0,0,0,0), (1,1,1,1), (2,2,2,2),
        (0,1,2,1), (1,2,0,2), (2,0,1,0),
        (0,2,1,2), (2,1,0,0), (1,0,2,1)
    ]
    
    # Convert Tetracode digits to indices in 4-space (0..80)
    tetracode_indices_4 = []
    for d in tetracode_digits:
        idx = 0
        for k in range(4):
            idx = idx * 3 + d[k]
        tetracode_indices_4.append(idx)
        
    # Construct initial code C0 = T x T
    # Index in 8-space: idx8 = idx4_first * 81 + idx4_second
    initial_code_indices = []
    for i1 in tetracode_indices_4:
        for i2 in tetracode_indices_4:
            idx8 = i1 * 81 + i2
            initial_code_indices.append(idx8)
            
    # 2. Greedy Hole Filling
    # Initialize coverage counts
    cover_count = [0] * total_words
    code_set = set(initial_code_indices)
    
    for c in code_set:
        for idx in neighbors[c]:
            cover_count[idx] += 1
            
    # List of uncovered words
    # We will maintain a list of indices where cover_count == 0
    # To speed up greedy, we can just iterate all words and check cover_count == 0
    # But updating cover_count is O(17).
    
    # Greedy Loop
    # We want to add words that cover the most uncovered words.
    # Since the initial code is structured, the uncovered words are "holes".
    
    while time.time() < deadline - 10.0:
        # Check if all covered
        # Optimization: Keep a count of uncovered words
        # For simplicity, we'll just check if min(cover_count) > 0? No, that's O(N).
        # Let's maintain uncovered_count.
        
        # Actually, let's just run a few iterations of greedy.
        # Find best candidate
        best_gain = 0
        best_idx = -1
        
        # To speed this up, we can restrict candidates to those that cover at least 1 new word.
        # But iterating 6561 words is fast.
        for i in range(total_words):
            if i in code_set:
                continue
            gain = 0
            for idx in neighbors[i]:
                if cover_count[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_gain == 0:
            break
            
        # Add best_idx
        code_set.add(best_idx)
        for idx in neighbors[best_idx]:
            if cover_count[idx] == 0:
                # It was uncovered
                pass
            cover_count[idx] += 1
            
    # 3. Local Search: Remove Redundant
    # Now we have a valid cover. Minimize size.
    # Try to remove any word that doesn't increase uncovered count.
    
    final_code = list(code_set)
    
    # Rebuild cover counts for final_code to be safe
    cover_count_final = [0] * total_words
    for c in final_code:
        for idx in neighbors[c]:
            cover_count_final[idx] += 1
            
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Shuffle to avoid bias
        random.shuffle(final_code)
        for i in range(len(final_code)):
            c = final_code[i]
            can_remove = True
            for idx in neighbors[c]:
                if cover_count_final[idx] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                final_code.pop(i)
                for idx in neighbors[c]:
                    cover_count_final[idx] -= 1
                improved = True
                break # Restart loop to pick up new redundancies
    
    # Convert to list of lists
    return [list(digits_of[i]) for i in final_code]
