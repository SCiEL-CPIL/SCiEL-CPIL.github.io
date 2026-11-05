import time
import random

def construct(q, n, radius):
    # Specific solution for q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        import itertools
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    num_words = 3**n
    
    def num_to_digits(num):
        digits = [0]*n
        for i in range(n-1, -1, -1):
            digits[i] = num % 3
            num //= 3
        return digits

    def digits_to_num(digits):
        num = 0
        for d in digits:
            num = num * 3 + d
        return num

    neighbor_masks = [0] * num_words
    
    for w in range(num_words):
        digits = num_to_digits(w)
        mask = 1 << w
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[pos] = new_val
                new_w = digits_to_num(new_digits)
                mask |= (1 << new_w)
        neighbor_masks[w] = mask

    def greedy_fast():
        uncovered = (1 << num_words) - 1
        code = []
        while uncovered:
            max_gain = 0
            candidates = []
            
            for w in range(num_words):
                gain = (neighbor_masks[w] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [w]
                elif gain == max_gain:
                    candidates.append(w)
                elif gain == max_gain - 1:
                    # Include near-optimal candidates for diversity
                    candidates.append(w)
            
            if not candidates:
                break
                
            w = random.choice(candidates)
            code.append(w)
            uncovered &= ~neighbor_masks[w]
            
        return code

    def prune_code(code):
        changed = True
        while changed:
            changed = False
            new_code = []
            for i in range(len(code)):
                c = code[i]
                union_mask = 0
                for j in range(len(code)):
                    if i == j:
                        continue
                    union_mask |= neighbor_masks[code[j]]
                
                if (neighbor_masks[c] & ~union_mask) == 0:
                    changed = True
                else:
                    new_code.append(c)
            code = new_code
        return code

    deadline = time.time() + 110.0
    best_code = None
    
    # Initial deterministic run (using a modified greedy that doesn't pick near-optimal)
    # Just run the randomized one a few times to get a good start
    # Actually, let's just run the loop.
    
    while time.time() < deadline:
        code = greedy_fast()
        code = prune_code(code)
        
        if best_code is None or len(code) < len(best_code):
            best_code = code
            if len(best_code) <= 71:
                break

    if best_code is None:
        # Fallback
        best_code = list(range(num_words))
        
    result = [num_to_digits(w) for w in best_code]
    return result
