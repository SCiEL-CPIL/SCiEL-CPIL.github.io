"""Superpermutation construction with time-bounded improvement search."""
import math
# EVOLVE-BLOCK-START
import time
from itertools import permutations

def construct(n):
    start = time.time()
    DEADLINE = 145.0
    
    # Step 1: Build the classic recursive superpermutation
    def build_classic(n):
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next((o for o in range(min(len(new), len(block)), -1, -1)
                               if o == 0 or new[len(new) - o:] == block[:o]), 0)
                new += block[overlap:]
            word = new
        return word
    
    # Step 2: Verify all permutations are present
    def all_perms_present(word, n):
        target = set(permutations(range(1, n + 1)))
        found = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i + n])
            found.add(w)
            if len(found) == len(target):
                return True
        return False
    
    # Step 3: Try to improve by finding better overlap orders
    def try_improve(word, n, max_time):
        """Try to reduce length by reordering permutations with better overlaps."""
        # Extract all (n-1)-permutations in order of appearance
        n_minus_1 = n - 1
        perms_in_order = []
        seen = set()
        for i in range(len(word) - n_minus_1 + 1):
            w = tuple(word[i:i + n_minus_1])
            if len(set(w)) == n_minus_1 and w not in seen:
                seen.add(w)
                perms_in_order.append(w)
        
        if len(perms_in_order) != n_minus_1 ** n_minus_1 and len(perms_in_order) < math.factorial(n_minus_1):
            return word
        
        # Try greedy reordering: at each step, pick the next permutation
        # that maximizes overlap with current end
        def greedy_construct(perms, n):
            n_minus_1 = n - 1
            remaining = list(perms)
            result = list(remaining[0])
            result.append(n)
            result.extend(remaining[0])
            remaining.pop(0)
            
            while remaining:
                best_idx = -1
                best_overlap = 0
                current_end = tuple(result[-n_minus_1:]) if len(result) >= n_minus_1 else None
                
                for idx, p in enumerate(remaining):
                    block = list(p) + [n] + list(p)
                    if current_end is None:
                        ov = 0
                    else:
                        # Find max overlap between end of result and start of block
                        max_ov = min(len(result), len(block))
                        ov = 0
                        for o in range(max_ov, -1, -1):
                            if o == 0 or result[len(result) - o:] == block[:o]:
                                ov = o
                                break
                    if ov > best_overlap or (ov == best_overlap and idx < best_idx):
                        best_overlap = ov
                        best_idx = idx
                
                if best_idx == -1:
                    best_idx = 0
                
                p = remaining.pop(best_idx)
                block = list(p) + [n] + list(p)
                # Recompute overlap with actual result
                max_ov = min(len(result), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0 or result[len(result) - o:] == block[:o]:
                        ov = o
                        break
                result.extend(block[ov:])
            
            return result
        
        # Try greedy construction
        greedy_word = greedy_construct(perms_in_order, n)
        if len(greedy_word) < len(word) and all_perms_present(greedy_word, n):
            return greedy_word
        
        return word
    
    # Step 4: Main construction
    word = build_classic(n)
    
    # Verify
    if not all_perms_present(word, n):
        # Fallback: brute force for small n
        if n <= 4:
            target = set(permutations(range(1, n + 1)))
            result = list(range(1, n + 1))
            for p in permutations(range(1, n + 1)):
                found = False
                for i in range(len(result) - n + 1):
                    if tuple(result[i:i + n]) == p:
                        found = True
                        break
                if not found:
                    # Try to append
                    best_append = p
                    for o in range(n - 1, -1, -1):
                        if len(result) >= o and result[len(result) - o:] == list(p[:o]):
                            best_append = list(p[o:])
                            break
                    result.extend(best_append)
            return "".join(map(str, result))
        return "".join(map(str, word))
    
    # Try to improve if we have time
    elapsed = time.time() - start
    if elapsed < DEADLINE - 5.0:
        improved = try_improve(word, n, DEADLINE - 5.0)
        if all_perms_present(improved, n):
            word = improved
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
