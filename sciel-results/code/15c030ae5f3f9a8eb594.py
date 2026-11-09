import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    total_words = q ** n
    
    def int_to_digits(val, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = val % q
            val //= q
        return digits

    def digits_to_int(digits, q):
        val = 0
        for i in range(n):
            val = val * q + digits[i]
        return val

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        digits = int_to_digits(i, n, q)
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                digits[pos] = new_val
                neighbors[i].append(digits_to_int(digits, q))
            digits[pos] = orig_val

    best_codewords = None
    best_count = total_words
    
    # Helper to run greedy + local search from a state
    def solve_from_state(uncovered, codewords, cover_count):
        # Greedy
        while uncovered:
            u = random.choice(list(uncovered))
            candidates = neighbors[u]
            best_c = None
            max_covered = -1
            for c in candidates:
                covered = 0
                for v in neighbors[c]:
                    if v in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_c = c
            
            codewords.append(best_c)
            for v in neighbors[best_c]:
                cover_count[v] += 1
                if cover_count[v] == 1:
                    uncovered.discard(v)
        
        # Local Search
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                c = codewords[i]
                can_remove = True
                for v in neighbors[c]:
                    if cover_count[v] == 1:
                        can_remove = False
                        break
                if can_remove:
                    codewords.pop(i)
                    for v in neighbors[c]:
                        cover_count[v] -= 1
                    improved = True
                    break
        
        return codewords, cover_count, uncovered

    while time.time() < deadline:
        # Initialize
        uncovered = set(range(total_words))
        codewords = []
        cover_count = [0] * total_words
        
        # Solve
        codewords, cover_count, uncovered = solve_from_state(uncovered, codewords, cover_count)
        
        current_count = len(codewords)
        if current_count < best_count:
            best_count = current_count
            best_codewords = [int_to_digits(c, n, q) for c in codewords]
            
        # Check time for perturbation
        if time.time() > deadline - 5:
            break
            
        # Perturbation
        if len(codewords) > 10:
            perm_uncovered = set(uncovered)
            perm_codewords = list(codewords)
            perm_cover_count = list(cover_count)
            
            k = random.randint(2, 5)
            indices_to_remove = random.sample(range(len(perm_codewords)), k)
            for idx in sorted(indices_to_remove, reverse=True):
                c = perm_codewords.pop(idx)
                for v in neighbors[c]:
                    perm_cover_count[v] -= 1
                    if perm_cover_count[v] == 0:
                        perm_uncovered.add(v)
            
            # Solve perturbed state
            perm_codewords, perm_cover_count, perm_uncovered = solve_from_state(perm_uncovered, perm_codewords, perm_cover_count)
            
            if len(perm_codewords) < best_count:
                best_count = len(perm_codewords)
                best_codewords = [int_to_digits(c, n, q) for c in perm_codewords]
                # Update main state to continue from best
                codewords = perm_codewords
                cover_count = perm_cover_count
                uncovered = perm_uncovered

    if best_codewords is None:
        # Fallback: return full space (should not happen)
        return [int_to_digits(i, n, q) for i in range(total_words)]
        
    return best_codewords
