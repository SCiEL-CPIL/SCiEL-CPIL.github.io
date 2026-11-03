import itertools
import random
import time
import sys

def construct(q, n, radius):
    # Hardcoded for the specific task instance q=3, n=7, radius=1
    # but we should handle general inputs if possible, though the task implies specific params.
    # The prompt says "evaluator passes this task's alphabet size q, length n and covering radius".
    # So we must use q, n, radius.
    
    total_words = q ** n
    if total_words > 10000:
        # Fallback to direct sum for larger instances to avoid timeout
        # Tetracode construction
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic greedy without pruning for speed, or just return all words?
            # Return all words is valid but score is bad.
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Precompute neighbors
    # Word representation: integer 0..total_words-1
    # Convert int to list of digits for distance calc? 
    # Better: Precompute neighbors as list of ints.
    
    def int_to_word(w):
        digits = [0] * n
        for i in range(n):
            digits[i] = w % q
            w //= q
        return digits

    def word_to_int(digits):
        w = 0
        for i in range(n-1, -1, -1):
            w = w * q + digits[i]
        return w

    # Generate all words
    all_words = list(range(total_words))
    
    # Precompute neighbors for each word
    # N(w) includes w itself (distance 0) and distance 1
    neighbors = []
    for w in all_words:
        digits = int_to_word(w)
        neigh = [w]
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val != orig:
                    new_digits = digits[:]
                    new_digits[pos] = val
                    neigh.append(word_to_int(new_digits))
        neighbors.append(neigh)

    # Greedy + Pruning algorithm
    def greedy_prune(seed):
        random.seed(seed)
        is_uncovered = [True] * total_words
        uncovered_count = total_words
        code = []
        code_set = set()
        coverers = [[] for _ in range(total_words)]
        
        # Greedy Phase
        while uncovered_count > 0:
            best_score = 0
            best_candidates = []
            
            # Find best candidate
            for w in all_words:
                if w in code_set:
                    continue
                score = 0
                for u in neighbors[w]:
                    if is_uncovered[u]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidates = [w]
                elif score == best_score and score > 0:
                    best_candidates.append(w)
            
            if best_score == 0:
                # Should not happen if uncovered_count > 0
                break
                
            w_star = random.choice(best_candidates)
            code.append(w_star)
            code_set.add(w_star)
            
            for u in neighbors[w_star]:
                if is_uncovered[u]:
                    is_uncovered[u] = False
                    uncovered_count -= 1
                coverers[u].append(w_star)

        # Pruning Phase
        # Iterate through code, try to remove
        # We iterate backwards to handle index changes if we pop, but we use a list and filter later
        # Or just iterate over a copy
        for i in range(len(code) - 1, -1, -1):
            w = code[i]
            # Tentatively remove
            for u in neighbors[w]:
                coverers[u].remove(w)
                
            # Check if any u in N(w) is now uncovered
            valid = True
            for u in neighbors[w]:
                if len(coverers[u]) == 0:
                    valid = False
                    break
            
            if not valid:
                # Revert
                for u in neighbors[w]:
                    coverers[u].append(w)
            else:
                # Keep removed
                code.pop(i)
                code_set.remove(w)

        return code

    # Search with multiple restarts
    best_code = None
    start_time = time.time()
    time_limit = 100  # Safety margin
    
    seed = 0
    while time.time() - start_time < time_limit:
        current_code = greedy_prune(seed)
        if best_code is None or len(current_code) < len(best_code):
            best_code = current_code
            # If we reach the theoretical lower bound or a very good number, we might stop?
            # Lower bound for K_3(7,1) is 156. Best known 186.
            if len(best_code) <= 186:
                break
        seed += 1
        
    if best_code is None:
        # Fallback
        best_code = greedy_prune(0)
        
    # Convert back to list of lists
    return [int_to_word(w) for w in best_code]
