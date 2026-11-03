import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code for the football pool problem K_q(n, radius).
    Strategy:
    1. Initialize with a structured Tetracode product code (for q=3, n=8).
    2. Extend the code using vectorized greedy set-cover to cover all words.
    3. Trim redundant codewords.
    4. Refine using local search with swap moves guided by an inverse index.
    
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    total_words = q ** n
    ball_size = 1 + n * (q - 1)

    def int_to_digits(num, length, base):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % base
            num //= base
        return digits

    def digits_to_int(digits, base):
        num = 0
        for d in reversed(digits):
            num = num * base + d
        return num

    # Precompute balls: balls[i] contains all words within Hamming distance 1 of i
    balls_list = [None] * total_words
    for i in range(total_words):
        d = int_to_digits(i, n, q)
        neighbors = [i]
        for pos in range(n):
            orig = d[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                d[pos] = new_val
                neighbors.append(digits_to_int(d, q))
            d[pos] = orig
        balls_list[i] = neighbors

    balls = np.array(balls_list, dtype=np.int32)

    # Precompute inverse index: inv_ball[w] is the set of centers whose ball contains w
    # We store as lists for speed, then convert to sets/frozensets as needed in search
    inv_ball = [None] * total_words
    for c in range(total_words):
        for w in balls_list[c]:
            if inv_ball[w] is None:
                inv_ball[w] = []
            inv_ball[w].append(c)
    
    # Convert to frozensets for fast intersection operations in local search
    for w in range(total_words):
        if inv_ball[w] is not None:
            inv_ball[w] = frozenset(inv_ball[w])

    def get_initial_code():
        """
        Generates a structured initial code using the Tetracode product construction
        for q=3, n=8. Falls back to empty list if parameters don't match.
        """
        if q == 3 and n == 8:
            # Tetracode is a [6,3,3] code over GF(3). 
            # Actually, the standard Tetracode is [6,3,3]. 
            # Wait, the inspiration used a 6-word code of length 4? 
            # Let's check the inspiration code:
            # tetracode = [[0,0,0,0], [0,1,2,1], ...] -> 6 words of length 4.
            # Product of two such codes gives 36 words of length 8.
            # This is a valid covering code seed.
            
            # The specific Tetracode used in inspiration 1:
            tetracode = [
                [0, 0, 0, 0],
                [0, 1, 2, 1],
                [0, 2, 1, 2],
                [1, 0, 1, 2],
                [1, 1, 2, 0],
                [1, 2, 0, 2],
            ]
            product_code = []
            for a in tetracode:
                for b in tetracode:
                    product_code.append(a + b)
            code_ints = list({digits_to_int(c, q) for c in product_code})
            return code_ints
        else:
            return []

    def run_greedy_extension(initial_code):
        """
        Extends the initial code using greedy set cover to ensure full coverage.
        """
        code = list(initial_code)
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=np.int32)
        
        for c in code:
            cover_count[balls[c]] += 1
            
        uncovered = (cover_count == 0)
        
        while np.any(uncovered) and time.time() < deadline - 10:
            # Calculate gain for each center
            # gain[c] = number of currently uncovered words in ball[c]
            gains = uncovered[balls].sum(axis=1)
            
            # Exclude centers already in code (though their gain is effectively 0 for *new* coverage, 
            # we can just mask them out to be safe or rely on argmax if we don't re-add)
            # Actually, if a center is in code, its unique contribution to *uncovered* set is 0 
            # because it's already added. But `uncovered` is dynamic.
            # It's safer to just find the best center not in code.
            
            # To avoid re-selecting existing centers, we can set their gain to -1
            for c in code:
                gains[c] = -1
                
            max_gain = int(np.max(gains))
            if max_gain <= 0:
                break
                
            best_center = int(np.argmax(gains))
            code.append(best_center)
            code_set.add(best_center)
            cover_count[balls[best_center]] += 1
            uncovered = (cover_count == 0)
            
        return code, cover_count

    def trim_code(code, cover_count):
        """
        Removes redundant codewords. A codeword is redundant if every word in its ball
        is covered by at least one other codeword.
        """
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            to_remove = []
            for i, c in enumerate(code):
                # If any word in ball[c] has cover_count == 1, that word is uniquely covered by c,
                # so c is NOT redundant.
                # If all words in ball[c] have cover_count > 1, c is redundant.
                if (cover_count[balls[c]] > 1).all():
                    to_remove.append(i)
            
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True

    def local_search(code, cover_count):
        """
        Local search using swap moves.
        Tries to replace a codeword c with another center j such that:
        1. All words uniquely covered by c are also covered by j.
        2. The swap minimizes the number of words that become "wasted" (covered by many codewords).
        """
        nonlocal code, cover_count
        code_set = set(code)
        rng = random.Random(42)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            indices = list(range(len(code)))
            rng.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 2:
                    return
                i = code[idx]
                bp_i = balls[i]
                
                # Find words uniquely covered by i
                unique_pts = bp_i[cover_count[bp_i] == 1]
                
                if len(unique_pts) == 0:
                    # i is redundant, remove it
                    code.pop(idx)
                    cover_count[bp_i] -= 1
                    code_set.discard(i)
                    improved = True
                    break
                
                # Find candidates j that cover all unique_pts
                # Intersection of inv_ball[w] for all w in unique_pts
                candidate_set = None
                for w in unique_pts:
                    cc = inv_ball[w]
                    if candidate_set is None:
                        candidate_set = set(cc)
                    else:
                        candidate_set &= cc
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                # Exclude current code words
                candidates = [c for c in candidate_set if c not in code_set]
                if not candidates:
                    continue
                
                # Evaluate candidates
                # We want to minimize the "waste" or maximize the "efficiency".
                # A good metric: after swap, how many words are covered by exactly 1 codeword?
                # Or simply: pick the candidate that covers the most words with low cover count.
                
                best_swap = None
                best_score = float('inf')
                
                # Limit candidates to speed up
                if len(candidates) > 50:
                    rng.shuffle(candidates)
                    candidates = candidates[:50]
                
                for j in candidates:
                    bp_j = balls[j]
                    # Simulate swap:
                    # cover_count_new = cover_count - 1 (for bp_i) + 1 (for bp_j)
                    # We care about the overlap.
                    
                    # A simple heuristic: minimize the number of words in bp_j that are already well-covered
                    # i.e., words w in bp_j where cover_count[w] is high.
                    # Or maximize the number of words in bp_j where cover_count[w] is low.
                    
                    # Let's use: sum of cover_count[w] for w in bp_j. Lower is better (less redundancy introduced).
                    score = int(np.sum(cover_count[bp_j]))
                    
                    if score < best_score:
                        best_score = score
                        best_swap = j
                
                if best_swap is not None:
                    j = best_swap
                    code[idx] = j
                    code_set.discard(i)
                    code_set.add(j)
                    cover_count[bp_i] -= 1
                    cover_count[bp_j] += 1
                    improved = True
                    break

    # 1. Get initial code
    init_code = get_initial_code()
    
    # 2. Extend with greedy
    code, cover_count = run_greedy_extension(init_code)
    
    # 3. Trim
    trim_code(code, cover_count)
    
    # 4. Local Search
    local_search(code, cover_count)
    
    # 5. Final Trim
    trim_code(code, cover_count)
    
    # Convert to list of lists
    result = [int_to_digits(c, n, q) for c in code]
    
    return result

if __name__ == "__main__":
    # Quick self-test
    code = construct(3, 8, 1)
    print(f"Code size: {len(code)}")
