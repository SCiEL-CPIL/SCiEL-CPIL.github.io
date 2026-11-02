import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Fallback to seed logic if parameters don't match the optimized case
    if not (q == 3 and n == 8 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Optimized path for K_3(8, 1)
    total_words = 3 ** 8
    # Precompute neighbors for all 6561 words
    # Word is an integer 0..6560
    # Neighbors include the word itself (distance 0) and distance 1 words
    neighbors = [None] * total_words
    
    # Helper to convert int to base-3 digits
    def to_digits(num, length=8):
        digits = [0] * length
        for i in range(length):
            digits[i] = num % 3
            num //= 3
        return digits

    # Helper to convert base-3 digits to int
    def to_int(digits):
        num = 0
        for d in reversed(digits):
            num = num * 3 + d
        return num

    # Precompute neighbors
    for w in range(total_words):
        digits = to_digits(w)
        neigh_set = {w}
        for i in range(8):
            original = digits[i]
            for new_val in range(3):
                if new_val != original:
                    digits[i] = new_val
                    neigh_set.add(to_int(digits))
            digits[i] = original
        neighbors[w] = neigh_set

    # 1. Greedy Set Cover
    uncovered = set(range(total_words))
    code = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick an arbitrary uncovered word
        w = next(iter(uncovered))
        
        # Candidates are w and its neighbors
        candidates = neighbors[w]
        
        best_c = -1
        best_count = -1
        
        for c in candidates:
            # Count how many uncovered words are in N(c)
            count = len(neighbors[c] & uncovered)
            if count > best_count:
                best_count = count
                best_c = c
        
        if best_c != -1:
            code.append(best_c)
            uncovered.difference_update(neighbors[best_c])
            
    # 2. Pruning (Remove redundant codewords)
    # Calculate coverage counts
    coverage_count = [0] * total_words
    for c in code:
        for nb in neighbors[c]:
            coverage_count[nb] += 1
            
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        for i in range(len(code)):
            c = code[i]
            is_redundant = True
            for nb in neighbors[c]:
                if coverage_count[nb] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                # Remove c
                for nb in neighbors[c]:
                    coverage_count[nb] -= 1
                code.pop(i)
                changed = True
                break  # Restart loop as indices changed

    # 3. Local Search: 2-for-1 reduction
    # Try to replace two codewords with one
    # This is O(|C|^2) * cost of checking
    # We limit iterations to ensure we finish in time
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Create a set of current codes for fast lookup
        code_set = set(code)
        
        for i in range(len(code)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(code)):
                if time.time() > deadline:
                    break
                
                c1 = code[i]
                c2 = code[j]
                
                # Calculate words that are ONLY covered by c1 or c2 (or both)
                # i.e., words in N(c1) U N(c2) where coverage_count == 1 or 2
                # Specifically, we need to find U = (N(c1) U N(c2)) \ (Union of N(c) for c in C \ {c1, c2})
                
                # Efficiently:
                # Words in N(c1) with count 1 or 2
                # Words in N(c2) with count 1 or 2
                # But if a word is in both, and count is 2, it's covered by c1 and c2.
                # If we remove c1, c2, it becomes 0.
                
                # Let's find the set of words that will be uncovered if we remove c1 and c2.
                # A word w is uncovered if:
                # w in N(c1) or w in N(c2)
                # AND count(w) == (1 if w not in N(c1) ^ N(c2) else 2) -- wait.
                # count(w) includes c1 if w in N(c1) and c2 if w in N(c2).
                
                # Simpler:
                # Candidate new codeword c_new must cover all words w such that:
                # count(w) == 1 and w in N(c1)
                # count(w) == 1 and w in N(c2)
                # count(w) == 2 and w in N(c1) AND w in N(c2)
                
                # Actually, just compute U directly:
                U = set()
                for w in neighbors[c1]:
                    # If we remove c1, count decreases by 1.
                    # If current count is 1, it becomes 0 (uncovered).
                    # If current count is 2, it becomes 1 (covered by c2 or others).
                    # We need to know if c2 also covers it.
                    if coverage_count[w] == 1:
                        U.add(w)
                    elif coverage_count[w] == 2:
                        if w in neighbors[c2]:
                             # Covered by c1 and c2. If we remove both, count becomes 0.
                             U.add(w)
                             
                for w in neighbors[c2]:
                    if w in U: continue
                    if coverage_count[w] == 1:
                        U.add(w)
                    elif coverage_count[w] == 2:
                        if w in neighbors[c1]:
                             # Covered by c2 and c1.
                             U.add(w)

                if not U:
                    # Both are redundant together?
                    # Remove both
                    for nb in neighbors[c1]: coverage_count[nb] -= 1
                    for nb in neighbors[c2]: coverage_count[nb] -= 1
                    code.pop(j)
                    code.pop(i)
                    improved = True
                    break # Restart outer loops
                
                # Check if there is a c_new that covers U
                # c_new must be a neighbor of all words in U?
                # No, c_new must be such that U is a subset of N(c_new).
                # This implies c_new is at distance <= 1 from all words in U.
                # If |U| > 1, c_new is constrained.
                
                # Heuristic: Try c_new from the neighbors of the first word in U
                # Or just iterate all 6561? Too slow.
                # Candidates for c_new:
                # Must be in intersection of N(w) for all w in U.
                
                if len(U) > 0:
                    # Find intersection of neighbors of all words in U
                    # Start with neighbors of first word
                    candidate_set = set(neighbors[next(iter(U))])
                    for w in U:
                        if not candidate_set: break
                        candidate_set &= neighbors[w]
                    
                    # Filter candidates: must not be in current code (except c1, c2 which we are removing)
                    # And must be distinct from c1, c2?
                    # If c_new == c1, we didn't reduce size.
                    
                    for c_new in candidate_set:
                        if c_new in code_set and c_new != c1 and c_new != c2:
                            continue # Already in code, no gain
                        if c_new == c1 or c_new == c2:
                            continue # No gain in size (1 for 1 or 1 for 1)
                        
                        # Valid swap
                        # Remove c1, c2
                        for nb in neighbors[c1]: coverage_count[nb] -= 1
                        for nb in neighbors[c2]: coverage_count[nb] -= 1
                        # Add c_new
                        for nb in neighbors[c_new]: coverage_count[nb] += 1
                        
                        code.pop(j)
                        code.pop(i)
                        code.append(c_new)
                        
                        improved = True
                        break # Restart outer loops
                        
                    if improved: break
                
            if improved: break
    
    # Convert integers back to lists of integers
    result = []
    for c in code:
        result.append(to_digits(c))
        
    return result
