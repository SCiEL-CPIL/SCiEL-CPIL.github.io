import time
import random
import itertools
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to the seed logic for other parameters, though task is specific
        # The task says "evaluator passes this task's alphabet size...", implying fixed 3,6,1
        # But robustness is good.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Precompute all words as integers 0..728
    # word = d0 * 3^5 + d1 * 3^4 + ... + d5
    # Or simpler: base-3 representation.
    
    # Precompute balls
    # A word is represented by an integer idx.
    # To get neighbors: change one digit.
    
    N = q ** n
    
    def idx_to_digits(idx):
        digits = [0] * n
        for i in range(n):
            digits[i] = idx % q
            idx //= q
        # digits[0] is least significant (last position)
        return digits

    # Precompute balls for all N words
    # balls[i] is a set of indices covered by word i
    balls = [set() for _ in range(N)]
    
    # Powers of q for position i
    # pos 0 is least significant digit.
    # In Hamming distance, position doesn't matter, just digit value.
    # Let's generate neighbors for each idx.
    
    for idx in range(N):
        digits = idx_to_digits(idx)
        balls[idx].add(idx)
        for pos in range(n):
            original_val = digits[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                # Create new idx
                # new_idx = idx + (new_val - original_val) * (q ** pos)
                new_idx = idx + (new_val - original_val) * (q ** pos)
                balls[idx].add(new_idx)
                
    # Greedy Set Cover
    best_code = None
    best_size = N # Worst case
    
    # Run multiple trials with different random tie-breaking
    num_trials = 100
    # If time allows, increase trials.
    # Each trial is roughly:
    # Steps ~ N/13 * N * (intersection cost)
    # Intersection of sets of size 13 is fast.
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        uncovered = set(range(N))
        code = []
        
        # To avoid always picking the same first codeword, we can shuffle the order of candidates
        # or just rely on the max logic. Ties are broken by index.
        # To diversify, we can add a small random noise to the score.
        
        while uncovered:
            best_idx = -1
            best_covered = 0
            
            # We can optimize by only checking indices that are in `uncovered`?
            # No, a codeword outside `uncovered` can still cover words in `uncovered`.
            # So we must check all N indices.
            
            for idx in range(N):
                # Optimization: if idx is already in code, skip? No, we don't add duplicates.
                # But we can skip if its ball is fully covered? 
                # len(balls[idx] & uncovered)
                
                # Small optimization: if best_covered is 0, we can break early if we find 1?
                # No, we want max.
                
                # To speed up tie-breaking and diversity:
                # score = len(balls[idx] & uncovered)
                # Add small random noise to break ties randomly
                # But calculating intersection is the cost.
                
                # Let's just calculate intersection.
                # Python set intersection is fast for small sets.
                covered = len(balls[idx] & uncovered)
                
                # Add tiny random noise to break ties differently in each trial
                # noise = random.random() * 1e-6
                # score = covered + noise
                
                if covered > best_covered:
                    best_covered = covered
                    best_idx = idx
                    if covered == len(uncovered):
                        break # Found a perfect cover for remaining? Unlikely unless 1 word left.
                        
            if best_idx == -1:
                break # Should not happen if uncovered not empty
                
            code.append(best_idx)
            uncovered -= balls[best_idx]
            
        # Pruning Phase
        # Try to remove codewords
        # We need to check if removing a codeword leaves any word uncovered.
        # A word w is covered if there exists c in code such that w in balls[c].
        # Equivalently, for each w in union(balls[c] for c in code), count how many c cover it.
        # If count > 1 for all w in the union, we can remove one?
        # Actually, if we remove c, w is only covered if another c' covers it.
        
        # Efficient pruning:
        # 1. Calculate coverage count for each word in the current code's union.
        # 2. If a word is covered by only 1 codeword, that codeword is essential.
        # 3. If all words are covered by >= 2 codewords, we can try removing any codeword?
        #    Not necessarily. If we remove c, some words might drop to 1, but that's fine as long as they don't drop to 0.
        #    So: A codeword c is removable if for ALL w in balls[c], coverage_count[w] > 1.
        
        # Let's implement iterative pruning.
        # Repeat until no more removals.
        
        # Build coverage map
        # coverage_count[w] = number of codewords in code that cover w
        
        # To do this efficiently:
        # Initialize counts to 0.
        # For each c in code, for each w in balls[c], counts[w] += 1
        
        # Then for each c in code:
        #   if all(counts[w] > 1 for w in balls[c]):
        #       remove c
        #       for w in balls[c]: counts[w] -= 1
        #       restart loop (or continue carefully)
        
        # This is O(|code| * 13 * |code|) roughly.
        
        code_set = set(code)
        # We need to be careful with the order of removal.
        # Greedy removal:
        # While True:
        #   found_removal = False
        #   Calculate counts
        #   For c in code:
        #      If all(counts[w] > 1 for w in balls[c]):
        #          Remove c from code
        #          Update counts
        #          found_removal = True
        #          break # Restart check because counts changed
        #   If not found_removal: break
        
        # This might be slow if done naively.
        # Alternative:
        # Just try removing each codeword one by one and check validity.
        # Validity check:
        #   new_code = code - {c}
        #   union = union of balls[new_code]
        #   if union == set(range(N)): valid.
        
        # Since N=729, set(range(N)) is a constant set.
        # Union of sets is fast.
        
        # Let's use the "try remove" approach, it's simpler and robust.
        # To make it faster, we can maintain the union set.
        
        current_union = set()
        for c in code:
            current_union |= balls[c]
            
        # Verify full coverage
        if current_union != set(range(N)):
             # Should not happen if greedy worked, but just in case
             continue

        # Prune
        changed = True
        while changed:
            changed = False
            # Iterate over a copy because we modify code
            for i in range(len(code)):
                c = code[i]
                # Try removing c
                # We need to know if current_union \ balls[c] is still a superset of set(range(N))?
                # No, we need to know if (current_union - balls[c]) covers all words?
                # Wait, if we remove c, the new union is NOT simply current_union - balls[c].
                # Because other codewords might cover those words too.
                # So we MUST recalculate union or use counts.
                
                # Let's use counts method for pruning, it's more efficient if implemented well.
                # But "try remove and check" is O(|code| * N) per removal attempt.
                # |code| ~ 70. N=729. 70*729 ~ 50k ops.
                # 70 removals * 50k = 3.5M ops. Fast.
                
                # Let's stick to the simpler "remove and check" to avoid bugs.
                # But we need to be careful: if we remove c, we must check if the REMAINING code covers everything.
                
                # Optimization: 
                # If c is the ONLY codeword covering some word w in balls[c], we cannot remove c.
                # So we can pre-calculate "essential" words.
                
                # Let's just do the full check for simplicity first, optimize if slow.
                
                new_code = code[:i] + code[i+1:]
                
                # Check coverage
                union_check = set()
                for c2 in new_code:
                    union_check |= balls[c2]
                
                if union_check == set(range(N)):
                    code = new_code
                    changed = True
                    break # Restart loop with new code
                
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    # Convert best_code indices back to lists of digits
    # If best_code is None (shouldn't happen), fallback
    if best_code is None:
         # Fallback to seed
         tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
         return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
         
    result = []
    for idx in best_code:
        digits = idx_to_digits(idx)
        # idx_to_digits returns least significant first?
        # My idx_to_digits:
        # digits[0] = idx % 3
        # idx //= 3
        # digits[1] = idx % 3
        # ...
        # So digits[0] is position 0 (LSB).
        # The problem doesn't specify position order, just "list of n integers".
        # Consistency is key.
        result.append(digits)
        
    return result
