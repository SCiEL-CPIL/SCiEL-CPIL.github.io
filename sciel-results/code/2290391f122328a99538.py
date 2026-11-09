# EVOLVE-BLOCK-START
import time
import numpy as np
import random


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet {0..q-1} with covering radius `radius`.
    The evaluator will pass q=3, n=7, radius=1 for this specific task instance.
    The goal is to minimize the number of codewords.
    """
    t_start = time.time()
    # Safety margin: return before 120s hard limit. 110s is a safe cutoff.
    deadline = 110.0

    # Total number of words in the space
    total_words = q ** n
    
    # If the space is small enough, we can use a vectorized greedy set cover approach.
    # For q=3, n=7, total_words = 2187. This is very small.
    
    # Precompute the "balls" (sets of words within radius r of each codeword candidate)
    # Since radius is likely 1 for this task, we can be efficient.
    # Let's handle general radius but optimize for radius=1.
    
    # Represent each word as an integer index 0..total_words-1
    # index = d1 * q^(n-1) + ... + dn * q^0
    
    # Function to get neighbors (words within Hamming distance <= radius)
    # For radius=1, a word is within distance 1 if it is the word itself or differs in exactly one position.
    
    if radius == 1:
        # Precompute neighbors for all words
        # neighbors[i] is a list of indices j such that dist(i, j) <= 1
        # Actually, for set cover, we need: for each candidate codeword c, which words does it cover?
        # Let's precompute `covers[c]` = set/list of word indices covered by codeword c.
        
        # It's more efficient to precompute the list of words covered by each potential codeword.
        # There are `total_words` potential codewords.
        
        covers = []
        for c_idx in range(total_words):
            # Decode c_idx to digits
            c_digits = []
            temp = c_idx
            for _ in range(n):
                c_digits.append(temp % q)
                temp //= q
            c_digits.reverse() # Now c_digits is [d1, ..., dn]
            
            covered = set()
            covered.add(c_idx)
            
            # Generate all neighbors at distance 1
            for pos in range(n):
                orig_val = c_digits[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    # Construct new word
                    new_digits = c_digits[:]
                    new_digits[pos] = new_val
                    # Encode back to index
                    new_idx = 0
                    for d in new_digits:
                        new_idx = new_idx * q + d
                    covered.add(new_idx)
            
            covers.append(list(covered))
        
        # Now we have a list of lists: covers[i] is the list of word indices covered by candidate i.
        # We want to select the minimum number of candidates such that their union is all words.
        
        # Greedy Set Cover Algorithm with multiple restarts and redundancy removal.
        # Since the space is small (2187), we can do this in numpy for speed.
        
        # Convert covers to a boolean matrix? 2187 x 2187 is ~4.7M booleans. ~5MB. This is fine.
        # But iterating over it in Python is slow. Let's use numpy.
        
        # Actually, a smarter approach for small N:
        # 1. Start with an empty set of chosen codewords.
        # 2. Maintain a set of uncovered words.
        # 3. At each step, pick the codeword that covers the most uncovered words.
        # 4. Remove covered words from the uncovered set.
        # 5. Repeat until all covered.
        
        # To speed this up, we can use numpy arrays.
        # `uncovered` is a boolean array of size total_words.
        # `covers` is a list of lists.
        
        # Better: Precompute a matrix M where M[i, j] = 1 if candidate i covers word j.
        # M is total_words x total_words.
        M = np.zeros((total_words, total_words), dtype=bool)
        for i in range(total_words):
            M[i, covers[i]] = True
        
        # Greedy Set Cover
        def greedy_set_cover(M, num_restarts=50):
            best_codewords = None
            best_size = total_words + 1
            
            for restart in range(num_restarts):
                # Randomly shuffle candidate order to break ties and explore different paths
                # Or, just run greedy with a random initial selection of a few?
                # Standard greedy is deterministic if ties are broken by index.
                # To get diversity, we can randomize the tie-breaking.
                
                uncovered = np.ones(total_words, dtype=bool)
                chosen = []
                
                # We can speed up by maintaining a count of how many uncovered words each candidate covers.
                # counts[i] = sum(M[i, j] for j where uncovered[j] is True)
                counts = M.sum(axis=1)
                
                while uncovered.any():
                    # Find candidate with max count
                    # If counts are all 0, we have a problem (should not happen if M is correct)
                    max_idx = np.argmax(counts)
                    
                    if counts[max_idx] == 0:
                        break # Should not happen
                    
                    chosen.append(max_idx)
                    
                    # Update uncovered
                    uncovered[M[max_idx]] = False
                    
                    # Update counts for remaining candidates
                    # counts[i] -= M[i, M[max_idx]] .sum()
                    # This is O(N^2) per step. N=2187. Steps ~ 186.
                    # 186 * 2187 * 2187 ~ 880M ops. Too slow in Python/NumPy loop?
                    # NumPy vectorization:
                    # new_counts = counts - (M @ M[max_idx])  # This is vectorized dot product
                    # But we only care about candidates not yet chosen?
                    # Actually, we can just recalculate or update.
                    
                    # Let's try the vectorized update:
                    # M[max_idx] is a boolean vector of length N.
                    # M @ M[max_idx] gives for each row i, the number of 1s in row i that are also 1s in the chosen row.
                    # This is exactly the number of newly covered words for each candidate i.
                    # So counts -= M @ M[max_idx]
                    
                    # However, M @ M[max_idx] creates a new array of size N.
                    # 186 steps * (N x N matrix-vector mult) = 186 * (2187 * 2187) ops?
                    # No, matrix-vector mult is O(N^2).
                    # 186 * 4.7M = 874M ops. NumPy can do this in ~1-2 seconds? Maybe 5-10s.
                    # Let's check if we can do better.
                    
                    # Alternative: Since N is small, maybe we can just recompute counts from scratch each time?
                    # counts = M[uncovered_indices].sum(axis=1)? No, M is full.
                    
                    # Let's stick with the update. It should be fast enough in NumPy.
                    counts = counts - (M.dot(M[max_idx]))
                    # Ensure counts don't go negative due to floating point? They are integers.
                    # Also, we should set counts for already chosen codewords to -1 or something to ignore them?
                    # Or just let argmax pick them? If they have 0 count, they won't be picked if there are others with >0.
                    # But if all remaining have 0, we break.
                    
                    # To be safe and slightly faster, we can mask out chosen ones.
                    # But creating a mask each time is O(N).
                    # Let's just rely on argmax. If a chosen one has 0 count, and others have 0, we break.
                    # If a chosen one has 0 count and others have >0, argmax picks the others.
                    
                    # Wait, `counts` for chosen codewords will decrease.
                    # Initially, a chosen codeword has count = size of its ball.
                    # After choosing it, we subtract its intersection with itself (size of ball).
                    # So its count becomes 0.
                    # So it won't be picked again unless all others are 0.
                    
                    if time.time() > deadline:
                        return None # Timeout during search
                    
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best_codewords = chosen[:]
                
                if best_size <= 160: # Heuristic: if we get very good, stop early?
                    break
            
            return best_codewords, best_size
        
        codeword_indices, size = greedy_set_cover(M, num_restarts=20)
        
        if codeword_indices is None:
            # Fallback if timeout
            # Just return a valid but possibly larger code
            # E.g., the seed program's output
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Convert indices back to codeword lists
        result = []
        for idx in codeword_indices:
            word = []
            temp = idx
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            word.reverse()
            result.append(word)
        
        # Optional: Post-processing to remove redundant codewords?
        # Greedy set cover doesn't necessarily produce a minimal set (no codeword is redundant).
        # We can try to remove codewords that are still covered by others.
        # This is the "redundancy removal" step.
        
        # Let's try a few passes of redundancy removal.
        # For each codeword c in result:
        #   Check if all words covered by c are also covered by (result - {c}).
        #   If yes, remove c.
        
        # This is expensive: O(K^2 * BallSize)
        # K ~ 186. BallSize ~ 15.
        # 186^2 * 15 ~ 500k ops. Very fast.
        
        # We'll do this on the final list.
        # We need to know which words are covered by each codeword.
        # We already have `covers` list.
        
        # Let's implement redundancy removal.
        def remove_redundancy(codeword_indices, covers, M):
            # codeword_indices is a list of ints
            # We want to remove any index i such that covers[i] is a subset of union of covers[j] for j != i.
            # Equivalent: M[i] is a subset of OR of M[j] for j != i.
            # Or: (M[i] & ~OR_others) is empty.
            
            chosen_set = set(codeword_indices)
            if not chosen_set:
                return codeword_indices
            
            # Precompute OR of all chosen?
            # No, we need OR of all *except* i.
            
            # Trick: 
            # Total_OR = OR of all chosen.
            # If M[i] is subset of Total_OR, it doesn't mean it's redundant.
            # We need: M[i] & ~ (OR of others) == 0.
            
            # Let's just do the check for each i.
            # For each i in chosen_set:
            #   other_or = OR of M[j] for j in chosen_set, j != i
            #   if M[i].all() <= other_or.all() (i.e., M[i] & ~other_or is empty)
            
            # To speed up, we can maintain a count of how many times each word is covered.
            # count[j] = number of chosen codewords that cover word j.
            # If for all j in covers[i], count[j] > 1, then i is redundant.
            
            counts = np.zeros(total_words, dtype=int)
            for idx in codeword_indices:
                counts[M[idx]] += 1
            
            to_remove = []
            for idx in codeword_indices:
                # Check if all words covered by idx have count > 1
                # i.e., min(counts[covers[idx]]) > 1
                # Since M[idx] is a boolean mask, we can do:
                if np.all(counts[M[idx]] > 1):
                    to_remove.append(idx)
            
            if to_remove:
                new_indices = [idx for idx in codeword_indices if idx not in to_remove]
                return remove_redundancy(new_indices, covers, M) # Recurse? Or loop?
            
            return codeword_indices
        
        # Use a loop instead of recursion to avoid stack depth issues (though K is small)
        def remove_redundancy_loop(codeword_indices, M):
            changed = True
            while changed:
                changed = False
                if not codeword_indices:
                    break
                counts = np.zeros(total_words, dtype=int)
                for idx in codeword_indices:
                    counts[M[idx]] += 1
                
                to_remove = set()
                for idx in codeword_indices:
                    if np.all(counts[M[idx]] > 1):
                        to_remove.add(idx)
                
                if to_remove:
                    codeword_indices = [idx for idx in codeword_indices if idx not in to_remove]
                    changed = True
            
            return codeword_indices
        
        final_indices = remove_redundancy_loop(codeword_indices, M)
        
        # Convert final indices to codewords
        result = []
        for idx in final_indices:
            word = []
            temp = idx
            for _ in range(n):
                word.append(temp % q)
                temp //= q
            word.reverse()
            result.append(word)
        
        return result

    else:
        # General case fallback (should not be hit for this specific task)
        # Use a simple greedy approach or seed method
        import itertools
        if q == 3 and n == 4:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return tetracode
        # Fallback to seed-like construction
        # This is just a placeholder for non-radius-1 cases
        # For the purpose of this task, radius is 1.
        # If radius > 1, we can just return a large set to be safe, or use a simpler greedy.
        # Given the task is specifically K_3(7,1), we assume radius=1.
        # If not, we return a trivial solution: all words? No, that's 2187.
        # Let's just use the seed logic if it fits, else return first few words?
        # Actually, let's just implement a basic greedy for any radius if needed.
        
        # For now, assume the specific case.
        raise NotImplementedError("Only radius=1 is optimized for this specific task instance.")

# EVOLVE-BLOCK-END
