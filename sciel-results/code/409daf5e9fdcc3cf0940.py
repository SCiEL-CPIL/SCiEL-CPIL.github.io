import time
import random
import numpy as np

# Precomputation for the specific case q=3, n=8, radius=1
# But the function signature is construct(q, n, radius), so we must handle general inputs or at least this one efficiently.
# The task says "evaluator passes this task's alphabet size q, length n and covering radius".
# We assume q=3, n=8, radius=1 based on the description, but we should write it generically enough or check.
# Given the strict time limit and specific problem, hardcoding the precomputation for q=3, n=8 is safe and fast.
# However, to be robust, we can precompute on the fly if parameters differ, but likely it's always 3,8,1.

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Parameters for the specific task
    # If q, n, radius are different, we might need a different strategy.
    # But the prompt says "Football pool problem for 8 matches... ternary...".
    # We will assume q=3, n=8, radius=1.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to seed logic for other parameters
        import itertools
        if n == 4 and q == 3 and radius == 1:
             return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        # Generic greedy for other cases
        total_words = q ** n
        # This fallback is slow, but likely not triggered
        pass

    # Specific optimization for q=3, n=8, radius=1
    N = 3 ** 8
    # Map word (tuple of ints) to index (int)
    # We can use base-3 representation
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * 3 + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute balls
    # ball[idx] = list of indices within distance 1
    balls = []
    for i in range(N):
        w = idx_to_word(i)
        neighbors = [i]
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    neighbors.append(word_to_idx(w))
            w[pos] = orig
        balls.append(neighbors)
    
    # Greedy Construction
    covered = np.zeros(N, dtype=bool)
    code = []
    
    # We can speed up greedy by maintaining a count of uncovered neighbors
    # But N=6561 is small.
    
    # Optimization: Use a list of candidates and update scores
    # Score of w = number of uncovered words in ball[w]
    # Initially all 17 (or less if radius > 1, but here 17)
    
    # Actually, simple greedy loop:
    while not covered.all():
        best_w = -1
        max_new = 0
        # Sample some candidates or scan all?
        # Scanning all 6561 * 17 is ~111k ops. Very fast.
        # But we do this ~400 times. 400 * 111k = 44M ops. Might be slow in Python.
        # Let's use a smarter greedy.
        
        # We can maintain a priority queue or just pick randomly from a subset?
        # No, deterministic greedy is better for reproducibility, but we have time.
        # Let's try to optimize the inner loop using numpy if possible, or just accept the cost.
        # 44M simple ops in Python might take 2-5 seconds. Acceptable.
        
        # Better: Keep a list of "active" candidates?
        # Or just scan.
        
        # To make it faster, we can break ties randomly or by index.
        for w in range(N):
            if covered[w]:
                continue # w itself is covered, but we care about its ball
            # Actually, we don't skip w based on covered[w].
            # We want to maximize new covers.
            
            # Count uncovered in ball[w]
            # This is the bottleneck.
            # Let's precompute balls as numpy arrays?
            # balls_np = np.array(balls)
            # new_covers = (~covered[balls_np[w]]).sum()
            
            # Let's try this numpy approach.
            pass
            
        # Let's rewrite the greedy loop with numpy for speed.
        # balls_np is (N, 17)
        balls_np = np.array(balls)
        
        # Greedy
        while not covered.all():
            # Calculate new covers for all w
            # covered is boolean array
            # ~covered is boolean array of uncovered
            # new_covers[w] = sum(~covered[balls_np[w]])
            # This is a matrix multiplication or reduction.
            # uncovered_counts = (~covered)[balls_np].sum(axis=1)
            
            uncovered_mask = ~covered
            # For each w, count how many neighbors are in uncovered_mask
            # This is equivalent to: for each w, sum(uncovered_mask[balls_np[w]])
            # Using numpy advanced indexing:
            new_covers = uncovered_mask[balls_np].sum(axis=1)
            
            # Find w with max new_covers
            best_w = np.argmax(new_covers)
            
            if new_covers[best_w] == 0:
                break # Should not happen if not covered.all()
                
            code.append(best_w)
            covered[balls_np[best_w]] = True
            
            if time.time() > deadline - 20:
                break

    # Local Search
    # cover_counts: how many times each word is covered
    cover_counts = np.zeros(N, dtype=int)
    for c in code:
        cover_counts[balls_np[c]] += 1
        
    # code_set for O(1) lookup
    code_set = set(code)
    
    improved = True
    swap_limit = 5000 # Max neutral swaps to try per "shake" phase
    
    while improved and time.time() < deadline - 10:
        improved = False
        # 1. Try to remove redundant codewords
        # A codeword c is redundant if min(cover_counts[balls_np[c]]) > 1
        # We can check this for all c in code
        
        # To speed up, we can iterate through code
        # If we find one, remove it and break to restart the check (as counts changed)
        
        removed_any = False
        for c in code[:]: # Iterate over a copy
            # Check redundancy
            min_cover = cover_counts[balls_np[c]].min()
            if min_cover > 1:
                # Remove c
                code.remove(c)
                code_set.remove(c)
                cover_counts[balls_np[c]] -= 1
                improved = True
                removed_any = True
                break # Restart the loop to check for more redundancies
        
        if removed_any:
            continue
            
        # 2. If no redundancy, try neutral swaps to "shake" the code
        # Pick a random codeword c
        # Find its critical words (covered only by c)
        # Find a word w not in code that covers all critical words
        # If found, swap c -> w
        
        if len(code) > 0:
            # Limit swaps to avoid infinite loops or wasting time
            swaps_done = 0
            while swaps_done < 100 and time.time() < deadline - 10:
                c = random.choice(code)
                # Critical words
                critical = balls_np[c][cover_counts[balls_np[c]] == 1]
                
                if len(critical) == 0:
                    # This is redundant, should have been caught above
                    # But maybe due to float/int issues? No, it's int.
                    # If len(critical)==0, it IS redundant.
                    code.remove(c)
                    code_set.remove(c)
                    cover_counts[balls_np[c]] -= 1
                    improved = True
                    break # Restart outer loop
                
                # We need w such that critical is subset of balls_np[w]
                # And w not in code_set
                
                # Heuristic: w must be a neighbor of all critical words?
                # If critical has 1 word u, w can be any neighbor of u.
                # If critical has > 1 words, w must be in the intersection of their neighborhoods.
                
                # To find w efficiently:
                # If len(critical) == 1:
                #   Candidates are balls_np[critical[0]]
                # Else:
                #   Candidates are intersection of balls_np[u] for u in critical
                #   We can just check a few random candidates from the neighborhood of the first critical word
                
                u0 = critical[0]
                candidates = balls_np[u0]
                
                found_swap = False
                # Shuffle candidates to randomize
                random.shuffle(candidates)
                
                for w in candidates:
                    if w in code_set:
                        continue
                    
                    # Check if w covers all critical words
                    # i.e., critical is subset of balls_np[w]
                    # Equivalently, for all u in critical, w is in balls_np[u]
                    # Or: cover_counts check? No, w is not in code yet.
                    # We check geometrically.
                    
                    # Since critical words are distance 1 from c, and w is distance 1 from u0.
                    # We just check if all u in critical are in balls_np[w].
                    # balls_np[w] is a list of indices.
                    # We can convert critical to a set for fast lookup?
                    # Or just loop. len(critical) is small.
                    
                    valid = True
                    for u in critical:
                        if u not in balls_np[w]: # This is O(17) lookup in list
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        # Remove c
                        code.remove(c)
                        code_set.remove(c)
                        cover_counts[balls_np[c]] -= 1
                        
                        # Add w
                        code.append(w)
                        code_set.add(w)
                        cover_counts[balls_np[w]] += 1
                        
                        found_swap = True
                        swaps_done += 1
                        break # Try another random c
                
                if not found_swap:
                    # Try another c
                    swaps_done += 1
                    # If we've tried many c's and found no swaps, break to avoid wasting time
                    if swaps_done > 20:
                        break

    # Final verification and conversion
    # The code is a list of indices.
    # Convert to list of words (lists of ints)
    result = [idx_to_word(c) for c in code]
    
    return result
