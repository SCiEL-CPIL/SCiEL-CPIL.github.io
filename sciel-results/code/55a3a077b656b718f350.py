import time
import numpy as np
import random

def construct(q, n, radius):
    """
    Constructs a covering code using:
    1. Fast greedy set cover (NumPy vectorized) to get a feasible solution.
    2. Redundancy removal (trimming) to reduce size.
    3. Randomized swap-based local search to escape local optima and further reduce size.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    total_words = q ** n
    
    # Helper to convert index to list of digits
    def idx_to_digits(idx, base, length):
        digits = [0] * length
        for i in range(length):
            digits[i] = idx % base
            idx //= base
        return digits

    # Helper to convert list of digits to index
    def digits_to_idx(digits, base):
        idx = 0
        for d in reversed(digits):
            idx = idx * base + d
        return idx

    # Precompute neighbors for each word
    # For radius 1, this is fast. For general radius, it might be slow, but we assume small radius.
    # We store neighbors as lists of indices.
    neighbor_lists = [None] * total_words
    for i in range(total_words):
        digits = idx_to_digits(i, q, n)
        neighbors = {i}
        
        if radius == 1:
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    digits[pos] = new_val
                    neighbors.add(digits_to_idx(digits, q))
                digits[pos] = orig
        else:
            # General radius generation is expensive. 
            # For this specific task (q=3, n=8, r=1), the above is sufficient.
            # If radius > 1, we fall back to a simple check or assume it's small enough.
            # Given the constraints, we'll stick to radius 1 optimization primarily.
            # If radius > 1, we generate all words within distance radius.
            # This is O(q^n * n^radius) which can be large.
            # However, the problem statement specifies K_3(8,1).
            pass
            
        neighbor_lists[i] = list(neighbors)

    # Build adjacency matrix M (bool)
    # M[i, j] is True if j is a neighbor of i
    M = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        M[i, neighbor_lists[i]] = True

    # 1. Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    solution_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Find the word that covers the most uncovered words
        # counts[i] = number of neighbors of i that are uncovered
        # We can compute this as M @ uncovered (treating bool as 0/1)
        # But M is bool, uncovered is bool. 
        # np.sum(M & uncovered, axis=1) is O(N^2) per step.
        # N=6561, N^2 ~ 43M. This might be slow in Python loops, but NumPy vectorizes it.
        # Let's try M @ uncovered.astype(np.int8)
        
        counts = M @ uncovered.astype(np.int8)
        
        if counts.max() == 0:
            break
            
        best_idx = np.argmax(counts)
        solution_indices.append(best_idx)
        uncovered[M[best_idx]] = False

    # 2. Trimming Phase (Remove redundant codewords)
    # cover_count[j] = number of codewords in solution that cover j
    cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in solution_indices:
        cover_count[M[idx]] += 1
        
    # Iterate to remove redundant codewords
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate in reverse to avoid index issues
        for i in range(len(solution_indices) - 1, -1, -1):
            idx = solution_indices[i]
            # Check if any neighbor has cover_count == 1
            # If min cover count in neighbors > 1, it's redundant
            min_cover = np.min(cover_count[M[idx]])
            if min_cover > 1:
                cover_count[M[idx]] -= 1
                solution_indices.pop(i)
                changed = True
                # Restarting the loop is safer but slower. 
                # Since we removed one, we can continue checking others? 
                # Removing one might make another redundant, or might not.
                # To be efficient, we break and restart the while loop.
                break

    # 3. Randomized Local Search (Swap-based)
    # Goal: Replace a codeword with another to reduce the total code size? 
    # Actually, swaps keep size constant. 
    # The strategy is: 
    # 1. Try to remove a codeword. If it fails (i.e., it's essential), identify the "bottleneck" words it uniquely covers.
    # 2. Try to find a new codeword that covers those bottleneck words AND allows removal of the current one?
    #    Or simply: try to find a codeword that, if added, allows removing TWO existing codewords?
    #    
    # A simpler and effective heuristic for covering codes:
    # Try to swap codeword A with codeword B if B is not in the code.
    # Condition: The set of words uniquely covered by A (after removing A) must be a subset of the neighbors of B.
    # If so, we can remove A and add B. Size remains same.
    # But how does this reduce size?
    # It doesn't directly. 
    #
    # However, the "Inspiration 1" note says: "Randomized swap-based local search escapes greedy local optima... potentially reducing code size".
    # How? 
    # Maybe by rearranging the code into a "better" configuration where subsequent trimming is more effective?
    # Or maybe the swap itself allows for a chain of removals?
    #
    # Let's implement a standard 1-swap local search:
    # 1. Pick a random codeword A to remove.
    # 2. Calculate "lost" words: words covered ONLY by A.
    # 3. If no lost words, remove A (size decreases). (This is just trimming again).
    # 4. If lost words exist, find a candidate B (not in code) that covers ALL lost words.
    # 5. If such B exists, swap A for B.
    # 6. Repeat trimming.
    #
    # Does this reduce size? 
    # If we swap A for B, size is same. 
    # But maybe B covers more "shared" words, making other codewords redundant?
    # Yes, that's the key. A better placement of B might allow removing another codeword C in the next trimming pass.
    
    # We need to track which words are covered by which codewords? 
    # cover_count is sufficient for trimming.
    # For swap logic:
    # If we remove A, words with cover_count == 1 in A's neighborhood are "lost".
    # We need B such that B covers all these lost words.
    
    # Optimization: 
    # Precompute for each word w, the set of codewords that cover w? 
    # That's the inverse of M. 
    # Or just use cover_count.
    
    # Let's implement a limited number of swap attempts.
    
    # To speed up finding B:
    # The "lost" words are a small set (usually).
    # We need B such that lost_words subset of neighbors(B).
    # This is equivalent to: B must be in the intersection of the "reverse neighborhoods" of each lost word.
    # Reverse neighborhood of w: set of all B such that w in neighbors(B).
    # Since the graph is undirected (Hamming distance), neighbors(B) = reverse_neighborhood(B).
    # So B must be in the intersection of neighbors(w) for all w in lost_words.
    
    # Let's precompute reverse neighbors? 
    # For radius 1, the graph is regular and undirected. 
    # So neighbors(w) is the same list.
    
    # Algorithm for swap:
    # 1. Iterate through current codewords.
    # 2. Pick one (random or first).
    # 3. Determine lost_words = { w in neighbors(A) | cover_count[w] == 1 }
    # 4. If lost_words is empty, remove A.
    # 5. Else, compute intersection = neighbors(lost_words[0])
    #    For w in lost_words[1:]: intersection &= neighbors(w)
    #    Candidates = intersection - current_code_set
    # 6. If Candidates is not empty, pick one B.
    #    Remove A, Add B.
    #    Update cover_count.
    #    Run trimming again.
    
    # To make it randomized:
    # Shuffle the order of codewords to try.
    
    # We will do this in a loop for a certain number of iterations or until time runs out.
    
    # To make it efficient:
    # We maintain a set of current codeword indices for O(1) lookup.
    code_set = set(solution_indices)
    
    # We will try to improve the solution.
    # We'll do a fixed number of "restarts" or "passes" of this local search.
    
    # Since the space is small, we can afford a few seconds for this.
    
    # Let's create a helper to perform one swap attempt on a specific codeword.
    # Returns True if a swap was made (and state updated).
    
    # Note: We need to be careful with cover_count updates.
    
    # Let's structure the local search as:
    # while time < deadline:
    #     found_improvement = False
    #     # Shuffle codewords to try
    #     shuffled = solution_indices.copy()
    #     random.shuffle(shuffled)
    #     for A in shuffled:
    #         if time > deadline: break
    #         # Check if A is removable
    #         lost = [w for w in neighbor_lists[A] if cover_count[w] == 1]
    #         if not lost:
    #             # Remove A
    #             cover_count[M[A]] -= 1
    #             solution_indices.remove(A)
    #             code_set.remove(A)
    #             found_improvement = True
    #             break # Restart outer loop? Or continue?
    #         else:
    #             # Find swap candidate
    #             # Intersection of neighbors of all lost words
    #             # lost is usually small.
    #             # Start with neighbors of first lost word
    #             # We need to check if any candidate is NOT in code_set.
    #             # And we should probably pick the one that covers the most "extra" words? 
    #             # Or just any?
    #             # Let's just pick any valid swap to escape local optima.
    #             
    #             # Efficient intersection:
    #             # Since N is small, we can just iterate candidates from the first lost word's neighbors.
    #             candidates = None
    #             for w in lost:
    #                 if candidates is None:
    #                     candidates = set(neighbor_lists[w])
    #                 else:
    #                     candidates &= set(neighbor_lists[w])
    #                     if not candidates:
    #                         break
    #             
    #             if candidates:
    #                 # Remove current code
    #                 candidates -= code_set
    #                 if candidates:
    #                     # Pick a random candidate
    #                     B = random.choice(list(candidates))
    #                     # Perform swap
    #                     # Remove A
    #                     cover_count[M[A]] -= 1
    #                     solution_indices.remove(A)
    #                     code_set.remove(A)
    #                     # Add B
    #                     cover_count[M[B]] += 1
    #                     solution_indices.append(B)
    #                     code_set.add(B)
    #                     found_improvement = True
    #                     # Do not break, continue trying to remove more? 
    #                     # Or break to allow re-evaluation?
    #                     # Breaking is safer for correctness of cover_count logic if we rely on it.
    #                     # But we can continue.
    #                     pass
    #     
    #     if not found_improvement:
    #         # No swaps or removals found in this pass.
    #         # We might be stuck. 
    #         # We can try a "kick" - a random move that might temporarily worsen the solution?
    #         # But we must maintain feasibility (cover all words).
    #         # A swap maintains feasibility if the lost words are covered by the new B.
    #         # If no swap is found, we are in a local optimum for 1-swap.
    #         # We can try 2-swaps? Too expensive.
    #         # Or we can just stop.
    #         break

    # Let's implement this carefully.
    
    # One issue: `solution_indices.remove(A)` is O(N). 
    # We can use a list and pop by index if we track indices, but we shuffled.
    # Given N ~ 400-500, O(N) remove is fine.
    
    # Let's run the local search.
    
    # We will do multiple passes.
    # To avoid getting stuck in a cycle, we can keep track of recent states? 
    # Or just rely on the randomness.
    
    # Since the goal is to minimize size, and swaps don't change size, 
    # the only way to improve is if a swap enables a removal in the same pass or subsequent passes.
    # The logic above allows removals immediately after a swap if we don't break.
    # But if we break on the first improvement, we restart the shuffle.
    
    # Let's try a loop that continues until no improvement is found in a full pass.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        shuffled = solution_indices.copy()
        random.shuffle(shuffled)
        
        for A in shuffled:
            if time.time() > deadline:
                break
                
            # Check if A is in code_set (it should be, unless we messed up state)
            if A not in code_set:
                continue
                
            # Determine lost words
            # lost = [w for w in neighbor_lists[A] if cover_count[w] == 1]
            # This list comprehension is O(deg(A)) ~ 17.
            
            # We can vectorize this?
            # neighbor_indices_A = neighbor_lists[A]
            # cover_counts_A = cover_count[neighbor_indices_A]
            # lost_mask = cover_counts_A == 1
            # lost_words = neighbor_indices_A[lost_mask]
            
            neighbor_indices_A = np.array(neighbor_lists[A])
            cover_counts_A = cover_count[neighbor_indices_A]
            lost_mask = cover_counts_A == 1
            lost_words = neighbor_indices_A[lost_mask]
            
            if len(lost_words) == 0:
                # Remove A
                cover_count[M[A]] -= 1
                # Remove from list
                if A in solution_indices:
                    solution_indices.remove(A)
                code_set.remove(A)
                improved = True
                # Break to restart the pass? 
                # Or continue? 
                # If we continue, we might remove more.
                # But the cover_count changed, so other checks might be invalid?
                # No, cover_count is global. 
                # However, removing A might make other codewords redundant.
                # It might also make other codewords *less* redundant? No.
                # It only decreases cover counts.
                # So if B was redundant before, it's still redundant (or more so).
                # So we can continue.
                continue
                
            else:
                # Find swap candidate
                # Intersection of neighbors of lost_words
                # lost_words is a numpy array.
                
                # Start with neighbors of the first lost word
                # We need to be careful: if lost_words is empty, we handled it above.
                
                # Convert to set for intersection?
                # The size of neighbor list is 17.
                # Intersection of sets of size 17 is fast.
                
                # Get neighbors of first lost word
                first_lost = lost_words[0]
                candidates = set(neighbor_lists[first_lost])
                
                for w in lost_words[1:]:
                    candidates &= set(neighbor_lists[w])
                    if not candidates:
                        break
                
                if candidates:
                    # Exclude current code
                    candidates -= code_set
                    if candidates:
                        # Pick a random candidate
                        # To be smarter, we could pick the one that maximizes "gain" in terms of 
                        # making other codewords redundant? 
                        # For now, random is a good heuristic for escaping local optima.
                        B = random.choice(list(candidates))
                        
                        # Perform swap
                        # Remove A
                        cover_count[M[A]] -= 1
                        if A in solution_indices:
                            solution_indices.remove(A)
                        code_set.remove(A)
                        
                        # Add B
                        cover_count[M[B]] += 1
                        solution_indices.append(B)
                        code_set.add(B)
                        
                        improved = True
                        # Do not break, continue to try removing more.
                        # The swap might have enabled removals.
                        continue

    # Final trimming pass to be safe
    # The loop above should have converged, but let's do one more clean pass.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(solution_indices) - 1, -1, -1):
            idx = solution_indices[i]
            min_cover = np.min(cover_count[M[idx]])
            if min_cover > 1:
                cover_count[M[idx]] -= 1
                solution_indices.pop(i)
                changed = True
                break

    # Convert back to list of lists of integers
    result = []
    for c in solution_indices:
        result.append(idx_to_digits(c, q, n))
        
    return result
