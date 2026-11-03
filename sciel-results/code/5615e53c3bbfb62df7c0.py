import time
import random
import itertools

def construct(q, n, radius):
    """
    Finds a covering code using a randomized greedy set cover heuristic.
    """
    # Deadline guard: 150s limit, keep 10s margin
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Only optimize for the specific case q=3, n=6, radius=1
    # The seed program handles general cases poorly for large n, but 
    # the task specifies this exact instance.
    # For robustness, fall back to the seed method if parameters differ.
    if not (q == 3 and n == 6 and radius == 1):
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n  # 729
    
    # Map word tuples to indices 0..728
    # Index = w0 * 3^5 + w1 * 3^4 + ... + w5
    # Precompute the "balls" (sets of indices covered by each center)
    # Each center covers 1 + n*(q-1) = 13 words.
    
    # To speed up the greedy step, we represent the set of covered words by a center
    # as a tuple of indices.
    balls = []
    for idx in range(total_words):
        # Convert index to word tuple
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % 3)
            temp //= 3
        w = tuple(reversed(w))
        
        # Generate all words within Hamming distance 1
        covered_indices = set()
        covered_indices.add(idx)
        
        for pos in range(n):
            for new_val in range(q):
                if new_val == w[pos]:
                    continue
                # Create new word
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_w = tuple(new_w_list)
                
                # Convert back to index
                new_idx = 0
                multiplier = 1
                for j in range(n-1, -1, -1):
                    new_idx += new_w[j] * multiplier
                    multiplier *= 3
                covered_indices.add(new_idx)
                
        balls.append(frozenset(covered_indices))
        
    # Greedy Set Cover with randomization
    best_solution = None
    best_size = total_words # Upper bound
    
    # Run multiple iterations
    while time.time() < deadline:
        uncovered = set(range(total_words))
        solution = []
        
        while uncovered:
            best_cover_count = -1
            best_candidate = -1
            candidates_with_max = []
            
            # To avoid O(N^2) scanning of all candidates every step if possible,
            # we could use a priority queue, but N=729 is small enough that 
            # linear scan is fast in Python if optimized.
            # However, iterating 729 candidates * 729 steps is ~500k operations.
            # The set intersection is the bottleneck.
            
            # Optimization: Instead of set intersection every time, 
            # we can just check membership.
            
            # Randomize candidate order to break ties and escape local minima
            candidates = list(range(total_words))
            random.shuffle(candidates)
            
            for c in candidates:
                if not uncovered:
                    break
                # Count how many words in ball[c] are in uncovered
                # Using set intersection is efficient in C
                count = len(balls[c] & uncovered)
                
                if count > best_cover_count:
                    best_cover_count = count
                    best_candidate = c
                    candidates_with_max = [c]
                elif count == best_cover_count and count > 0:
                    candidates_with_max.append(c)
                    
            # If no candidate covers anything, we have a bug (should not happen)
            if best_cover_count <= 0:
                break
                
            # Pick a random candidate from the ones with max coverage
            # This adds randomness to the greedy choice
            chosen = random.choice(candidates_with_max)
            
            solution.append(chosen)
            uncovered -= balls[chosen]
            
        if len(solution) < best_size:
            best_size = len(solution)
            best_solution = solution[:]
            
            # If we found 73, we are very likely at the optimum. 
            # We can stop early to save time, or continue to try for 72.
            # Given 73 is the known best, 72 is highly unlikely. 
            # Let's run a few more times if we hit 73 just in case, but stop if time is tight.
            if best_size <= 73:
                # Check time
                if time.time() > deadline - 10.0:
                    break
                # If we have plenty of time, keep trying a bit more to see if 72 exists
                # But 73 is the target.
                pass

    # If no solution was found (should not happen), return empty or fallback
    if best_solution is None:
         # Fallback to seed logic
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Convert indices back to list of lists of integers
    final_codewords = []
    for idx in best_solution:
        w = []
        temp = idx
        for _ in range(n):
            w.append(temp % 3)
            temp //= 3
        w.reverse()
        final_codewords.append(w)
        
    return final_codewords
