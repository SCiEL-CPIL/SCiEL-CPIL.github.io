import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    """
    Constructs a covering code using Simulated Annealing on the set of codewords.
    
    Strategy:
    1. Initialize with a greedy maximal covering set.
    2. Perform local search (Hill Climbing) to remove redundant codewords.
    3. Perform Simulated Annealing to explore swaps and moves that might
       reduce the total count by escaping local minima of the greedy solution.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger instances if this ever gets called with different params, 
        # though task specifies 7 matches, q=3.
        pass

    # Precompute all words
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)
    
    # Helper to compute Hamming distances between a candidate set and all words
    # This is O(|S| * N * n). For |S|~200, N=2187, n=7, this is ~3M ops, very fast in numpy.
    
    def get_coverage_status(codewords_list):
        """
        Returns an array of size total_words where entry i is the number of 
        codewords within radius of word i.
        """
        if not codewords_list:
            return np.zeros(total_words, dtype=np.int32)
        
        cw = np.array(codewords_list, dtype=np.uint8)
        # Compute distances: (len(cw), 1, n) vs (1, total_words, n)
        # dist[i, j] = sum(cw[i] != all_words[j])
        # We need to count how many i have dist[i, j] <= radius
        
        # To save memory and time, we can do this in chunks if |cw| is large, 
        # but |cw| is small (~200), so direct broadcast is fine.
        diffs = (cw[:, np.newaxis, :] != all_words[np.newaxis, :, :])
        dists = diffs.sum(axis=2)
        
        # covered[j] = sum over i of (dists[i, j] <= radius)
        covered_counts = (dists <= radius).sum(axis=0)
        return covered_counts

    def is_valid(codewords_list):
        """Check if every word is covered."""
        if not codewords_list:
            return total_words == 0
        counts = get_coverage_status(codewords_list)
        return np.all(counts > 0)

    # Phase 1: Greedy Initialization
    # Standard greedy set cover
    covered = np.zeros(total_words, dtype=bool)
    selected_indices = []
    
    # Precompute balls for greedy phase efficiency? 
    # Actually, for initialization, we can just pick random words and check coverage,
    # or do a quick greedy. Let's do a fast greedy.
    
    # To speed up greedy, we can use the same distance logic but only for candidates.
    # However, a simpler initialization: Pick random words until covered? No, greedy is better.
    
    # Let's implement a simple greedy that picks the word covering most new points.
    # To avoid O(N^2) overhead in Python loops, we'll do a few passes.
    
    # Actually, for q=3, n=7, N=2187. 
    # Greedy:
    # 1. Calculate initial coverage (0)
    # 2. Loop:
    #    a. Find best candidate. 
    #    b. Add to selected.
    #    c. Update covered.
    
    # We can speed this up by precomputing the "ball" of each word as a boolean mask or index list.
    # Since N is small, let's precompute the index lists for radius=1 balls.
    
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    balls = []
    for i in range(total_words):
        w = all_words[i]
        inds = [i]
        for j in range(n):
            for v in range(q):
                if v == w[j]:
                    continue
                new_w = w.copy()
                new_w[j] = v
                inds.append(word_to_idx[tuple(new_w)])
        balls.append(np.array(inds, dtype=np.int32))
        
    balls_arr = np.array(balls, dtype=np.int32)
    
    # Greedy Loop
    while True:
        if time.time() > deadline - 60.0: # Leave 60s for SA
            break
        if covered.all():
            break
            
        # Compute new coverage for each candidate
        # new_covered[i] = number of words in ball[i] not currently covered
        # We can use matrix multiplication or broadcasting.
        # covered is bool vector. 
        # For each i, sum(~covered[balls[i]])
        
        # Efficient way:
        # ~covered is bool. 
        # We want sum over rows of balls_arr of ~covered[balls_arr]
        # This is effectively a sparse matrix-vector product if we had the matrix.
        # But balls_arr is dense enough (2187 x 19).
        
        # Let's just do it in a vectorized way if possible, or simple loop if N is small.
        # N=2187, loop is 2187 iterations. Each iteration is sum of 19 bools. Very fast.
        
        # However, doing this in Python loop for 2187 items might be slow if repeated many times.
        # Let's try to vectorize:
        # uncovered_mask = ~covered
        # We need to count how many 1s in uncovered_mask are in each ball.
        # This is equivalent to: for each word j, it contributes 1 to ball[i] if j in ball[i] and not covered[j].
        # Let M be the incidence matrix where M[i, j] = 1 if j in ball[i].
        # Then new_counts = M @ (uncovered_mask.astype(int))
        
        # We can build M once? M is 2187x2187 binary. 4.7M entries. 
        # Memory: 4.7M bytes if uint8. Fine.
        
        # Let's build M once outside the loop.
        # M[i, j] = 1 if dist(i, j) <= radius
        
        # Actually, we already have balls_arr. 
        # Let's stick to the loop for simplicity in greedy, it's fast enough for 200-300 iterations.
        
        max_new = 0
        best_idx = -1
        # Optimization: Only check candidates that aren't already selected? 
        # Actually, we can select any word. 
        # To speed up, we can just iterate.
        
        # Vectorized approach for greedy step:
        # Create a matrix where row i is the ball of i.
        # We want to sum ~covered over each row.
        # ~covered is a vector v.
        # We want v[balls[i]] sum.
        
        # Let's use the precomputed balls_arr.
        # new_counts = np.array([~covered[b].sum() for b in balls_arr])
        # This list comprehension might be slow.
        
        # Alternative: 
        # uncovered_indices = np.where(~covered)[0]
        # For each uncovered index j, find which balls contain j.
        # This is the transpose of the incidence matrix.
        
        # Let's just use the loop. 2187 iterations is ~1ms.
        # If we do this 200 times, it's 200ms. Acceptable.
        
        new_counts = np.zeros(total_words, dtype=np.int32)
        # We can update incrementally? No, just recompute.
        
        # Actually, a faster way:
        # Let U be the set of uncovered words.
        # For each u in U, it is covered by all c such that dist(c, u) <= radius.
        # So we can increment new_counts[c] for all c in ball(u).
        # Since balls are symmetric, ball(u) is the same as the list of centers covering u.
        # We have balls_arr[center] = list of words covered.
        # We need the reverse mapping: for each word u, list of centers c covering it.
        # Let's precompute cover_centers_arr where cover_centers_arr[u] is list of c such that u in ball(c).
        
        # Symmetry of Hamming distance means cover_centers_arr[u] == balls_arr[u].
        # Wait, if u is in ball(c), then c is in ball(u). Yes.
        # So we can just iterate over uncovered words and increment the count for their "center balls".
        
        uncovered_indices = np.where(~covered)[0]
        # For each u in uncovered_indices, the candidates that cover u are in balls_arr[u].
        # So we can do:
        # for u in uncovered_indices:
        #     new_counts[balls_arr[u]] += 1
        
        # This is efficient if number of uncovered words is small.
        # Initially 2187. Then decreases.
        
        # Let's try this.
        if len(uncovered_indices) < total_words / 2:
             for u in uncovered_indices:
                 new_counts[balls_arr[u]] += 1
        else:
             # Fallback to direct sum if many uncovered
             for i in range(total_words):
                 new_counts[i] = (~covered[balls_arr[i]]).sum()
                 
        if new_counts.max() == 0:
            break
            
        # Pick one with max count. Random tie-breaking to diversify?
        # Let's pick one randomly from top 10 to avoid deterministic bias.
        max_val = new_counts.max()
        candidates = np.where(new_counts == max_val)[0]
        if len(candidates) > 0:
             # Pick one
             idx = random.choice(candidates.tolist())
             selected_indices.append(idx)
             covered[balls_arr[idx]] = True

    # Convert to list of codewords
    current_code = [all_words[i].tolist() for i in selected_indices]
    
    # Phase 2: Local Search (Hill Climbing) - Remove redundant
    def try_remove_redundant(code_list):
        removed_any = True
        while removed_any:
            removed_any = False
            # Check each codeword
            for i in range(len(code_list) - 1, -1, -1):
                # Temporarily remove i
                temp_code = code_list[:i] + code_list[i+1:]
                if is_valid(temp_code):
                    code_list.pop(i)
                    removed_any = True
                    break # Restart loop to update indices
        return code_list

    current_code = try_remove_redundant(current_code)
    
    # Phase 3: Simulated Annealing
    # Objective: Minimize len(code). 
    # Moves:
    # 1. Remove a codeword (if valid? No, SA allows invalid states with penalty).
    #    Actually, standard SA for set cover often keeps validity by only accepting valid moves,
    #    or uses a penalty for uncovered words.
    #    Let's use a penalty approach: Cost = len(code) * 1000 + number_of_uncovered_words.
    #    This strongly favors validity.
    
    def get_cost(code_list):
        if not code_list:
            return 1000000
        counts = get_coverage_status(code_list)
        uncovered = (counts == 0).sum()
        return len(code_list) * 1000 + uncovered
        
    current_cost = get_cost(current_code)
    best_code = current_code.copy()
    best_cost = current_cost
    
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.99
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Generate a move
        move_type = random.random()
        
        new_code = current_code.copy()
        
        if move_type < 0.4:
            # Add a random codeword
            idx = random.randint(0, total_words - 1)
            # Avoid duplicates
            if all_words[idx].tolist() not in new_code:
                new_code.append(all_words[idx].tolist())
            else:
                continue
        elif move_type < 0.8:
            # Remove a random codeword
            if len(new_code) > 1:
                idx = random.randint(0, len(new_code) - 1)
                new_code.pop(idx)
            else:
                continue
        else:
            # Swap a random codeword with another random one
            if len(new_code) > 0:
                idx1 = random.randint(0, len(new_code) - 1)
                idx2 = random.randint(0, total_words - 1)
                if all_words[idx2].tolist() not in new_code:
                    new_code[idx1] = all_words[idx2].tolist()
                else:
                    continue
        
        new_cost = get_cost(new_code)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_code = new_code
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_code = current_code.copy()
                
        T *= cooling_rate
        
        # Occasionally try to remove redundant to keep code tight
        if random.random() < 0.1:
             current_code = try_remove_redundant(current_code)
             current_cost = get_cost(current_code)
             if current_cost < best_cost:
                 best_cost = current_cost
                 best_code = current_code.copy()

    # Final cleanup
    best_code = try_remove_redundant(best_code)
    
    # Ensure validity
    if not is_valid(best_code):
        # Fallback to greedy if SA broke it (shouldn't happen with penalty, but safe)
        best_code = [all_words[i].tolist() for i in selected_indices]
        best_code = try_remove_redundant(best_code)

    return best_code
