import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code for the given parameters using a greedy set cover
    approach followed by local optimization (redundancy removal).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    total_space = q ** n
    ball_size = 1 + n * (q - 1) * radius  # For radius 1, this is 1 + n(q-1)
    
    # For the specific task K_3(8,1), total_space=6561, ball_size=17.
    # We will use a greedy algorithm.
    
    # Precompute all possible codewords? 
    # 3^8 = 6561. This is small enough to keep in memory.
    # However, iterating over all 6561 candidates at each step of greedy might be slow 
    # if we do it naively. 
    # Number of steps in greedy is approx total_space / ball_size ~ 386.
    # 386 * 6561 * (cost to compute overlap) 
    # We can optimize by maintaining an array of 'uncovered' points.
    
    # Represent the entire space as a 1D array of indices 0 to total_space-1.
    # covered[i] = True if point i is covered.
    covered = np.zeros(total_space, dtype=bool)
    
    # We need to map integer indices to base-q vectors to compute Hamming distance/balls.
    # Or, more efficiently, precompute the list of points in the ball for each candidate center?
    # 6561 centers * 17 points per ball = ~111k entries. This is very small.
    
    # Precompute balls for all possible centers
    # ball_centers[c] is a list of indices of points covered by center c
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    # To compute the ball of a center, we need to find all words within Hamming distance 'radius'.
    # For radius 1, this is the center itself plus all words with exactly one position changed.
    
    def get_ball_indices(center_idx):
        center = all_centers[center_idx]
        indices = [center_idx]
        if radius >= 1:
            for pos in range(n):
                original_val = center[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    # Create new word
                    new_word = center.copy()
                    new_word[pos] = new_val
                    # Find its index
                    # Using tuple lookup is O(1) on average
                    indices.append(center_to_idx[tuple(new_word)])
        return np.array(indices, dtype=np.int32)
    
    # Precompute all balls
    # This takes a bit of time upfront but makes the greedy loop fast.
    balls = []
    for i in range(total_space):
        balls.append(get_ball_indices(i))
        
    # Greedy Set Cover
    chosen_centers = []
    # We want to pick the center that covers the most 'new' points (uncovered points in its ball).
    
    # To speed up, we can maintain a count of uncovered points in each ball.
    # But updating all balls when we pick one is O(total_space).
    # Alternative: At each step, iterate through all remaining candidate centers and compute
    # the number of uncovered points in their ball.
    # Candidates: Initially all 6561.
    # Step 1: Pick best. Update covered.
    # Step 2: Pick best from remaining.
    # ...
    # Number of steps ~ 400.
    # Cost per step: 6561 * (check 17 points) ~ 111k operations.
    # Total cost: 400 * 111k = 44 million operations. In Python, this might be too slow (seconds).
    # In NumPy, we can vectorize the "score" calculation.
    
    # Let's use a vectorized approach for the greedy step.
    # We have a boolean array `covered`.
    # For each candidate center i, score[i] = sum(~covered[balls[i]])
    # We can't easily vectorize the "sum over variable length lists" without a matrix.
    # Let's create a sparse matrix or just use a loop with NumPy slicing if possible.
    # Actually, since ball size is fixed (17), we can stack them into a 2D array?
    # 6561 x 17 array of indices.
    
    ball_indices_matrix = np.array(balls, dtype=np.int32) # Shape (total_space, ball_size)
    
    while True:
        if time.time() > deadline:
            break
            
        # Calculate score for each candidate: number of uncovered points in its ball
        # uncovered_in_ball[i, j] = ~covered[ball_indices_matrix[i, j]]
        # score[i] = sum(uncovered_in_ball[i, :])
        
        # This creates a large temporary array (6561 * 17 bools), which is fine.
        # ~covered is a view.
        # We need to index `covered` with `ball_indices_matrix`.
        # This is an advanced indexing operation.
        
        # To avoid recomputing for already chosen centers, we could mask them, 
        # but they will have low scores anyway if we remove them from consideration.
        # Let's just pick the max score.
        
        # Vectorized score calculation
        # Note: covered is 1D bool. ball_indices_matrix is 2D int.
        # covered[ball_indices_matrix] results in a 2D bool array of shape (total_space, ball_size)
        # We want the sum of True values (uncovered).
        # uncovered_mask = ~covered[ball_indices_matrix]
        # scores = uncovered_mask.sum(axis=1)
        
        # This is O(total_space * ball_size) per iteration.
        # 6561 * 17 = 111k operations.
        # 400 iterations * 111k = 44M operations.
        # NumPy handles this very fast (milliseconds per iteration).
        
        scores = (~covered[ball_indices_matrix]).sum(axis=1)
        
        # Find the candidate with the highest score
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            # All points are covered
            break
            
        # Add this center to our code
        chosen_centers.append(best_idx)
        
        # Update covered array
        covered[balls[best_idx]] = True
        
        # Optional: Remove chosen center from future consideration?
        # Not strictly necessary as its score will drop to 0 or near 0 relative to others,
        # but to be safe and efficient, we can set its score to -1 in future iterations?
        # No, the score calculation is dynamic. Once covered, its contribution to new coverage is 0.
        # However, if we pick it again, it's redundant. We should avoid picking the same center twice.
        # We can keep a list of chosen indices and ignore them.
        # Or simpler: since the score will be 0 for fully covered balls, and we break if max score is 0,
        # we might pick a redundant center if multiple centers have score 0? 
        # No, if max score is 0, we break.
        # But what if max score > 0, but the best center is already chosen?
        # If a center is already chosen, its ball is fully covered, so its score is 0.
        # So it won't be picked again unless all other scores are 0, in which case we break.
        # So we don't need to explicitly exclude them.
        
        # Check for time
        if time.time() > deadline:
            break
        
    # Convert indices back to codewords
    code = [all_centers[i] for i in chosen_centers]
    
    # Post-processing: Local Search / Redundancy Removal
    # Try to remove any codeword that is redundant.
    # A codeword is redundant if removing it does not uncover any points.
    # i.e., the union of the balls of the other codewords covers the ball of this codeword.
    
    if time.time() < deadline - 5.0: # Leave some time
        # Rebuild covered array for the current code
        covered_final = np.zeros(total_space, dtype=bool)
        for i in chosen_centers:
            covered_final[balls[i]] = True
        
        redundant_removed = True
        while redundant_removed and time.time() < deadline - 5.0:
            redundant_removed = False
            # Check each codeword
            for idx_in_code, c_idx in enumerate(list(chosen_centers)):
                if time.time() > deadline - 5.0:
                    break
                    
                # Temporarily remove this codeword
                # We need to know if any point in its ball becomes uncovered.
                # This is equivalent to checking if any point in its ball is covered ONLY by this codeword.
                # We can compute the coverage count for each point.
                
                # Efficient check:
                # Let B be the set of points in the ball of c_idx.
                # For each point p in B, if covered_final[p] is True, is it covered by any other codeword?
                # We can sum the coverage of all other codewords on B.
                
                # Alternative: 
                # Calculate how many times each point in B is covered by the current set.
                # If min count > 1 for all p in B, it's redundant.
                
                # This is expensive to do from scratch for each candidate.
                # Let's try a simpler heuristic: 
                # Just remove the first redundant one found and restart the check?
                
                # Let's compute coverage counts for the whole space once? No, too big.
                # Let's just check the specific ball.
                
                ball_pts = balls[c_idx]
                
                # Count coverage from other centers
                # We can iterate through other centers and mark their coverage on these specific points?
                # Or, simpler: 
                # Let's assume we can just try to remove it and check if `covered_final` still covers `ball_pts`.
                # We need to know if `covered_final` (without this center) covers `ball_pts`.
                
                # Let's maintain a count array? 
                # Given the small size (6561), we can afford to recompute coverage for the ball of the candidate
                # by checking all other centers.
                
                # Actually, a faster way:
                # We know `covered_final` is True for all points.
                # If we remove center `c_idx`, a point `p` in `ball_pts` becomes uncovered 
                # if and only if no other center covers `p`.
                
                # Let's check if there is any point in `ball_pts` that is covered ONLY by `c_idx`.
                # We can do this by:
                # 1. Create a temp array `covered_others` for points in `ball_pts`.
                # 2. Initialize to False.
                # 3. For each other center `c2`, mark `covered_others` where `balls[c2]` intersects `ball_pts`.
                
                # This is still O(N_centers * ball_size).
                
                # Let's try a randomized local search instead, which is often faster to implement and run.
                # But redundancy removal is a clean structural improvement.
                
                # Let's stick to a simple removal check:
                # For each point p in ball_pts, check if any other center covers p.
                # This is O(len(ball_pts) * len(chosen_centers)).
                # 17 * 400 = 6800 operations per candidate.
                # 400 candidates * 6800 = 2.7M operations.
                # This is fast enough in Python if optimized, but let's use NumPy.
                
                # Vectorized check for redundancy of c_idx:
                # We want to know if for all p in ball_pts, exists c2 != c_idx such that p in ball(c2).
                
                # Let's just iterate through other centers and see if they cover any part of ball_pts that c_idx uniquely covers?
                # No, we need to know if c_idx is the *only* one covering some p.
                
                # Let's compute the "unique coverage" of c_idx.
                # unique_covered = ball_pts
                # for c2 in chosen_centers:
                #    if c2 == c_idx: continue
                #    unique_covered = unique_covered[~np.isin(unique_covered, balls[c2])]
                # if len(unique_covered) == 0: redundant
                
                # np.isin is slow.
                
                # Let's use the coverage count idea but only for the points in ball_pts.
                # counts = np.zeros(len(ball_pts), dtype=int)
                # for c2 in chosen_centers:
                #    if c2 == c_idx: continue
                #    # Find intersection of balls[c2] and ball_pts
                #    # This is still tricky.
                
                # Given the time constraints and the fact that greedy is already good, 
                # let's just do a simple pass of redundancy removal using a coverage count array for the WHOLE space.
                # It's 6561 integers. Very small.
                
                # Rebuild coverage counts
                counts = np.zeros(total_space, dtype=np.int32)
                for c in chosen_centers:
                    counts[balls[c]] += 1
                
                # Now check each center
                removed_any = False
                new_chosen = []
                for c in chosen_centers:
                    # If min(counts[balls[c]]) > 1, then every point in its ball is covered by at least 2 centers.
                    # So this center is redundant.
                    if np.min(counts[balls[c]]) > 1:
                        # It is redundant, do not add to new_chosen
                        # But we must update counts? 
                        # If we remove it, the counts for its ball decrease by 1.
                        # If we remove multiple, we need to be careful.
                        # Let's just remove one at a time to be safe, or mark it and rebuild counts.
                        # Marking and rebuilding is safer.
                        pass 
                    else:
                        new_chosen.append(c)
                
                # If we found redundant ones, we need to actually remove them and update counts.
                # The above logic just identifies them. 
                # Let's refine:
                redundant_indices = [i for i, c in enumerate(chosen_centers) if np.min(counts[balls[c]]) > 1]
                
                if redundant_indices:
                    # Remove the first one found (or all? Removing all at once might be unsafe if they cover each other's unique spots? 
                    # No, if min count > 1, it means even if I remove THIS one, the others still cover it.
                    # But if I remove TWO, do they still cover each other?
                    # If A is redundant w.r.t B, and B is redundant w.r.t A, they might form a cycle.
                    # However, if min count > 1 for A, it means B (and others) cover A's ball.
                    # If we remove A, B is still there.
                    # If we remove B, A is still there.
                    # If we remove both, we might uncover points.
                    # So we should remove one at a time.
                    
                    # Pick the first redundant one
                    c_to_remove = chosen_centers[redundant_indices[0]]
                    chosen_centers.remove(c_to_remove)
                    redundant_removed = True
                    # Loop will restart and recompute counts
                    
    return [all_centers[i] for i in chosen_centers]

# Ensure the function signature matches
def construct(q, n, radius):
    return _construct_impl(q, n, radius)

def _construct_impl(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_space = q ** n
    ball_size = 1 + n * (q - 1) * radius
    
    all_centers = [list(c) for c in itertools.product(range(q), repeat=n)]
    center_to_idx = {tuple(c): i for i, c in enumerate(all_centers)}
    
    def get_ball_indices(center_idx):
        center = all_centers[center_idx]
        indices = [center_idx]
        if radius >= 1:
            for pos in range(n):
                original_val = center[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_word = center.copy()
                    new_word[pos] = new_val
                    indices.append(center_to_idx[tuple(new_word)])
        return np.array(indices, dtype=np.int32)
    
    balls = []
    for i in range(total_space):
        balls.append(get_ball_indices(i))
        
    covered = np.zeros(total_space, dtype=bool)
    chosen_centers = []
    ball_indices_matrix = np.array(balls, dtype=np.int32)
    
    while True:
        if time.time() > deadline:
            break
            
        scores = (~covered[ball_indices_matrix]).sum(axis=1)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
            break
            
        chosen_centers.append(best_idx)
        covered[balls[best_idx]] = True
        
        if time.time() > deadline:
            break
        
    # Redundancy Removal
    if time.time() < deadline - 5.0:
        redundant_removed = True
        while redundant_removed and time.time() < deadline - 5.0:
            redundant_removed = False
            counts = np.zeros(total_space, dtype=np.int32)
            for c in chosen_centers:
                counts[balls[c]] += 1
            
            for i, c in enumerate(chosen_centers):
                if time.time() > deadline - 5.0:
                    break
                if np.min(counts[balls[c]]) > 1:
                    chosen_centers.pop(i)
                    redundant_removed = True
                    break # Restart loop with new set
                    
    return [all_centers[i] for i in chosen_centers]
