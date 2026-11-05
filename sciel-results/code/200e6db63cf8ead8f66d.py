"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _hamming_distance(a, b):
    """Compute Hamming distance between two tuples/lists of integers."""
    return sum(x != y for x, y in zip(a, b))


def _generate_ball_mask(center, q, n, radius, word_to_idx, total_words):
    """Generate a bitmask for all words within Hamming distance <= radius from center."""
    mask = 0
    # Iterate through all words in the space and check distance
    # This is O(q^n * n) per ball, which is 729 * 6 = 4374 ops per ball.
    # 729 balls * 4374 = ~3.2M ops total precomputation. This is fast.
    for i in range(total_words):
        # Reconstruct word from index? Or precompute list of words.
        # Let's precompute list of words to avoid repeated conversion.
        pass 
    return mask

def construct(q, n, radius):
    """
    Greedy set-covering with simulated annealing refinement for ternary covering code.
    Optimized using bitmask representations for fast set operations.
    """
    start_time = time.time()
    time_limit = 115  # leave some buffer before 120s timeout
    
    total_words = q ** n
    full_mask = (1 << total_words) - 1
    
    # Precompute all words as tuples and their indices
    all_words = []
    word_to_idx = {}
    idx_to_word = {}
    for i in range(total_words):
        # Convert integer index to base-q representation of length n
        w = []
        tmp = i
        for _ in range(n):
            w.append(tmp % q)
            tmp //= q
        w = tuple(w)
        all_words.append(w)
        word_to_idx[w] = i
        idx_to_word[i] = w

    # Precompute ball masks for all possible centers
    # ball_mask[i] is a bitmask where bit j is 1 if word j is within radius of word i
    ball_masks = [0] * total_words
    for i, center in enumerate(all_words):
        mask = 0
        for j, word in enumerate(all_words):
            if _hamming_distance(center, word) <= radius:
                mask |= (1 << j)
        ball_masks[i] = mask

    # Greedy Initialization
    # We want to find a small set of centers such that OR of their ball_masks == full_mask
    
    def greedy_init():
        uncovered = full_mask
        centers = []
        # To speed up greedy, we can iterate through all candidates each time
        # Since N=729, this is 729 * 729 * (bit ops) which is fast.
        while uncovered:
            best_center_idx = -1
            best_newly_covered = 0
            best_mask = 0
            
            for i in range(total_words):
                new_covered = ball_masks[i] & uncovered
                count = new_covered.bit_count()
                if count > best_newly_covered:
                    best_newly_covered = count
                    best_center_idx = i
                    best_mask = new_covered
            
            if best_center_idx == -1 or best_newly_covered == 0:
                break # Should not happen if radius is sufficient
            
            centers.append(best_center_idx)
            uncovered ^= best_mask # Remove covered bits
            
        return centers

    # Run greedy a few times with slight randomization to get diverse starts?
    # Standard greedy is deterministic. Let's add noise.
    best_initial_centers = None
    best_initial_cost = float('inf')
    
    # Objective function for SA
    M = total_words + 1 # Large penalty for uncovered words
    
    def get_cost(centers):
        if not centers:
            return M * total_words
        union_mask = 0
        for c in centers:
            union_mask |= ball_masks[c]
        uncovered = full_mask ^ union_mask
        num_uncovered = uncovered.bit_count()
        return M * num_uncovered + len(centers)

    # Generate initial solution
    centers = greedy_init()
    current_cost = get_cost(centers)
    best_centers = list(centers)
    best_cost = current_cost
    
    # Simulated Annealing
    T = 100.0
    T_min = 1.0
    cooling_rate = 0.9995
    random.seed(42)
    
    current_centers = list(centers)
    current_union_mask = 0
    for c in current_centers:
        current_union_mask |= ball_masks[c]
    current_uncovered = full_mask ^ current_union_mask
    current_num_uncovered = current_uncovered.bit_count()
    current_cost = M * current_num_uncovered + len(current_centers)

    while T > T_min and time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random center
            if len(current_centers) >= total_words: continue
            new_center = random.randint(0, total_words - 1)
            if new_center in current_centers:
                continue
            
            # Calculate new cost
            new_union_mask = current_union_mask | ball_masks[new_center]
            new_uncovered = full_mask ^ new_union_mask
            new_num_uncovered = new_uncovered.bit_count()
            new_cost = M * new_num_uncovered + (len(current_centers) + 1)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_centers.append(new_center)
                current_union_mask = new_union_mask
                current_uncovered = new_uncovered
                current_num_uncovered = new_num_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_centers = list(current_centers)
                    
        elif move_type < 0.7:
            # Remove a random center
            if len(current_centers) <= 1: continue
            idx_remove = random.randint(0, len(current_centers) - 1)
            center_to_remove = current_centers[idx_remove]
            
            # We can't just remove and check if valid easily with bitmask union
            # because we need to know if other centers cover the words that were ONLY covered by this one.
            # Recalculating union mask is O(N) bit ops.
            new_union_mask = 0
            for c in current_centers:
                if c == center_to_remove: continue
                new_union_mask |= ball_masks[c]
                
            new_uncovered = full_mask ^ new_union_mask
            new_num_uncovered = new_uncovered.bit_count()
            new_cost = M * new_num_uncovered + (len(current_centers) - 1)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_centers.pop(idx_remove)
                current_union_mask = new_union_mask
                current_uncovered = new_uncovered
                current_num_uncovered = new_num_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_centers = list(current_centers)

        else:
            # Replace a random center with another
            if len(current_centers) < 1: continue
            idx_replace = random.randint(0, len(current_centers) - 1)
            old_center = current_centers[idx_replace]
            new_center = random.randint(0, total_words - 1)
            if new_center == old_center or new_center in current_centers:
                continue
            
            # Recalculate union mask
            # Optimization: We can't easily subtract the old center's contribution from the union
            # without knowing if other centers cover those bits.
            # So we recalculate.
            new_union_mask = 0
            for i, c in enumerate(current_centers):
                if i == idx_replace:
                    new_union_mask |= ball_masks[new_center]
                else:
                    new_union_mask |= ball_masks[c]
            
            new_uncovered = full_mask ^ new_union_mask
            new_num_uncovered = new_uncovered.bit_count()
            new_cost = M * new_num_uncovered + len(current_centers)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                # Accept
                current_centers[idx_replace] = new_center
                current_union_mask = new_union_mask
                current_uncovered = new_uncovered
                current_num_uncovered = new_num_uncovered
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_centers = list(current_centers)

        T *= cooling_rate
        
        # Periodic restart if stuck?
        # If no improvement for a long time, maybe restart from greedy?
        # For now, just let it run.

    # Final verification and cleanup
    # Ensure best_centers is a valid cover
    final_union = 0
    for c in best_centers:
        final_union |= ball_masks[c]
    
    if final_union != full_mask:
        # Fallback to greedy if SA failed to find valid cover (shouldn't happen if M is large)
        best_centers = greedy_init()
        final_union = 0
        for c in best_centers:
            final_union |= ball_masks[c]
            
    # Convert indices back to lists
    return [list(idx_to_word[i]) for i in best_centers]
# EVOLVE-BLOCK-END
