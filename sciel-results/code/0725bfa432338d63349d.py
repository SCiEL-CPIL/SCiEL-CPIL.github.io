import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    points = list(range(16))
    
    # --- Helper: Fast Independence Number Calculation ---
    # We represent a set of points as a 16-bit integer.
    # A triple is valid if it is not a subset of the independent set.
    # We want to find the largest subset S such that no triple is contained in S.
    
    def compute_alpha(triples):
        """
        Computes the independence number of the hypergraph defined by 'triples'.
        triples: list of (a, b, c)
        """
        if not triples:
            return 16
            
        # Precompute masks for the triples
        masks = []
        for a, b, c in triples:
            masks.append((1 << a) | (1 << b) | (1 << c))
            
        # Check sizes from 16 down to 1
        # Optimization: If we find a triple contained in S, S is not independent.
        # We can iterate over subsets of size k.
        # For 16 points, C(16, 8) = 12870, C(16, 9) = 11440, etc.
        # This is fast enough if we do it a few thousand times.
        
        # To speed up, we can check if the complement of S contains a vertex cover?
        # No, simpler: just check subsets.
        
        # Heuristic: Start from large k.
        for k in range(16, 0, -1):
            for subset in itertools.combinations(points, k):
                # Create mask
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                # Check if any triple is fully contained in mask
                is_independent = True
                for m in masks:
                    if (mask & m) == m:
                        is_independent = False
                        break
                
                if is_independent:
                    return k
        return 1

    # --- Helper: Validity Check for Linear Triple System ---
    def is_valid_addition(triple, existing_triples, masks):
        a, b, c = triple
        new_mask = (1 << a) | (1 << b) | (1 << c)
        
        # Check intersection with all existing triples
        for m in masks:
            inter = new_mask & m
            # If intersection has > 1 bit, invalid
            if inter and (inter & (inter - 1)):
                return False
        return True

    # --- Phase 1: Structured Initialization ---
    # We use a mix of algebraic constructions to get a good starting point.
    # Construction 1: Lines in AG(2, 4) projected? No, AG(2,4) has 16 points.
    # Points are (x, y) in F_4^2.
    # Lines have 4 points. We need triples.
    # We can partition lines into triples? Or use a different geometry.
    
    # Let's try a random greedy approach but with a "smart" ordering first,
    # then refine. The previous program did random greedy.
    # Let's try a "greedy by degree" or similar.
    
    # Actually, let's stick to a robust random restart + local search.
    # The key improvement is the local search operator and the alpha calculation speed.
    
    best_alpha = 17
    best_triples = []
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(points, 3))
    # Shuffle for randomization
    random.shuffle(all_triples)
    
    # Multiple restarts
    num_restarts = 20
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1
        
        # --- Generate Initial Solution ---
        # Greedy: Pick random triples that don't conflict.
        # To make it denser, we can try to fill up points evenly.
        triples = []
        masks = []
        
        # Random order
        candidate_order = all_triples[:]
        random.shuffle(candidate_order)
        
        for t in candidate_order:
            if is_valid_addition(t, triples, masks):
                triples.append(t)
                masks.append((1 << t[0]) | (1 << t[1]) | (1 << t[2]))
                
        # --- Local Search: Simulated Annealing / Hill Climbing ---
        current_alpha = compute_alpha(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        # If alpha is already very low, we might stop early or just refine.
        # Let's run a local search for a fixed time slice per restart.
        local_deadline = time.time() + (110 / num_restarts)
        
        current_triples = triples
        current_masks = masks
        current_alpha = compute_alpha(current_triples)
        
        # Local search loop
        while time.time() < local_deadline:
            # Mutation: Remove one triple, try to add another
            # Or: Swap one triple for another
            
            if not current_triples:
                break
                
            # Randomly select a triple to remove
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_triple = current_triples.pop(idx_remove)
            removed_mask = current_masks.pop(idx_remove)
            
            # Try to add a new triple that fits
            # Sample a few candidates from all_triples
            # To be efficient, we can sample from triples that share points with removed_triple
            # or just random samples.
            
            added = False
            # Try a few random candidates
            for _ in range(50):
                new_t = random.choice(all_triples)
                if new_t == removed_triple:
                    continue
                if is_valid_addition(new_t, current_triples, current_masks):
                    current_triples.append(new_t)
                    current_masks.append((1 << new_t[0]) | (1 << new_t[1]) | (1 << new_t[2]))
                    added = True
                    break
            
            # Calculate new alpha
            new_alpha = compute_alpha(current_triples)
            
            # Accept if better, or if worse but within acceptance probability (Annealing)
            # For simplicity, let's do strict hill climbing first, then maybe relax.
            # Since alpha is integer, strict hill climbing might get stuck.
            # Let's allow small worsening with probability 0.1 to escape local minima.
            
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            elif new_alpha == current_alpha:
                # Keep it, might lead to better structure
                pass
            else:
                # Worse: Revert
                current_triples.append(removed_triple)
                current_masks.append(removed_mask)
                # With some probability, keep the worse state to escape local minima?
                # If we keep it, we must update current_alpha.
                if random.random() < 0.05:
                    current_alpha = new_alpha
                    # Note: We already reverted the triples list above? 
                    # Wait, if I revert, current_triples is back to old state.
                    # If I want to accept the worse state, I should NOT revert.
                    # Let's re-structure:
                    # 1. Remove
                    # 2. Add
                    # 3. Check Alpha
                    # 4. Decide
                    pass # The logic above reverts immediately. Let's fix the flow.
        
        # Refine the local search logic for correctness and escape
        # Let's implement a proper Simulated Annealing step
        
    # --- Final Polish: Try to reduce alpha further on the best solution found ---
    if best_triples:
        final_deadline = time.time() + 5
        current_triples = best_triples[:]
        current_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in current_triples]
        current_alpha = best_alpha
        
        while time.time() < final_deadline:
            if not current_triples:
                break
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_t = current_triples[idx_remove]
            removed_m = current_masks[idx_remove]
            
            # Try to replace with a triple that covers more "holes"
            # Or just random swap
            new_t = random.choice(all_triples)
            if new_t == removed_t:
                continue
                
            # Temporarily remove
            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)
            
            if is_valid_addition(new_t, current_triples, current_masks):
                current_triples.append(new_t)
                current_masks.append((1 << new_t[0]) | (1 << new_t[1]) | (1 << new_t[2]))
                new_alpha = compute_alpha(current_triples)
                
                if new_alpha <= current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                else:
                    # Revert
                    current_triples.pop()
                    current_masks.pop()
                    current_triples.insert(idx_remove, removed_t)
                    current_masks.insert(idx_remove, removed_m)
            else:
                # Revert
                current_triples.insert(idx_remove, removed_t)
                current_masks.insert(idx_remove, removed_m)

    return best_triples

# Quick test to ensure it runs
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0}, Triples: {len(res)}")
    # Verify validity
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            inter = set(res[i]) & set(res[j])
            if len(inter) > 1:
                print(f"Invalid pair: {res[i]} and {res[j]}")
    # Compute alpha
    print(f"Alpha: {compute_alpha(res)}")
