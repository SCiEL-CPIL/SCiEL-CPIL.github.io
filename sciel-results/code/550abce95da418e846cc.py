import random
import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    points = list(range(16))
    all_points_mask = (1 << 16) - 1
    
    # --- Fast Independence Number Calculation using branch-and-bound ---
    def compute_alpha(triples):
        """
        Computes the independence number of the hypergraph defined by 'triples'.
        Uses branch-and-bound: find max subset S with no triple fully contained in S.
        """
        if not triples:
            return 16
        
        # Build adjacency: for each point, which triples contain it
        # For B&B: state = (available_points_mask, remaining_triples_indices)
        # But we need a faster approach.
        
        # Alternative: use the dual. Alpha = 16 - min vertex cover.
        # Find min vertex cover via branch and bound.
        
        n_triples = len(triples)
        if n_triples == 0:
            return 16
        
        # For each point, precompute which triples it hits
        point_hits = [0] * 16
        for i, (a, b, c) in enumerate(triples):
            point_hits[a] |= (1 << i)
            point_hits[b] |= (1 << i)
            point_hits[c] |= (1 << i)
        
        # Branch and bound for minimum vertex cover
        # State: (covered_triples_mask, points_used)
        # We want to minimize points_used such that covered_triples_mask == all_triples_mask
        
        all_triples_mask = (1 << n_triples) - 1
        
        best_cover = 16  # worst case
        
        # Sort points by degree (number of triples they hit) for better pruning
        point_order = sorted(range(16), key=lambda p: bin(point_hits[p]).count('1'), reverse=True)
        
        def branch_and_bound(covered_mask, used, remaining_points_idx):
            nonlocal best_cover
            
            # Prune: if used >= best_cover, stop
            if used >= best_cover:
                return
            
            # If all covered, update best
            if covered_mask == all_triples_mask:
                best_cover = used
                return
            
            # If no remaining points but not all covered
            if remaining_points_idx >= 16:
                return
            
            # Upper bound: even if we add all remaining points, can we beat best?
            # (optimistic: assume each remaining point covers at least 1 new triple)
            remaining_count = 16 - remaining_points_idx
            if used + 0 >= best_cover:
                return
            
            # Try each remaining point
            for i in range(remaining_points_idx, 16):
                p = point_order[i]
                new_covered = covered_mask | point_hits[p]
                if new_covered == covered_mask:
                    # This point doesn't help, skip
                    continue
                branch_and_bound(new_covered, used + 1, i + 1)
            
            # Also try NOT using this point (implicitly handled by trying all subsets)
            # Actually the above loop tries all combinations. Let me restructure.
        
        # Better B&B: recursive with include/exclude
        def bb(covered_mask, used, idx):
            nonlocal best_cover
            
            if used >= best_cover:
                return
            
            if covered_mask == all_triples_mask:
                best_cover = used
                return
            
            if idx >= 16:
                return
            
            p = point_order[idx]
            
            # Option 1: include point p
            new_covered = covered_mask | point_hits[p]
            bb(new_covered, used + 1, idx + 1)
            
            # Option 2: exclude point p
            # Pruning: if even adding all remaining points can't cover uncovered triples
            uncovered = all_triples_mask & ~covered_mask
            if uncovered:
                # Check if remaining points can cover all uncovered
                rem_cover = 0
                for j in range(idx + 1, 16):
                    rem_cover |= point_hits[point_order[j]]
                if (uncovered & ~rem_cover) != 0:
                    # Can't cover all, prune
                    return
            bb(covered_mask, used, idx + 1)
        
        bb(0, 0, 0)
        
        return 16 - best_cover
    
    # --- Helper: Validity Check for Linear Triple System ---
    def is_valid_addition(triple, existing_masks):
        a, b, c = triple
        new_mask = (1 << a) | (1 << b) | (1 << c)
        
        # Check intersection with all existing triples
        for m in existing_masks:
            inter = new_mask & m
            # If intersection has > 1 bit, invalid
            if inter and (inter & (inter - 1)):
                return False
        return True
    
    # --- Phase 1: Build initial solution with greedy approach ---
    best_alpha = 17
    best_triples = []
    
    # Precompute all possible triples
    all_triples = list(itertools.combinations(points, 3))
    all_triple_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in all_triples]
    
    # Multiple restarts
    num_restarts = 50
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1
        
        # --- Generate Initial Solution ---
        # Greedy: Pick random triples that don't conflict.
        triples = []
        masks = []
        
        # Random order
        candidate_order = list(range(len(all_triples)))
        random.shuffle(candidate_order)
        
        for idx in candidate_order:
            t = all_triples[idx]
            if is_valid_addition(t, masks):
                triples.append(t)
                masks.append(all_triple_masks[idx])
                
        current_alpha = compute_alpha(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
        
        # --- Local Search: Simulated Annealing ---
        local_deadline = min(time.time() + (105 / num_restarts), deadline - 2)
        
        current_triples = triples[:]
        current_masks = masks[:]
        current_alpha = compute_alpha(current_triples)
        
        temperature = 1.0
        cooling_rate = 0.98
        
        while time.time() < local_deadline:
            if not current_triples:
                break
                
            # Mutation: Remove one triple, try to add another
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_triple = current_triples[idx_remove]
            removed_mask = current_masks[idx_remove]
            
            # Temporarily remove
            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)
            
            # Try to add a new triple
            added = False
            # Try several random candidates
            for _ in range(200):
                new_idx = random.randint(0, len(all_triples) - 1)
                new_t = all_triples[new_idx]
                if new_t == removed_triple:
                    continue
                if is_valid_addition(new_t, current_masks):
                    current_triples.append(new_t)
                    current_masks.append(all_triple_masks[new_idx])
                    added = True
                    break
            
            if not added:
                # Revert
                current_triples.insert(idx_remove, removed_triple)
                current_masks.insert(idx_remove, removed_mask)
                continue
            
            # Calculate new alpha
            new_alpha = compute_alpha(current_triples)
            
            # Accept if better, or with some probability if worse (simulated annealing)
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            elif new_alpha == current_alpha:
                # Keep it
                pass
            else:
                # Worse: accept with probability exp(-(delta)/temperature)
                delta = new_alpha - current_alpha
                if random.random() < pow(2.718281828, -delta / max(temperature, 0.01)):
                    current_alpha = new_alpha
                else:
                    # Revert
                    current_triples.pop()
                    current_masks.pop()
                    current_triples.insert(idx_remove, removed_triple)
                    current_masks.insert(idx_remove, removed_mask)
            
            # Cool down
            temperature *= cooling_rate
    
    # --- Final Polish ---
    if best_triples and time.time() < deadline - 5:
        final_deadline = min(time.time() + 5, deadline)
        current_triples = best_triples[:]
        current_masks = [(1 << t[0]) | (1 << t[1]) | (1 << t[2]) for t in current_triples]
        current_alpha = best_alpha
        
        while time.time() < final_deadline:
            if not current_triples:
                break
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_t = current_triples[idx_remove]
            removed_m = current_masks[idx_remove]
            
            new_idx = random.randint(0, len(all_triples) - 1)
            new_t = all_triples[new_idx]
            if new_t == removed_t:
                continue
                
            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)
            
            if is_valid_addition(new_t, current_masks):
                current_triples.append(new_t)
                current_masks.append(all_triple_masks[new_idx])
                new_alpha = compute_alpha(current_triples)
                
                if new_alpha <= current_alpha:
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_triples = current_triples[:]
                else:
                    current_triples.pop()
                    current_masks.pop()
                    current_triples.insert(idx_remove, removed_t)
                    current_masks.insert(idx_remove, removed_m)
            else:
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
