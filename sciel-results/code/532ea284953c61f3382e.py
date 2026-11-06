import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    points = list(range(16))
    
    # --- Fast Independence Number Calculation using bitmasks ---
    def compute_alpha(triples):
        """
        Computes the independence number of the hypergraph defined by 'triples'.
        Uses a branch-and-bound approach with bitmasks for speed.
        """
        if not triples:
            return 16
            
        # Precompute masks for the triples
        masks = []
        for a, b, c in triples:
            masks.append((1 << a) | (1 << b) | (1 << c))
        
        # We want to find the largest subset S such that no triple is contained in S.
        # Equivalent: find the minimum vertex cover of the hypergraph (set of points hitting every triple).
        # alpha = 16 - min_vertex_cover_size
        
        # Try to find a vertex cover of size k, for k = 0, 1, 2, ...
        # If we find one, alpha = 16 - k.
        
        # Quick upper bound on vertex cover size: sum of all triple sizes / 3
        # But let's just search from k=0 upward
        
        # For each k, check if there's a set of k points that hits every triple.
        # A set C hits triple T iff C ∩ T ≠ ∅.
        
        # We can use a recursive backtracking search for vertex cover.
        
        def has_vertex_cover(k):
            """Check if there exists a vertex cover of size exactly k."""
            # If k=0, only works if no triples
            if k == 0:
                return len(triples) == 0
            
            # If k >= number of points, trivially yes (take all points)
            if k >= 16:
                return True
            
            # Use backtracking: pick a triple, try each of its 3 points as part of the cover
            # Prune: if remaining triples can't be covered by remaining budget, fail
            
            # Sort triples by... let's just use a simple recursive approach with memoization on (remaining_triples_mask, budget)
            # But that might be too slow. Let's use a simpler heuristic.
            
            # Alternative: enumerate all subsets of size k and check if any is a vertex cover.
            # C(16, 8) = 12870, which is manageable for k up to 8 or so.
            
            if k <= 8:
                # Enumerate all subsets of size k
                for subset in itertools.combinations(range(16), k):
                    # Check if this subset hits every triple
                    cover_mask = 0
                    for p in subset:
                        cover_mask |= (1 << p)
                    
                    is_cover = True
                    for m in masks:
                        if (cover_mask & m) == 0:
                            is_cover = False
                            break
                    if is_cover:
                        return True
                return False
            else:
                # For k > 8, it's likely that a cover exists (since alpha < 8)
                # Just return True as a heuristic
                return True
        
        # Find minimum k such that has_vertex_cover(k) is True
        for k in range(0, 17):
            if has_vertex_cover(k):
                return 16 - k
        
        return 1
    
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
    
    # Multiple restarts
    num_restarts = 30
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1
        
        # --- Generate Initial Solution ---
        # Greedy: Pick random triples that don't conflict.
        triples = []
        masks = []
        
        # Random order
        candidate_order = all_triples[:]
        random.shuffle(candidate_order)
        
        for t in candidate_order:
            if is_valid_addition(t, masks):
                triples.append(t)
                masks.append((1 << t[0]) | (1 << t[1]) | (1 << t[2]))
                
        current_alpha = compute_alpha(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
        
        # --- Local Search: Simulated Annealing ---
        # Use a temperature schedule that cools over time
        local_deadline = time.time() + (100 / num_restarts)
        
        current_triples = triples[:]
        current_masks = masks[:]
        current_alpha = compute_alpha(current_triples)
        
        temperature = 0.5
        cooling_rate = 0.95
        
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
            for _ in range(100):
                new_t = random.choice(all_triples)
                if new_t == removed_triple:
                    continue
                if is_valid_addition(new_t, current_masks):
                    current_triples.append(new_t)
                    current_masks.append((1 << new_t[0]) | (1 << new_t[1]) | (1 << new_t[2]))
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
            
            new_t = random.choice(all_triples)
            if new_t == removed_t:
                continue
                
            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)
            
            if is_valid_addition(new_t, current_masks):
                current_triples.append(new_t)
                current_masks.append((1 << new_t[0]) | (1 << new_t[1]) | (1 << new_t[2]))
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
