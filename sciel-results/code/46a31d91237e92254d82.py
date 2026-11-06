import random
import time
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    points = list(range(16))
    
    # --- Fast Independence Number Calculation using bitmask branch-and-bound ---
    def compute_alpha(triples):
        """
        Computes the independence number of the hypergraph defined by 'triples'.
        Uses branch-and-bound with bitmasks.
        """
        if not triples:
            return 16
            
        # Precompute masks for the triples
        masks = []
        for a, b, c in triples:
            masks.append((1 << a) | (1 << b) | (1 << c))
        
        # Find minimum vertex cover size using iterative deepening
        # A vertex cover hits every triple
        
        # For small k, enumerate subsets; for larger k, use the fact that
        # if alpha <= 8, then vertex cover >= 8
        # We search for the minimum k where a vertex cover of size k exists
        
        # Quick check: if k=0, only valid if no triples
        # For k >= 1, try to find a cover
        
        # Strategy: for k from 1 to 8, enumerate C(16,k) subsets
        # C(16,1)=16, C(16,2)=120, C(16,3)=560, C(16,4)=1820, C(16,5)=4368, 
        # C(16,6)=8008, C(16,7)=11440, C(16,8)=12870
        # Total: ~40k checks, each O(n_triples) bit ops - fast enough
        
        n_triples = len(masks)
        
        for k in range(0, 9):
            if k == 0:
                if n_triples == 0:
                    return 16
                continue
            
            # Enumerate all subsets of size k
            for subset in itertools.combinations(range(16), k):
                cover_mask = 0
                for p in subset:
                    cover_mask |= (1 << p)
                
                # Check if this hits every triple
                is_cover = True
                for m in masks:
                    if (cover_mask & m) == 0:
                        is_cover = False
                        break
                if is_cover:
                    return 16 - k
        
        # If no cover of size <= 8, then alpha < 8
        # Try to find exact alpha by checking subsets of size alpha
        # For alpha from 1 to 7, check if there's a subset of size alpha with no triple contained
        for alpha in range(1, 8):
            found = False
            for subset in itertools.combinations(range(16), alpha):
                subset_mask = 0
                for p in subset:
                    subset_mask |= (1 << p)
                
                # Check if any triple is contained in this subset
                has_triple = False
                for m in masks:
                    if (subset_mask & m) == m:  # triple m is subset of subset_mask
                        has_triple = True
                        break
                if not has_triple:
                    found = True
                    break
            if found:
                return alpha
        
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
    
    # --- Precompute all possible triples ---
    all_triples = list(itertools.combinations(points, 3))
    
    # --- Phase 1: Build initial solution with structured greedy approach ---
    best_alpha = 17
    best_triples = []
    
    # Multiple restarts with different strategies
    num_restarts = 40
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1
        
        # --- Generate Initial Solution ---
        triples = []
        masks = []
        
        # Use a mix of random and structured approaches
        if restarts_done % 3 == 1:
            # Pure random greedy
            candidate_order = all_triples[:]
            random.shuffle(candidate_order)
        elif restarts_done % 3 == 2:
            # Greedy with preference for covering new points
            candidate_order = all_triples[:]
            random.shuffle(candidate_order)
            # Sort by number of new points they'd cover (approximate)
            covered = set()
            scored = []
            for t in candidate_order:
                new_pts = sum(1 for p in t if p not in covered)
                scored.append((-new_pts, random.random(), t))
            scored.sort()
            candidate_order = [s[2] for s in scored]
        else:
            # Random with slight bias toward balanced coverage
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
        
        # --- Local Search: Simulated Annealing with targeted mutations ---
        local_deadline = time.time() + (100 / num_restarts)
        
        current_triples = triples[:]
        current_masks = masks[:]
        current_alpha = compute_alpha(current_triples)
        
        temperature = 0.3
        cooling_rate = 0.93
        
        while time.time() < local_deadline:
            if not current_triples:
                break
            
            # Choose mutation type
            if random.random() < 0.7:
                # Type 1: Swap - remove one triple, add another
                idx_remove = random.randint(0, len(current_triples) - 1)
                removed_triple = current_triples[idx_remove]
                removed_mask = current_masks[idx_remove]
                
                # Temporarily remove
                current_triples.pop(idx_remove)
                current_masks.pop(idx_remove)
                
                # Try to add a new triple that covers "uncovered" points
                added = False
                for _ in range(150):
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
            else:
                # Type 2: Add-only - try to add a triple without removing
                added = False
                for _ in range(100):
                    new_t = random.choice(all_triples)
                    if is_valid_addition(new_t, current_masks):
                        current_triples.append(new_t)
                        current_masks.append((1 << new_t[0]) | (1 << new_t[1]) | (1 << new_t[2]))
                        added = True
                        break
                
                if not added:
                    continue
        
            # Calculate new alpha
            new_alpha = compute_alpha(current_triples)
            
            # Accept if better, or with some probability if worse
            if new_alpha < current_alpha:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            elif new_alpha == current_alpha:
                pass
            else:
                # Worse: accept with probability exp(-(delta)/temperature)
                delta = new_alpha - current_alpha
                if random.random() < math.exp(-delta / max(temperature, 0.01)):
                    current_alpha = new_alpha
                else:
                    # Revert the last change
                    if len(current_triples) > len(triples):
                        # We added a triple, remove it
                        current_triples.pop()
                        current_masks.pop()
                    else:
                        # We did a swap, need to revert
                        # Find the last added triple and remove it, restore removed
                        if current_triples:
                            current_triples.pop()
                            current_masks.pop()
                            if removed_triple not in current_triples:
                                current_triples.append(removed_triple)
                                current_masks.append(removed_mask)
            
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
