import random
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for safety
    
    points = list(range(16))
    N = 16
    
    # Precompute all possible triples as masks and tuples
    all_triples = []
    all_triple_masks = []
    for combo in itertools.combinations(points, 3):
        a, b, c = combo
        mask = (1 << a) | (1 << b) | (1 << c)
        all_triples.append(combo)
        all_triple_masks.append(mask)
        
    # Map from mask to index for quick lookup? Not strictly needed if we check validity via masks.
    # We need a fast way to check if a triple can be added to a set of triples.
    # Linearity constraint: any two triples share at most 1 point.
    # This is equivalent to: for any two triples T1, T2, |T1 ∩ T2| <= 1.
    # In terms of masks: (mask1 & mask2) must have popcount <= 1.
    
    def popcount(x):
        return bin(x).count('1')
        
    def is_valid_addition(new_mask, existing_masks):
        for m in existing_masks:
            inter = new_mask & m
            if popcount(inter) > 1:
                return False
        return True

    # --- Fast Independence Number Calculator ---
    # We want the size of the largest subset S of {0..15} such that no triple is fully contained in S.
    # This is the Maximum Independent Set (MIS) problem on the 3-uniform hypergraph.
    # Equivalently, it is the complement of the Minimum Vertex Cover (MVC) of the hypergraph.
    # Let alpha be the independence number.
    # We can compute this exactly using a branch and bound algorithm on the hypergraph.
    # Given the small size (16 vertices), a well-optimized recursive search is feasible.
    
    def compute_alpha_exact(triples):
        if not triples:
            return 16
            
        # Build adjacency lists for the hypergraph?
        # Or use a bitset representation for the "conflict" graph of the hypergraph?
        # Actually, we are looking for a set of vertices that contains NO edge.
        # This is exactly the Independent Set problem in the hypergraph.
        
        # Let's use a recursive backtracking with pruning.
        # State: (remaining_vertices_mask, current_size)
        # We want to maximize current_size.
        
        # Precompute for each vertex, the set of triples containing it?
        # No, we need to know if a set of vertices contains a triple.
        
        # Alternative: Use the complement.
        # A set S is independent if for every triple T, T is not a subset of S.
        # This is equivalent to: For every triple T, at least one vertex of T is NOT in S.
        # Let U = {0..15}. Let C = U \ S.
        # Then C must be a Vertex Cover of the hypergraph.
        # We want to minimize |C| to maximize |S| = 16 - |C|.
        # So, compute Minimum Vertex Cover size.
        
        triples_masks = [(1 << a) | (1 << b) | (1 << c) for a, b, c in triples]
        
        # Branch and Bound for Minimum Vertex Cover
        # We select vertices to add to the cover C.
        # If all triples are covered (i.e., for every triple, C intersects it), we have a valid cover.
        
        best_cover_size = 16 # Worst case
        
        # Sort triples by frequency? Or just use them as is.
        # Optimization: If a triple is a subset of the union of other triples? No, they are all size 3.
        
        def min_vertex_cover(triples_masks, current_cover_mask, depth):
            nonlocal best_cover_size
            
            # Pruning: if current depth + lower_bound > best_cover_size, return
            # Lower bound: We need to cover the remaining triples.
            # A simple lower bound is the number of disjoint remaining triples.
            # Or just depth + 1 if there are uncovered triples?
            
            # Check if all triples are covered
            all_covered = True
            for tm in triples_masks:
                if (tm & current_cover_mask) == 0:
                    all_covered = False
                    break
                    
            if all_covered:
                if depth < best_cover_size:
                    best_cover_size = depth
                return
                
            # If we have already selected more than best, stop
            if depth >= best_cover_size:
                return
                
            # Find an uncovered triple
            # To improve branching, pick the uncovered triple with the fewest available vertices?
            # All triples have 3 vertices.
            # Pick the first uncovered triple.
            target_triple = None
            for tm in triples_masks:
                if (tm & current_cover_mask) == 0:
                    target_triple = tm
                    break
                    
            if target_triple is None:
                return # Should be handled by all_covered check
            
            # Branch on the 3 vertices of this triple
            # We must pick at least one vertex from this triple to cover it.
            # We can pick v1, or v2, or v3.
            
            # Get vertices of target_triple
            verts = []
            for i in range(16):
                if target_triple & (1 << i):
                    verts.append(i)
                    
            for v in verts:
                new_cover_mask = current_cover_mask | (1 << v)
                min_vertex_cover(triples_masks, new_cover_mask, depth + 1)
                
            # Optimization: If we picked v1, we don't need to explore picking v2 or v3 
            # if the set of remaining uncovered triples is the same? No, that's not right.
            # However, we can use symmetry breaking or dominance.
            # For now, simple branching is fast enough for 16 nodes if the cover is small.
            
        min_vertex_cover(triples_masks, 0, 0)
        return 16 - best_cover_size

    # --- Initialization: Algebraic Construction ---
    # We start with a known good structure.
    # Let's try to construct a linear triple system from AG(2,4).
    # AG(2,4) has 16 points and 21 lines of size 4.
    # We can split each line of 4 into two triples? No, that would create overlaps.
    # If we take a line {a,b,c,d}, we can take {a,b,c} and {a,b,d}? They share {a,b}, invalid.
    # We can take {a,b,c} and {d, ...}? 
    # Actually, a standard way to get a 2-(16, 3, 1) design is not possible (lambda=1 requires v(v-1)/6 = 16*15/6 = 40 triples).
    # A 2-(16, 3, 1) design exists? 16 is not a valid v for Steiner Triple System (v=1,3,7,9,13,15,21...).
    # So we cannot have a perfect packing where every pair appears exactly once.
    # But we want to minimize independence number.
    # High density is good.
    
    # Let's use a greedy construction with a specific ordering to ensure linearity.
    # We will use multiple random restarts of a "Greedy + Local Search" approach, 
    # but with the EXACT alpha calculator to guide the search properly.
    
    best_alpha = 17
    best_triples = []
    
    # To speed up is_valid_addition, we can maintain a list of masks.
    # For local search, we need to be able to remove and add quickly.
    
    num_restarts = 50
    restarts_done = 0
    
    while time.time() < deadline and restarts_done < num_restarts:
        restarts_done += 1
        
        triples = []
        masks = []
        
        # Randomized Greedy Construction
        # Shuffle all triples
        indices = list(range(len(all_triples)))
        random.shuffle(indices)
        
        for idx in indices:
            t = all_triples[idx]
            m = all_triple_masks[idx]
            if is_valid_addition(m, masks):
                triples.append(t)
                masks.append(m)
                
        # Calculate initial alpha
        current_alpha = compute_alpha_exact(triples)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_triples = triples[:]
            
        # Local Search: Simulated Annealing
        # We want to reduce alpha.
        # Moves:
        # 1. Remove a triple.
        # 2. Add a different triple that fits.
        # 3. Swap: Remove one, add two? (Complex)
        # Let's stick to single-triple swaps first.
        
        current_triples = triples[:]
        current_masks = masks[:]
        current_alpha = compute_alpha_exact(current_triples)
        
        # Time slice for local search
        local_deadline = time.time() + 2.0
        
        while time.time() < local_deadline:
            if not current_triples:
                break
                
            # Pick a random triple to remove
            idx_remove = random.randint(0, len(current_triples) - 1)
            removed_t = current_triples[idx_remove]
            removed_m = current_masks[idx_remove]
            
            # Remove it
            current_triples.pop(idx_remove)
            current_masks.pop(idx_remove)
            
            # Try to find a replacement that improves or maintains alpha
            # We can sample a few candidates
            found_replacement = False
            new_alpha = current_alpha
            
            # Sample 100 random candidates
            for _ in range(100):
                new_idx = random.randint(0, len(all_triples) - 1)
                new_t = all_triples[new_idx]
                new_m = all_triple_masks[new_idx]
                
                if new_t == removed_t:
                    continue
                    
                if is_valid_addition(new_m, current_masks):
                    # Add it
                    current_triples.append(new_t)
                    current_masks.append(new_m)
                    
                    # Calculate new alpha
                    # This is the expensive part. 
                    # Optimization: If the new triple doesn't affect the "critical" independent set, alpha might stay same.
                    # But we must calculate to be sure.
                    new_alpha = compute_alpha_exact(current_triples)
                    
                    if new_alpha < current_alpha:
                        found_replacement = True
                        break
                    elif new_alpha == current_alpha:
                        # Accept with some probability to explore?
                        # For now, accept if it's the same, to keep density up.
                        # But we want to minimize alpha.
                        # If we accept same alpha, we might get stuck in a plateau.
                        # Let's accept same alpha with prob 0.5
                        if random.random() < 0.5:
                            found_replacement = True
                            break
                        else:
                            # Revert
                            current_triples.pop()
                            current_masks.pop()
                    else:
                        # Worse, revert
                        current_triples.pop()
                        current_masks.pop()
                        
            if found_replacement:
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = current_triples[:]
            else:
                # Revert the removal
                current_triples.insert(idx_remove, removed_t)
                current_masks.insert(idx_remove, removed_m)
                
    # Final check
    if best_triples:
        # Verify validity one last time
        valid = True
        masks_final = [(1 << a) | (1 << b) | (1 << c) for a, b, c in best_triples]
        for i in range(len(masks_final)):
            for j in range(i+1, len(masks_final)):
                if popcount(masks_final[i] & masks_final[j]) > 1:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to a simple valid system if something went wrong
            # This should not happen, but better safe than sorry
            return []
            
    return best_triples

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time() - t0}, Triples: {len(res)}")
    if res:
        # Verify
        masks_final = [(1 << a) | (1 << b) | (1 << c) for a, b, c in res]
        valid = True
        for i in range(len(masks_final)):
            for j in range(i+1, len(masks_final)):
                if bin(masks_final[i] & masks_final[j]).count('1') > 1:
                    valid = False
                    break
            if not valid: break
        print(f"Valid: {valid}")
        
        # Compute alpha for display
        # Note: compute_alpha_exact is local to construct, so we can't call it directly here
        # unless we refactor. For testing, let's just print the length.
        print(f"Result Length: {len(res)}")
