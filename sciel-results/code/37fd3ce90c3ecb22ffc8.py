"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    Uses a GRASP heuristic with optimized conflict checking and more iterations.
    """
    best_selected = None
    best_size = float('inf')
    
    # Run multiple iterations to find a smaller set
    # Use a time limit to ensure we finish within 30s
    start_time = time.time()
    max_time = 25.0  # Leave 5s buffer
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        selected = []
        sums = set()
        available = set(range(1, N + 1))
        
        # Precompute nothing for now, as selected changes
        
        while available:
            candidates = []
            
            # Evaluate each candidate in 'available'
            for v in available:
                # 1. Check if v is compatible with current 'selected'
                new_sums_v = {2 * v}
                for s in selected:
                    new_sums_v.add(v + s)
                
                # If any new sum collides with existing sums, v is invalid
                if sums.intersection(new_sums_v):
                    continue
                
                # v is a valid candidate to add.
                # Calculate 'degree': how many other available elements u
                # become incompatible if we add v?
                
                blocked = set()
                
                # Precompute pair sums of selected for faster lookup
                # pair_sums = {s1 + s2 for s1 in selected for s2 in selected}
                # This is O(|S|^2) per v, which is expensive.
                # Let's stick to the loop but optimize if possible.
                
                # Condition 2: u = (v+s)/2
                for s in selected:
                    val = v + s
                    if val % 2 == 0:
                        u = val // 2
                        if u in available:
                            blocked.add(u)
                
                # Condition 3: u = 2v - s
                for s in selected:
                    u = 2 * v - s
                    if u in available:
                        blocked.add(u)
                
                # Condition 4: u = v + s' - s
                for s in selected:
                    for sp in selected:
                        u = v + sp - s
                        if u in available:
                            blocked.add(u)
                
                # Condition 5: u = s' + s'' - v
                for s in selected:
                    for sp in selected:
                        u = s + sp - v
                        if u in available:
                            blocked.add(u)
                
                # Remove v itself from blocked if it was added
                blocked.discard(v)
                
                degree = len(blocked)
                candidates.append((degree, v))
            
            if not candidates:
                # No valid vertex, set is maximal
                break
            
            # Sort by degree descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # GRASP: Pick from top K candidates
            # Dynamic K: larger K for more exploration, smaller for exploitation
            # Let's try a fixed small K or random K
            k = max(1, min(5, len(candidates) // 5))
            chosen = random.choice(candidates[:k])
            best_v = chosen[1]
            
            # Update state with best_v
            selected.append(best_v)
            
            # Update sums
            sums.add(2 * best_v)
            for s in selected[:-1]:
                sums.add(best_v + s)
            
            # Update available: remove best_v and any u incompatible with new selected
            new_available = set()
            for u in available:
                if u == best_v:
                    continue
                
                # Check if u is compatible with new selected
                if 2 * u in sums:
                    continue
                
                ok = True
                for s in selected:
                    if (u + s) in sums:
                        ok = False
                        break
                
                if ok:
                    new_available.add(u)
            
            available = new_available
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = selected[:]
            
            # Early exit if we find a very small set?
            # Theoretical lower bound?
            # For N=256, max Sidon size is ~16. Min maximal is likely ~5-8?
            if best_size <= 5:
                break

    return best_selected if best_selected else [1]
# EVOLVE-BLOCK-END
