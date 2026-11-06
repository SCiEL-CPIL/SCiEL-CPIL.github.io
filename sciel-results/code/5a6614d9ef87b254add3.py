"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    # Deadline guard: return best result found before 145s wall-clock
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    # Triangular lattice neighbors in (a, b) coordinates
    # Distance squared between (a1,b1) and (a2,b2): (a1-a2)^2 + (a1-a2)(b1-b2) + (b1-b2)^2
    # Neighbors at distance 1:
    neighbors = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Build candidate pool: hexagonal region of radius 4 (37 points)
    # This is enough to contain the optimal 32-point cluster
    candidates = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if a*a + a*b + b*b <= 16:  # radius 4 hexagon
                candidates.append((a, b))
    
    n_cand = len(candidates)  # should be 37
    
    # Map from point to index
    point_to_idx = {p: i for i, p in enumerate(candidates)}
    
    # Precompute adjacency: adj[i] = set of j such that candidates[i] and candidates[j] are distance 1
    adj = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        a1, b1 = candidates[i]
        for da, db in neighbors:
            nb = (a1 + da, b1 + db)
            if nb in point_to_idx:
                j = point_to_idx[nb]
                adj[i].add(j)
    
    # Also precompute all pairs at distance >= 1 (i.e., not the same point)
    # Actually we need: mutual distance at least 1, meaning no two selected points can be at distance < 1
    # On the triangular lattice, the minimum distance between distinct lattice points is 1.
    # So the constraint "mutual distance at least 1" is automatically satisfied for distinct points.
    # Wait, let me re-read: "mutual distance at least one" - this means all pairs must have distance >= 1.
    # Since all distinct lattice points have distance >= 1, this is automatically satisfied.
    # The objective is to maximize the number of pairs at distance exactly 1.
    
    # So the problem is: choose 32 distinct points from the lattice (|a|,|b| <= 64)
    # to maximize the number of edges (pairs at distance 1) in the induced subgraph.
    
    # Since the optimal solution is a connected cluster (Harborth's construction is a spiral/hexagonal shape),
    # we can restrict to a finite candidate pool. Radius 4 hexagon has 37 points, enough for 32.
    
    # Use simulated annealing / local search on the bitmask
    import random
    
    def count_edges(mask):
        """Count number of edges in the induced subgraph defined by mask."""
        count = 0
        for i in range(n_cand):
            if mask & (1 << i):
                for j in adj[i]:
                    if j > i and (mask & (1 << j)):
                        count += 1
        return count
    
    def count_selected(mask):
        return bin(mask).count('1')
    
    # Initialize with a good starting point: the 32 points closest to origin in a hexagonal spiral
    # Or just start with the first 32 candidates and improve
    
    # Let's start with a greedy initialization
    # Pick the 32 points that form the densest cluster
    
    # Actually, let's just start with all 37 points minus 5, and use local search
    # Better: start with a specific good configuration
    
    # Greedy: start empty, add points that maximize edge gain
    mask = 0
    selected = set()
    remaining = set(range(n_cand))
    
    for _ in range(32):
        best_gain = -1
        best_point = None
        for pt in remaining:
            gain = len(adj[pt] & selected)
            if gain > best_gain:
                best_gain = gain
                best_point = pt
        if best_point is not None:
            mask |= (1 << best_point)
            selected.add(best_point)
            remaining.discard(best_point)
    
    best_mask = mask
    best_score = count_edges(best_mask)
    
    # Local search: try swapping one selected point with one unselected point
    def try_swap(mask, selected, unselected):
        """Try all single swaps and return improved mask if found."""
        current_score = count_edges(mask)
        for sel in list(selected):
            for unsel in list(unselected):
                # Check if swap is valid (no two points at distance < 1, which is automatic)
                new_mask = mask
                new_mask &= ~(1 << sel)
                new_mask |= (1 << unsel)
                new_score = count_edges(new_mask)
                if new_score > current_score:
                    return new_mask, new_score
        return None, current_score
    
    # Simulated annealing
    import random
    random.seed(42)
    
    current_mask = best_mask
    current_score = best_score
    selected = set()
    unselected = set()
    for i in range(n_cand):
        if current_mask & (1 << i):
            selected.add(i)
        else:
            unselected.add(i)
    
    temperature = 5.0
    cooling = 0.995
    min_temp = 0.01
    
    while temperature > min_temp:
        if time.time() > DEADLINE - 5:
            break
        
        # Pick a random swap
        if selected and unselected:
            sel = random.choice(list(selected))
            unsel = random.choice(list(unselected))
            new_mask = current_mask
            new_mask &= ~(1 << sel)
            new_mask |= (1 << unsel)
            new_score = count_edges(new_mask)
            
            delta = new_score - current_score
            if delta >= 0 or random.random() < math.exp(delta / temperature):
                current_mask = new_mask
                current_score = new_score
                selected.discard(sel)
                selected.add(unsel)
                unselected.discard(unsel)
                unselected.add(sel)
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
        
        temperature *= cooling
    
    # Final local search pass
    selected = set()
    unselected = set()
    for i in range(n_cand):
        if best_mask & (1 << i):
            selected.add(i)
        else:
            unselected.add(i)
    
    improved = True
    while improved:
        if time.time() > DEADLINE - 5:
            break
        improved = False
        new_mask, new_score = try_swap(best_mask, selected, unselected)
        if new_mask is not None:
            best_mask = new_mask
            best_score = new_score
            selected = set()
            unselected = set()
            for i in range(n_cand):
                if best_mask & (1 << i):
                    selected.add(i)
                else:
                    unselected.add(i)
            improved = True
    
    # Convert back to list of [a, b] pairs
    result = []
    for i in range(n_cand):
        if best_mask & (1 << i):
            result.append([candidates[i][0], candidates[i][1]])
    
    # Verify we have exactly 32 points
    assert len(result) == 32, f"Expected 32 points, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
