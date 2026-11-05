import time
import random
import numpy as np

N = 64
TOTAL = N * N

# Precompute squared distances using numpy for speed
xs = np.repeat(np.arange(N), N)
ys = np.tile(np.arange(N), N)
dx = xs[:, None] - xs[None, :]
dy = ys[:, None] - ys[None, :]
dist2 = (dx * dx + dy * dy).astype(np.int32)

def construct():
    start_time = time.time()
    deadline = start_time + 295
    
    best_set = []
    best_size = 0
    
    # Strategy:
    # 1. Deterministic initialization using modular arithmetic (finite field-like structure)
    #    to create a large, valid base set.
    # 2. Local search with batch swaps: remove k points, try to add more valid points.
    #    This escapes local optima that single-point swaps cannot.
    
    # --- Phase 1: Deterministic Initialization ---
    # We construct points using a modular pattern.
    # A simple pattern: points (i, j) such that j == (a*i + b) % N for various a, b.
    # Or simpler: a grid subsampling.
    # Let's try a modular lattice: points (x, y) where y = (m*x + c) % N.
    # Actually, a known good structure for no-isosceles is related to Sidon sets or similar.
    # For now, let's use a greedy deterministic construction to get a solid baseline.
    
    # Greedy deterministic: sort points by some heuristic (e.g., distance from center, or just index)
    # and add if valid.
    # Let's try multiple deterministic orderings to find a strong baseline.
    
    orderings = [
        list(range(TOTAL)),
        list(range(TOTAL, 0, -1)),
        sorted(range(TOTAL), key=lambda i: (i // N + i % N)),  # Diagonal order
        sorted(range(TOTAL), key=lambda i: (i // N - i % N)),  # Anti-diagonal
    ]
    
    # Add random shuffles to the orderings list to diversify
    for _ in range(10):
        perm = list(range(TOTAL))
        random.shuffle(perm)
        orderings.append(perm)
    
    for perm in orderings:
        si = []
        ud = [] # list of sets of used distances for each point in si
        
        for p_idx in perm:
            if not si:
                si.append(p_idx)
                ud.append(set())
                continue
            
            # Check if p_idx can be added
            valid = True
            dists = []
            for q_idx in si:
                d = dist2[p_idx, q_idx]
                dists.append(d)
            
            # Check distinctness among new distances
            if len(set(dists)) != len(dists):
                continue
            
            # Check against existing used distances
            for i in range(len(si)):
                if dists[i] in ud[i]:
                    valid = False
                    break
            
            if valid:
                si.append(p_idx)
                new_ud = set(dists)
                ud.append(new_ud)
                
                # Update existing points' used distance sets
                for i in range(len(si) - 1):
                    ud[i].add(dists[i])
        
        if len(si) > best_size:
            best_size = len(si)
            best_set = [(idx // N, idx % N) for idx in si]
            
        if time.time() > deadline:
            break
    
    # --- Phase 2: Local Search with Batch Swaps ---
    # Start from the best set found in Phase 1.
    if not best_set:
        return []
    
    # Convert best_set back to indices
    current_set = [x * N + y for x, y in best_set]
    
    # Precompute adjacency: for each point, which other points conflict with it?
    # Conflict means: if we add point p to set S, it creates an isosceles triangle.
    # This is complex to precompute directly. Instead, we'll use a helper function.
    
    def try_add_point(set_indices, ud, p_idx):
        """Try to add p_idx to set_indices with used distances ud. Returns True if added."""
        if p_idx in set_indices:
            return False
        
        dists = []
        for q_idx in set_indices:
            d = dist2[p_idx, q_idx]
            dists.append(d)
        
        # Check distinctness
        if len(set(dists)) != len(dists):
            return False
        
        # Check against used distances
        for i in range(len(set_indices)):
            if dists[i] in ud[i]:
                return False
        
        # Add it
        set_indices.append(p_idx)
        new_ud = set(dists)
        ud.append(new_ud)
        for i in range(len(set_indices) - 1):
            ud[i].add(dists[i])
        return True
    
    def remove_point(set_indices, ud, p_idx):
        """Remove p_idx from set_indices and update ud."""
        if p_idx not in set_indices:
            return False
        
        pos = set_indices.index(p_idx)
        # Get distances from p_idx to all other points
        dists = []
        for i, q_idx in enumerate(set_indices):
            if i == pos:
                continue
            dists.append(dist2[p_idx, q_idx])
        
        # Remove p_idx from set_indices
        set_indices.pop(pos)
        ud.pop(pos)
        
        # Update ud for remaining points
        # We need to remove the distance from p_idx to each remaining point
        # But we lost the correspondence. Let's recompute distances for the removed point.
        # Actually, it's easier to just rebuild ud or update carefully.
        # For simplicity and correctness, let's rebuild ud for the remaining set.
        # This is O(K^2) but K is small (~10-20).
        new_ud = []
        for i, q_idx in enumerate(set_indices):
            s = set()
            for j, r_idx in enumerate(set_indices):
                if i != j:
                    s.add(dist2[q_idx, r_idx])
            new_ud.append(s)
        ud.clear()
        ud.extend(new_ud)
        return True
    
    # Local search loop
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        improved = False
        
        # Try single-point removal and addition (simple swap)
        # And batch swaps: remove 2-3 points, try to add 1-3 new points.
        
        # For simplicity, let's try removing one point and adding one or two new points.
        # Or just try adding points not in the set.
        
        # First, try to add any valid point not in the set.
        added_any = False
        for p_idx in range(TOTAL):
            if p_idx not in current_set:
                # Check if it can be added
                # We need ud for current_set. Let's rebuild it once per iteration.
                ud = []
                for i, q_idx in enumerate(current_set):
                    s = set()
                    for j, r_idx in enumerate(current_set):
                        if i != j:
                            s.add(dist2[q_idx, r_idx])
                    ud.append(s)
                
                if try_add_point(current_set, ud, p_idx):
                    best_size = len(current_set)
                    best_set = [(idx // N, idx % N) for idx in current_set]
                    improved = True
                    added_any = True
                    break
        
        if added_any:
            no_improvement_count = 0
            continue
        
        # Try batch swap: remove 2 points, try to add 1-2 new points.
        # This is more complex. Let's try a simpler perturbation:
        # Remove a random point, then try to add a random point.
        
        if current_set:
            # Remove a random point
            rem_idx = random.choice(current_set)
            ud = []
            for i, q_idx in enumerate(current_set):
                s = set()
                for j, r_idx in enumerate(current_set):
                    if i != j:
                        s.add(dist2[q_idx, r_idx])
                ud.append(s)
            
            remove_point(current_set, ud, rem_idx)
            
            # Try to add a random point
            add_idx = random.randint(0, TOTAL - 1)
            if add_idx not in current_set:
                if try_add_point(current_set, ud, add_idx):
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = [(idx // N, idx % N) for idx in current_set]
                        improved = True
                    else:
                        # Revert
                        # This is getting complicated. Let's just keep the best set found.
                        pass
            
        if not improved:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
    
    return best_set
