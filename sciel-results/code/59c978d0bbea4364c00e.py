import time
import numpy as np
from itertools import combinations

# Precompute all C6 cycles for a given n
# A C6 is a set of 6 vertices. The edges are (v0,v1), (v1,v2), ..., (v5,v0).
# We need to map these vertex pairs to edge indices in the lexicographic list.
# Edge index for (i,j) with i<j is i*n - i*(i+1)/2 + (j-i-1) ? 
# Let's use a helper to get edge index.
def get_edge_index(n, i, j):
    if i > j:
        i, j = j, i
    # Number of edges starting from k < i is sum_{k=0}^{i-1} (n - 1 - k)
    # This formula is tricky. Let's just precompute a lookup table for n <= 20.
    pass

# Since n is small (<=20), we can precompute the mapping for each n.
# But we don't know n in advance. We'll generate C6s for specific n values.

def generate_c6s(n):
    """
    Returns a numpy array of shape (M, 6) where each row is a C6 represented by its 6 edge indices.
    """
    # Precompute edge index mapping
    edge_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_map[(i, j)] = idx
            idx += 1
    
    c6_list = []
    # Iterate over all combinations of 6 vertices
    for verts in combinations(range(n), 6):
        # There are 5 distinct cycles on 6 labeled vertices (up to rotation/reflection? No, 5! / 2 = 60 cycles? No.)
        # Number of distinct cycles on 6 labeled vertices is (6-1)! / 2 = 60.
        # We can generate them by permuting the 5 vertices relative to the first.
        # Fix v0. Permute v1..v5.
        # To avoid duplicates, we can just generate all 5! permutations of v1..v5,
        # but note that reverse order is the same cycle.
        # Actually, for a fixed set of 6 vertices, there are 5! = 120 Hamiltonian cycles.
        # Since the graph is undirected, each cycle is counted twice (once in each direction).
        # So there are 60 distinct cycles.
        
        v0 = verts[0]
        others = list(verts[1:])
        
        # Generate all permutations of others
        # This is 120 permutations.
        # We can optimize by only taking half, but 120 is small.
        from itertools import permutations
        for perm in permutations(others):
            # Cycle: v0 - perm[0] - perm[1] - perm[2] - perm[3] - perm[4] - v0
            # Edges:
            # (v0, perm[0]), (perm[0], perm[1]), ..., (perm[4], v0)
            
            # To avoid duplicate cycles (reverse), we can impose an order.
            # E.g., perm[0] < perm[4] (indices in the original set? No, vertex labels).
            # Actually, simpler: just add all 120. When checking for monochromaticity,
            # duplicates don't hurt correctness, just efficiency. 
            # But 120 * 12376 (for n=17) = 1.4M cycles. Still fine.
            # Let's just generate all 120 to be safe and simple, or filter.
            # Filtering: ensure perm[0] < perm[4] to cut duplicates in half.
            if perm[0] > perm[4]:
                continue
            
            edges = []
            try:
                edges.append(edge_map[(v0, perm[0])])
                edges.append(edge_map[(perm[0], perm[1])])
                edges.append(edge_map[(perm[1], perm[2])])
                edges.append(edge_map[(perm[2], perm[3])])
                edges.append(edge_map[(perm[3], perm[4])])
                edges.append(edge_map[(perm[4], v0)])
            except KeyError:
                # Should not happen
                continue
            
            c6_list.append(edges)
    
    return np.array(c6_list, dtype=np.int32)

# Cache for generated C6s
_c6_cache = {}

def get_c6s(n):
    if n not in _c6_cache:
        _c6_cache[n] = generate_c6s(n)
    return _c6_cache[n]

def count_violations(n, colors, c6s):
    """
    colors: numpy array of shape (E,)
    c6s: numpy array of shape (M, 6)
    Returns number of monochromatic C6s.
    """
    if len(c6s) == 0:
        return 0
    # Get colors for all edges in all cycles
    # c6s is (M, 6), colors is (E,)
    # colors[c6s] gives (M, 6)
    cycle_colors = colors[c6s]
    # Check if all 6 are the same
    # Compare with the first element in each row
    first_col = cycle_colors[:, 0:1]
    is_mono = np.all(cycle_colors == first_col, axis=1)
    return np.sum(is_mono)

def find_violating_cycle(n, colors, c6s, rng):
    """
    Returns the index of a random violating cycle, or -1 if none.
    """
    if len(c6s) == 0:
        return -1
    cycle_colors = colors[c6s]
    first_col = cycle_colors[:, 0:1]
    is_mono = np.all(cycle_colors == first_col, axis=1)
    violating_indices = np.where(is_mono)[0]
    if len(violating_indices) == 0:
        return -1
    return violating_indices[rng.integers(len(violating_indices))]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We want to maximize n.
    # Try n from 17 down to 10.
    # If we find a valid coloring for n, we return it.
    # If we run out of time, we return the best valid one found so far.
    
    best_result = None
    best_n = 0
    
    # Precompute C6s for candidate n values
    # We'll try 17, 16, 15, 14, 13, 12, 11, 10
    target_ns = [17, 16, 15, 14, 13, 12, 11, 10]
    
    # Precompute C6s for all targets to avoid overhead during search
    # This might take a few seconds.
    for n in target_ns:
        if time.time() > deadline:
            break
        if n not in _c6_cache:
            _c6_cache[n] = generate_c6s(n)
            
    # Search for each n
    for n in target_ns:
        if time.time() > deadline:
            break
            
        c6s = _c6_cache.get(n, None)
        if c6s is None:
            continue
            
        m = n * (n - 1) // 2
        
        # Try multiple restarts
        # Each restart: random init, then local search
        found_valid = False
        best_colors_for_n = None
        min_violations = m # Large number
        
        # Number of restarts depends on time
        # Let's do time-sliced restarts
        while time.time() < deadline - 5:
            # Random initialization
            rng = np.random.default_rng()
            colors = rng.integers(0, 4, size=m)
            
            # Local search
            # We will try to reduce violations to 0
            current_violations = count_violations(n, colors, c6s)
            
            if current_violations == 0:
                best_result = {'n': n, 'colors': colors.tolist()}
                return best_result
            
            # If this is the best so far for this n, save it
            if current_violations < min_violations:
                min_violations = current_violations
                best_colors_for_n = colors.copy()
                
            # Local search loop
            # Use a simple hill climbing: pick a violating cycle, flip an edge
            # To escape local minima, we can allow some "bad" moves or just restart.
            # Given the time budget, restarts are better.
            # Let's do a limited number of steps per restart to allow more restarts.
            max_steps = 500
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                # Find a violating cycle
                viol_idx = find_violating_cycle(n, colors, c6s, rng)
                if viol_idx == -1:
                    # Should not happen if current_violations > 0, but just in case
                    best_result = {'n': n, 'colors': colors.tolist()}
                    return best_result
                    
                # Get the cycle edges
                cycle_edges = c6s[viol_idx]
                # Pick a random edge in the cycle
                edge_pos = rng.integers(6)
                edge_idx = cycle_edges[edge_pos]
                
                # Flip color
                old_color = colors[edge_idx]
                new_color = (old_color + 1 + rng.integers(0, 3)) % 4
                colors[edge_idx] = new_color
                
                # Check if we improved or stayed same?
                # Just accept if it doesn't increase violations too much?
                # Simpler: Accept if it reduces violations.
                # If it increases, revert.
                new_violations = count_violations(n, colors, c6s)
                if new_violations < current_violations:
                    current_violations = new_violations
                    if current_violations < min_violations:
                        min_violations = current_violations
                        best_colors_for_n = colors.copy()
                    if current_violations == 0:
                        best_result = {'n': n, 'colors': colors.tolist()}
                        return best_result
                else:
                    # Revert
                    colors[edge_idx] = old_color
                    
                # If we get stuck, break and restart
                # This simple hill climbing often gets stuck.
                # We could add a "kick" where we randomly flip K edges.
                # But let's rely on restarts for now.
                # To avoid infinite loops on stuck states, we can check if we've made progress in last 50 steps.
                # For simplicity, just run max_steps and restart.
                
        # If we finished the time slice for this n without finding a valid one,
        # and we have a best_colors_for_n, we might want to keep it?
        # But the task requires a VALID coloring. Invalid ones score 0.
        # So we only return valid ones.
        
    # If no valid coloring found for higher n, try to ensure we return something valid
    # The seed program returned n=9 or 10.
    # Let's do a quick check for n=10, 11 if we haven't found anything.
    # Actually, the loop above handles this. If it fails for 17..11, it tries 10.
    # If it fails for 10 (unlikely), we return the best valid found in previous iterations?
    # No, we only return valid ones.
    
    # Fallback: Return a known valid small coloring?
    # n=4 is trivially valid (no C6).
    # n=9 is valid (seed).
    # Let's just return the best valid one we found.
    if best_result is not None:
        return best_result
        
    # If we found nothing (should not happen), return a trivial valid coloring
    # n=5 is valid.
    n = 5
    m = n * (n-1) // 2
    colors = [0]*m
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
