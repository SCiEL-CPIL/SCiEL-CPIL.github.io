"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

def construct():
    deadline = time.time() + 135  # Safety margin before 150s limit
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    
    # Precompute all quads and their edge indices
    quads = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        # Edges: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
        e1 = a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
        e2 = a * (n - 1) - a * (a + 1) // 2 + (c - a - 1)
        e3 = a * (n - 1) - a * (a + 1) // 2 + (d - a - 1)
        e4 = b * (n - 1) - b * (b + 1) // 2 + (c - b - 1)
        e5 = b * (n - 1) - b * (b + 1) // 2 + (d - b - 1)
        e6 = c * (n - 1) - c * (c + 1) // 2 + (d - c - 1)
        quads.append((e1, e2, e3, e4, e5, e6))
    quads_arr = np.array(quads, dtype=np.int32)  # shape (1820, 6)
    
    def count_colors_in_quad(edge_colors):
        """Count distinct colors in a 6-edge tuple. Returns array of counts."""
        # For each quad, count distinct colors among 6 edges
        # Use the fact that we need >= 5 distinct colors
        # A quad has < 5 distinct colors iff at least 2 pairs of edges share a color
        # More efficiently: count = 6 - (number of duplicate pairs)
        # But easier: use a vectorized approach
        # For each of the 1820 quads, get the 6 colors
        # Check if any two are equal (which would reduce distinct count)
        # We need distinct_count >= 5, i.e., at most 1 pair of equal colors
        
        # Get all 6 colors for all quads
        c = edge_colors[quads_arr]  # shape (1820, 6)
        
        # Count pairs that are equal
        # There are C(6,2) = 15 pairs
        pair_counts = np.zeros(c.shape[0], dtype=np.int32)
        for i in range(6):
            for j in range(i + 1, 6):
                pair_counts += (c[:, i] == c[:, j]).astype(np.int32)
        
        # distinct_count = 6 - pair_counts (approximately, for small duplicates)
        # Actually: if there are k pairs equal, distinct count is 6 - k (when no triple)
        # For our purposes, we need distinct >= 5, so pair_counts <= 1
        # But this isn't exactly right if there are triples. Let's use a different approach.
        
        # Better: a quad is valid iff NOT (at least 2 pairs share color OR a triple exists)
        # Simplest correct check: distinct count >= 5 iff the number of elements equal to the min,
        # or use: 6 - (number of "extra" occurrences) >= 5
        # Let's just check: is there a color appearing 3+ times, or two colors each appearing 2+ times?
        # If max frequency >= 3: invalid (distinct <= 4)
        # If two colors have freq >= 2: invalid (distinct <= 4)
        
        # Vectorized: for each quad, check if any color appears 3+ times
        # And if any two different colors each appear 2+ times
        
        # Simpler: compute distinct count directly using sorting
        # c_sorted = np.sort(c, axis=1)  # (1820, 6)
        # distinct = 1 + np.sum(c_sorted[:, 1:] != c_sorted[:, :-1], axis=1)
        
        c_sorted = np.sort(c, axis=1)
        distinct = 1 + np.sum(c_sorted[:, 1:] != c_sorted[:, :-1], axis=1)
        
        return distinct
    
    def is_valid_vectorized(coloring):
        """Vectorized validation: returns True if all quads have >= 5 distinct colors."""
        edge_colors = np.array(coloring, dtype=np.int32)
        distinct_counts = count_colors_in_quad(edge_colors)
        return np.all(distinct_counts >= 5)
    
    def make_coloring_formula(formula_fn):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(formula_fn(i, j))
        return colors
    
    def local_search(coloring, max_iters=5000):
        """Try to reduce colors by changing individual edge colors."""
        coloring = list(coloring)
        nc = max(coloring) + 1
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            # Try to reduce the highest color used
            target_color = max(coloring)
            if target_color == 0:
                break
            # Find all edges with this color
            edges_with_color = [i for i, c in enumerate(coloring) if c == target_color]
            if not edges_with_color:
                break
            # Try removing this color by reassigning one edge
            for edge_idx in edges_with_color:
                # Try assigning a lower color
                for new_color in range(target_color):
                    test_coloring = coloring[:]
                    test_coloring[edge_idx] = new_color
                    if is_valid_vectorized(test_coloring):
                        coloring = test_coloring
                        improved = True
                        break
                if improved:
                    break
        return coloring, max(coloring) + 1
    
    candidates = []
    
    # Generate diverse formula families
    # 1. Linear forms (a*i + b*j) % m
    for m in range(13, 20):
        for a in range(1, 10):
            for b in range(1, 10):
                if a == b:
                    continue
                cand = make_coloring_formula(lambda i, j, a=a, b=b, m=m: (a * i + b * j) % m)
                candidates.append(cand)
                
    # 2. (i + j) % m
    for m in range(13, 20):
        cand = make_coloring_formula(lambda i, j, m=m: (i + j) % m)
        candidates.append(cand)
        
    # 3. (i*j) % m
    for m in range(13, 20):
        cand = make_coloring_formula(lambda i, j, m=m: (i * j) % m)
        candidates.append(cand)
        
    # 4. (i*i + j*j) % m
    for m in range(13, 20):
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + j*j) % m)
        candidates.append(cand)
    
    # 5. XOR-based: (i ^ j) % m
    for m in range(13, 20):
        cand = make_coloring_formula(lambda i, j, m=m: (i ^ j) % m)
        candidates.append(cand)
    
    # 6-20. Various linear and nonlinear forms
    for m in range(13, 20):
        for coeff_i in range(1, 10):
            for coeff_j in range(1, 10):
                if coeff_i == coeff_j:
                    continue
                cand = make_coloring_formula(
                    lambda i, j, ci=coeff_i, cj=coeff_j, m=m: (ci * i + cj * j) % m)
                candidates.append(cand)
    
    # More nonlinear forms
    for m in range(13, 20):
        # (i*j + i + j) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*j + i + j) % m)
        candidates.append(cand)
        # (i^2 + j^2 + i*j) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + j*j + i*j) % m)
        candidates.append(cand)
        # (i^2 - j^2) % m
        cand = make_coloring_formula(lambda i, j, m=m: ((i-j)*(i+j)) % m)
        candidates.append(cand)
        # (i*j + i^2 + j^2) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*j + i*i + j*j) % m)
        candidates.append(cand)
        # (i + j + i*j) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i + j + i*j) % m)
        candidates.append(cand)
        # (i^3 + j^3) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*i*i + j*j*j) % m)
        candidates.append(cand)
        # i*j*(i+j) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*j*(i+j)) % m)
        candidates.append(cand)
        # (i^2 + i*j + j^2) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*i + i*j + j*j) % m)
        candidates.append(cand)
        # (i*j + i - j) % m handled as (i*j + i + (m-j)) % m
        cand = make_coloring_formula(lambda i, j, m=m: (i*j + i + (m-j)) % m)
        candidates.append(cand)
    
    # Add some structured constructions
    # Block-based: split vertices into groups
    for m in range(13, 20):
        # Color based on group difference
        cand = []
        for i in range(n):
            for j in range(i+1, n):
                gi = i // 4
                gj = j // 4
                if gi == gj:
                    c = (i % 4) * 5 + (j % 4)
                else:
                    c = gi * 5 + gj
                cand.append(c % m)
        candidates.append(cand)
        
        # Another block structure
        cand = []
        for i in range(n):
            for j in range(i+1, n):
                gi = i // 8
                gj = j // 8
                if gi == gj:
                    c = (i % 8) + (j % 8)
                else:
                    c = (gi * 8 + gj) + (i % 8)
                cand.append(c % m)
        candidates.append(cand)
    
    # Sort candidates by number of colors used (ascending)
    candidates.sort(key=lambda c: max(c) + 1)
    
    best = None
    best_num_colors = float('inf')
    
    # Phase 1: Check formula-based candidates
    for cand in candidates:
        if time.time() > deadline - 10:
            break
        nc = max(cand) + 1
        if nc >= best_num_colors:
            break
        if is_valid_vectorized(cand):
            best_num_colors = nc
            best = cand
    
    # Phase 2: Local search to improve best candidate
    if best is not None:
        if time.time() < deadline - 5:
            improved, nc = local_search(best, max_iters=10000)
            if nc < best_num_colors:
                best = improved
                best_num_colors = nc
    
    # Phase 3: Try more random perturbations if time allows
    if best is not None and time.time() < deadline - 10:
        rng = np.random.default_rng(42)
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            # Start from best and randomly perturb some edges
            perturbed = list(best)
            num_perturb = rng.integers(1, 10)
            for _ in range(num_perturb):
                idx = rng.integers(0, num_edges)
                new_color = rng.integers(0, best_num_colors)
                perturbed[idx] = new_color
            nc = max(perturbed) + 1
            if nc < best_num_colors and is_valid_vectorized(perturbed):
                best = perturbed
                best_num_colors = nc
    
    if best is not None:
        return best
    
    # Fallback: return a known valid coloring with 17 colors
    return make_coloring_formula(lambda i, j: (i + j) % 17)
    
# EVOLVE-BLOCK-END
