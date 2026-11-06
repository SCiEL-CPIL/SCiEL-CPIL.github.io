"""
Combined approach: Simulated annealing for finding valid colorings + 
backtracking/local search for color minimization.
Uses incremental violation counting for fast SA, and constraint propagation
for exact minimization.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    N = 128

    # Precompute all 4-APs (0-indexed): (a, a+d, a+2d, a+3d) with a+3d < N
    aps = []
    ap_indices = [[] for _ in range(N)]
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
            ap_id = len(aps)
            aps.append((i0, i1, i2, i3))
            ap_indices[i0].append(ap_id)
            ap_indices[i1].append(ap_id)
            ap_indices[i2].append(ap_id)
            ap_indices[i3].append(ap_id)

    def count_violations(coloring):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        violations = 0
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                violations += 1
        return violations

    def validate(coloring):
        """Check every nonconstant 4-term AP has >= 3 distinct colors."""
        for ap in aps:
            c0, c1, c2, c3 = coloring[ap[0]], coloring[ap[1]], coloring[ap[2]], coloring[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                return False
        return True

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def count_colors(colors):
        return len(set(colors))

    def get_forbidden(colors, pos, k):
        """Get set of colors forbidden at position pos given current coloring (for backtracking)."""
        forbidden = set()
        
        # pos as 4th term: a, a+d, a+2d, pos where pos = a+3d => a = pos-3d
        for d in range(1, pos // 3 + 1):
            a = pos - 3 * d
            if a < 0:
                continue
            b, c = a + d, a + 2 * d
            s = {colors[a], colors[b], colors[c]}
            if len(s) <= 2:
                forbidden |= s
        
        # pos as 3rd term: a, a+d, pos, a+3d where pos = a+2d => a = pos-2d
        for d in range(1, min(pos // 2, (N - 1 - pos)) + 1):
            a = pos - 2 * d
            if a < 0:
                continue
            b = a + d  # = pos - d
            e = a + 3 * d  # = pos + d
            if e >= N:
                continue
            s = {colors[a], colors[b], colors[e]}
            if len(s) <= 2:
                forbidden |= s
        
        # pos as 2nd term: a, pos, a+2d, a+3d where pos = a+d => a = pos-d
        for d in range(1, min(pos, (N - 1 - pos) // 2) + 1):
            a = pos - d
            if a < 0:
                continue
            c = pos + d  # a+2d
            e = pos + 2 * d  # a+3d
            if e >= N:
                continue
            s = {colors[a], colors[c], colors[e]}
            if len(s) <= 2:
                forbidden |= s
        
        # pos as 1st term: pos, pos+d, pos+2d, pos+3d
        for d in range(1, (N - 1 - pos) // 3 + 1):
            b = pos + d
            c = pos + 2 * d
            e = pos + 3 * d
            s = {colors[b], colors[c], colors[e]}
            if len(s) <= 2:
                forbidden |= s
        
        return forbidden

    def try_k_colors_sa(k, time_budget=8.0, rng=None):
        """Use simulated annealing to find a valid coloring with at most k colors."""
        if rng is None:
            rng = random.Random()
        start_t = time.time()
        
        # Initialize with random colors
        coloring = [rng.randint(0, k - 1) for _ in range(N)]
        
        current_violations = count_violations(coloring)
        best_coloring = coloring[:]
        best_violations = current_violations
        
        T = 5.0
        cooling = 0.9995
        steps = 0
        max_steps = 200000
        
        while current_violations > 0 and steps < max_steps and time.time() - start_t < time_budget:
            steps += 1
            idx = rng.randint(0, N - 1)
            old_color = coloring[idx]
            new_color = rng.randint(0, k - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate change in violations incrementally
            delta_violations = 0
            for ap_id in ap_indices[idx]:
                ap = aps[ap_id]
                c0 = coloring[ap[0]]
                c1 = coloring[ap[1]]
                c2 = coloring[ap[2]]
                c3 = coloring[ap[3]]
                
                old_set_size = len({c0, c1, c2, c3})
                
                new_c0 = new_color if ap[0] == idx else c0
                new_c1 = new_color if ap[1] == idx else c1
                new_c2 = new_color if ap[2] == idx else c2
                new_c3 = new_color if ap[3] == idx else c3
                
                new_set_size = len({new_c0, new_c1, new_c2, new_c3})
                
                if old_set_size < 3 and new_set_size >= 3:
                    delta_violations -= 1
                elif old_set_size >= 3 and new_set_size < 3:
                    delta_violations += 1
            
            if delta_violations < 0:
                # Always accept improvements
                coloring[idx] = new_color
                current_violations += delta_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_coloring = coloring[:]
            else:
                # Accept worse solutions with some probability
                if delta_violations == 0:
                    accept = rng.random() < 0.5
                else:
                    if T > 0.001:
                        accept = rng.random() < np.exp(-delta_violations / T)
                    else:
                        accept = False
                
                if accept:
                    coloring[idx] = new_color
                    current_violations += delta_violations
            
            T *= cooling
        
        if best_violations == 0:
            return best_coloring
        if current_violations == 0:
            return coloring
        return None

    def local_search_reduce(colors, max_time=5.0, rng=None):
        """Try to reduce the number of distinct colors via local search."""
        if rng is None:
            rng = random.Random()
        colors = colors[:]
        n_colors = count_colors(colors)
        start_t = time.time()
        
        # Get unique colors
        unique_colors = list(set(colors))
        
        for _ in range(50000):
            if time.time() - start_t > max_time:
                break
            
            improved = False
            # Try to recolor one position to a color that doesn't add new distinct colors
            pos = rng.randint(0, N - 1)
            old_c = colors[pos]
            
            # Try each existing color
            for nc in unique_colors:
                if nc == old_c:
                    continue
                colors[pos] = nc
                if validate(colors):
                    new_count = count_colors(colors)
                    if new_count < n_colors:
                        n_colors = new_count
                        unique_colors = list(set(colors))
                        improved = True
                        break
                colors[pos] = old_c
            
            if not improved:
                # Try a different approach: try to eliminate a color entirely
                for target_color in unique_colors:
                    # Find all positions with this color
                    positions = [i for i in range(N) if colors[i] == target_color]
                    if len(positions) == 0:
                        continue
                    
                    # Try to recolor each position to another color
                    success = True
                    for p in positions:
                        old = colors[p]
                        found = False
                        for nc in unique_colors:
                            if nc == old:
                                continue
                            colors[p] = nc
                            if validate(colors):
                                found = True
                                break
                            colors[p] = old
                        if not found:
                            colors[p] = old
                            success = False
                            break
                    
                    if success:
                        n_colors = count_colors(colors)
                        unique_colors = list(set(colors))
                        improved = True
                        break
            
            if not improved:
                # Try random single-position recoloring with acceptance
                pos = rng.randint(0, N - 1)
                old_c = colors[pos]
                nc = rng.choice(unique_colors)
                if nc != old_c:
                    colors[pos] = nc
                    if validate(colors):
                        new_count = count_colors(colors)
                        if new_count < n_colors:
                            n_colors = new_count
                            unique_colors = list(set(colors))
                    else:
                        colors[pos] = old_c
        
        return colors

    # Strategy: Try to find a valid coloring with k colors for k = 3, 4, 5, ...
    # Use simulated annealing (fast) first, then local search to reduce colors.
    
    best_colors = None
    best_count = N + 1
    
    rng = random.Random(42)
    
    # Try k=3, 4, 5, 6 with increasing time budgets
    time_allocations = [8.0, 8.0, 8.0, 8.0]
    
    for idx, k in enumerate([3, 4, 5, 6]):
        if time.time() > deadline - 15:
            break
        
        print(f"Trying k={k} colors with SA...", file=sys.stderr)
        time_budget = min(time_allocations[idx], deadline - time.time() - 10)
        result = try_k_colors_sa(k, time_budget=time_budget, rng=rng)
        
        if result is not None and validate(result):
            best_colors = result
            best_count = k
            print(f"Found valid coloring with k={k} colors", file=sys.stderr)
            break
        else:
            print(f"k={k} failed", file=sys.stderr)
    
    # If no valid coloring found via SA, use greedy fallback
    if best_colors is None:
        print("SA failed, using greedy fallback...", file=sys.stderr)
        best_colors = [0] * N
        for i in range(N):
            forbidden = get_forbidden(best_colors, i, N)
            found = False
            for c in range(N):
                if c not in forbidden:
                    best_colors[i] = c
                    found = True
                    break
            if not found:
                best_colors[i] = 0
        
        # Validate and fix if needed
        if not validate(best_colors):
            # Try a different initial assignment
            for init in range(1, 10):
                if time.time() > deadline - 5:
                    break
                best_colors = [init % 4] * N
                valid = True
                for i in range(1, N):
                    forbidden = get_forbidden(best_colors, i, 4)
                    found = False
                    for c in range(4):
                        if c not in forbidden:
                            best_colors[i] = c
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if valid and validate(best_colors):
                    break
        
        best_count = count_colors(best_colors)
    
    # Local search to try to reduce colors further
    if best_count > 3 and time.time() < deadline - 10:
        print(f"Local search to reduce from {best_count} colors...", file=sys.stderr)
        reduced = local_search_reduce(best_colors, max_time=min(10.0, deadline - time.time() - 5), rng=rng)
        if count_colors(reduced) < best_count:
            best_colors = reduced
            best_count = count_colors(best_colors)
            print(f"Reduced to {best_count} colors", file=sys.stderr)
    
    # Final relabel
    best_colors, final_count = relabel(best_colors)
    
    # Final validation
    if not validate(best_colors):
        # Emergency fallback
        print("WARNING: final validation failed, using periodic fallback", file=sys.stderr)
        best_colors = [i % 4 for i in range(N)]
        best_colors, final_count = relabel(best_colors)
    
    return best_colors
# EVOLVE-BLOCK-END
