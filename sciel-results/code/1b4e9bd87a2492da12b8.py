"""Program for R_5(C4) maximization: bitmask-accelerated local search with
structured initialisation, spectral relaxation guidance, and multi-restart strategy."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    def edge_idx(i, j, n):
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    def build_bitmasks(n, colors):
        """Build bitmask adjacency for each color."""
        masks = [0] * (5 * n)  # masks[c*n + v] = bitmask of neighbors of v in color c
        for idx in range(len(colors)):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            masks[c * n + i] |= (1 << j)
            masks[c * n + j] |= (1 << i)
        return masks

    def count_c4_fast(n, colors):
        """Count total C4 violations using bitmask operations."""
        masks = build_bitmasks(n, colors)
        total = 0
        for c in range(5):
            base = c * n
            for u in range(n):
                mu = masks[base + u]
                if mu == 0:
                    continue
                for v in range(u + 1, n):
                    mv = masks[base + v]
                    if mv == 0:
                        continue
                    common = mu & mv
                    # Count bits in common
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt - 1
        return total

    def count_c4_incremental(n, colors, eidx):
        """Count C4 violations, but only check pairs affected by edge eidx.
        Returns total violation count."""
        # Full count is simpler and fast enough with bitmasks for n<=28
        return count_c4_fast(n, colors)

    def local_search(n, colors, time_limit, max_stagnation=5000):
        """Local search: minimize C4 count. Returns (colors, c4_count)."""
        m = n * (n - 1) // 2
        current = count_c4_fast(n, colors)
        if current == 0:
            return colors, 0

        t_start = time.time()
        stagnant = 0
        best_color = colors[:]
        best_c4 = current

        # Precompute edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))

        # Build masks
        masks = build_bitmasks(n, colors)

        while time.time() < t_start + time_limit:
            if current == 0:
                return colors, 0

            # Pick random edge
            eidx = random.randint(0, m - 1)
            old_c = colors[eidx]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue

            # Apply change
            i, j = idx_to_edge(eidx, n)
            colors[eidx] = new_c

            # Update masks
            masks[old_c * n + i] &= ~(1 << j)
            masks[old_c * n + j] &= ~(1 << i)
            masks[new_c * n + i] |= (1 << j)
            masks[new_c * n + j] |= (1 << i)

            # Count C4 (full recount with bitmasks is fast)
            new_val = count_c4_fast(n, colors)

            if new_val > current:
                # Revert
                colors[eidx] = old_c
                masks[old_c * n + i] |= (1 << j)
                masks[old_c * n + j] |= (1 << i)
                masks[new_c * n + i] &= ~(1 << j)
                masks[new_c * n + j] &= ~(1 << i)
                stagnant += 1
            else:
                current = new_val
                stagnant = 0
                if current < best_c4:
                    best_c4 = current
                    best_color = colors[:]
                    # Reset stagnation tracking for restarts
                    if stagnant > max_stagnation:
                        # Restart from best
                        colors = best_color[:]
                        current = best_c4
                        masks = build_bitmasks(n, colors)
                        stagnant = 0

            if stagnant > max_stagnation:
                # Restart from best found
                colors = best_color[:]
                current = best_c4
                masks = build_bitmasks(n, colors)
                stagnant = 0

        return colors, current

    def init_structured(n, mode=0):
        """Structured initialisation to give local search a good starting point."""
        m = n * (n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, n)
            if mode == 0:
                # Cyclic difference mod 5
                d = (j - i) % n
                colors[idx] = d % 5
            elif mode == 1:
                # Sum-based
                colors[idx] = (i + j) % 5
            elif mode == 2:
                # Product-based
                colors[idx] = (i * j) % 5
            elif mode == 3:
                # Difference mod 5 with offset
                d = (j - i) % 5
                colors[idx] = (d + i) % 5
            elif mode == 4:
                # Random
                colors[idx] = random.randint(0, 4)
            elif mode == 5:
                # Spectral-inspired: use eigenvector-like structure
                # Approximate by using a hash of position
                colors[idx] = ((i * 7 + j * 13) % 5)
            elif mode == 6:
                # Two-level: coarse by floor(i/5), fine by j%5
                coarse = (i // 5) % 5
                fine = j % 5
                colors[idx] = (coarse + fine) % 5
            else:
                colors[idx] = random.randint(0, 4)
        return colors

    def spectral_relaxation_guide(n, time_budget):
        """
        Use a spectral relaxation to guide initial color assignment.
        Model: minimize sum over colors of trace(A_c^2) = 2 * (# of P3 in color c).
        This is a proxy for C4 potential.
        
        We use a simple iterative approach:
        1. Start with random assignment
        2. For each edge, try all 5 colors and pick the one that minimizes
           the local contribution to the objective
        3. This is essentially a greedy spectral-like heuristic
        """
        m = n * (n - 1) // 2
        colors = [random.randint(0, 4) for _ in range(m)]
        
        # Build adjacency per color
        adj = [set() for _ in range(5)]
        for idx in range(m):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            adj[c].add((i, j))
            adj[c].add((j, i))
        
        # Greedy improvement: for each edge, try reassigning to minimize P3 count
        # P3 count in color c = sum_v deg_c(v) * (deg_c(v)-1) / 2 ... actually
        # trace(A_c^2) = sum_v deg_c(v)^2
        # So we want to minimize sum_c sum_v deg_c(v)^2
        
        # Compute current degrees
        degs = [[0] * n for _ in range(5)]
        for idx in range(m):
            i, j = idx_to_edge(idx, n)
            c = colors[idx]
            degs[c][i] += 1
            degs[c][j] += 1
        
        obj = sum(sum(d * d for d in degs[c]) for c in range(5))
        
        t_start = time.time()
        for iteration in range(5):
            if time.time() > t_start + time_budget:
                break
            improved = False
            for idx in range(m):
                if time.time() > t_start + time_budget:
                    break
                i, j = idx_to_edge(idx, n)
                old_c = colors[idx]
                # Try each color
                best_c = old_c
                best_obj = obj
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    # Remove from old_c, add to new_c
                    degs[old_c][i] -= 1
                    degs[old_c][j] -= 1
                    degs[new_c][i] += 1
                    degs[new_c][j] += 1
                    new_obj = obj
                    # Update objective: change in sum of squares
                    # For old_c: decrease by 2*degs[old_c][i] + 2*degs[old_c][j] - 2 (since (d-1)^2 - d^2 = -2d+1)
                    # Actually: (d-1)^2 - d^2 = -2d + 1
                    # For new_c: (d+1)^2 - d^2 = 2d + 1
                    # But we need to be careful with the order
                    # Let's just recompute the relevant terms
                    # Simpler: just compute the delta
                    # delta_old = (d_i-1)^2 + (d_j-1)^2 - d_i^2 - d_j^2 = -2*d_i + 1 - 2*d_j + 1 = -2*(d_i+d_j) + 2
                    # But d_i, d_j are AFTER removal
                    # Let d_i_old = degs[old_c][i] + 1 (before removal)
                    # This is getting complicated. Let's just recompute.
                    pass
                
                # Simpler approach: just compute objective from scratch for each candidate
                # But that's O(n) per candidate. For n<=28, that's fine.
                # Actually, let's do it properly:
                best_c = old_c
                best_obj = obj
                for new_c in range(5):
                    if new_c == old_c:
                        continue
                    # Compute delta
                    di_old = degs[old_c][i]
                    dj_old = degs[old_c][j]
                    di_new = degs[new_c][i]
                    dj_new = degs[new_c][j]
                    # After moving:
                    # old_c loses: di_old -> di_old-1, dj_old -> dj_old-1
                    # new_c gains: di_new -> di_new+1, dj_new -> dj_new+1
                    delta = ((di_old - 1) ** 2 + (dj_old - 1) ** 2 - di_old ** 2 - dj_old ** 2) + \
                            ((di_new + 1) ** 2 + (dj_new + 1) ** 2 - di_new ** 2 - dj_new ** 2)
                    if obj + delta < best_obj:
                        best_obj = obj + delta
                        best_c = new_c
                
                if best_c != old_c:
                    # Apply change
                    degs[old_c][i] -= 1
                    degs[old_c][j] -= 1
                    degs[best_c][i] += 1
                    degs[best_c][j] += 1
                    # Update adj
                    adj[old_c].discard((i, j))
                    adj[old_c].discard((j, i))
                    adj[best_c].add((i, j))
                    adj[best_c].add((j, i))
                    colors[idx] = best_c
                    obj = best_obj
                    improved = True
            
            if not improved:
                break
        
        return colors

    # Main strategy
    best_n = 1
    best_colors = []
    
    # Try n=28 first (would be new)
    for target_n in [28, 27, 26, 25]:
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Multiple strategies for this n
        strategies = []
        
        # Strategy 1: Spectral relaxation guided
        if remaining > 10:
            strategies.append(('spectral', 10))
        
        # Strategy 2: Structured init + local search
        for mode in range(7):
            strategies.append((f'structured_{mode}', 8))
        
        # Strategy 3: Random init + local search
        strategies.append(('random', 8))
        
        # Try each strategy
        for strat_name, time_budget in strategies:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < time_budget + 2:
                continue
            
            if strat_name == 'spectral':
                colors = spectral_relaxation_guide(target_n, min(time_budget, remaining - 2))
            elif strat_name.startswith('structured_'):
                mode = int(strat_name.split('_')[1])
                colors = init_structured(target_n, mode)
            else:
                colors = init_structured(target_n, 4)  # random
            
            # Run local search
            time_for_ls = min(remaining - 2, 15)
            if time_for_ls < 2:
                continue
            colors, c4_count = local_search(target_n, colors, time_for_ls)
            
            if c4_count == 0:
                return {'n': target_n, 'colors': colors}
        
        # If we got here, no valid coloring found for this n
        # Continue to next n
    
    # Fallback: known small constructions
    for fallback_n in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        m = fallback_n * (fallback_n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            colors[idx] = (j - i) % 5
        if count_c4_fast(fallback_n, colors) == 0:
            return {'n': fallback_n, 'colors': colors}
    
    return {'n': 1, 'colors': []}
