import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 16
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute K4s with their 6 edge indices
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4s.append((
                        pairs.index((a, b)),
                        pairs.index((a, c)),
                        pairs.index((a, d)),
                        pairs.index((b, c)),
                        pairs.index((b, d)),
                        pairs.index((c, d)),
                    ))
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        for e in k4s:
            if len({colors[e[0]], colors[e[1]], colors[e[2]], colors[e[3]], colors[e[4]], colors[e[5]]}) < 5:
                return False
        return True
    
    # Initial coloring: (i + j) % 17
    colors = [(i + j) % 17 for i, j in pairs]
    best_colors = list(colors)
    best_count = count_colors(best_colors)
    
    # Try to improve by merging colors
    def try_merge_into(colors, src, dst):
        """Try to merge all edges with color src into color dst."""
        new_colors = list(colors)
        changed = False
        for i, c in enumerate(new_colors):
            if c == src:
                new_colors[i] = dst
                changed = True
        if not changed:
            return None
        if not is_valid(new_colors):
            return None
        return new_colors
    
    def greedy_reduce(colors):
        """Repeatedly merge the least-used color into another color."""
        colors = list(colors)
        nc = count_colors(colors)
        progress = True
        while progress and time.time() < deadline:
            progress = False
            # Count usage of each color
            from collections import Counter
            cnt = Counter(colors)
            active = sorted(cnt.keys(), key=lambda c: (cnt[c], c))
            for src in active:
                if cnt[src] == 0:
                    continue
                # Try to merge src into each other active color
                for dst in active:
                    if dst == src:
                        continue
                    new_colors = try_merge_into(colors, src, dst)
                    if new_colors is not None:
                        colors = new_colors
                        nc -= 1
                        progress = True
                        break
                if progress:
                    break
        return colors, nc
    
    # Phase 1: Greedy reduction
    colors, nc = greedy_reduce(colors)
    if nc < best_count:
        best_colors = list(colors)
        best_count = nc
    
    # Phase 2: Local search with random perturbations
    rng = random.Random(42)
    
    def local_search(colors, max_iters=50000):
        """Hill climbing: try to recolor edges to reduce color count."""
        colors = list(colors)
        nc = count_colors(colors)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Pick a random edge
            idx = rng.randrange(120)
            old_color = colors[idx]
            
            # Get active colors
            active = list(set(colors))
            if len(active) <= 1:
                break
            
            # Try to change to an existing color (to potentially eliminate old_color)
            candidates = [c for c in active if c != old_color]
            if not candidates:
                continue
            
            new_color = rng.choice(candidates)
            colors[idx] = new_color
            
            # Check validity
            if is_valid(colors):
                new_nc = count_colors(colors)
                if new_nc <= nc:
                    nc = new_nc
                else:
                    # Revert if it increases color count
                    colors[idx] = old_color
            else:
                colors[idx] = old_color
        return colors, nc
    
    # Multiple restarts with perturbations
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from best
        cur = list(best_colors)
        
        # Apply small perturbation
        for _ in range(rng.randint(2, 6)):
            idx = rng.randrange(120)
            cur[idx] = rng.randrange(17)  # Use colors in 0..16
        
        # Validate
        if not is_valid(cur):
            continue
        
        # Local search
        cur, nc = local_search(cur, max_iters=10000)
        
        if nc < best_count:
            best_colors = list(cur)
            best_count = nc
            
            # Try greedy reduction from here
            gcol, gnc = greedy_reduce(best_colors)
            if gnc < best_count:
                best_colors = list(gcol)
                best_count = gnc
    
    # Final greedy pass
    final_colors, final_nc = greedy_reduce(best_colors)
    if final_nc < best_count:
        best_colors = list(final_colors)
        best_count = final_nc
    
    return best_colors
