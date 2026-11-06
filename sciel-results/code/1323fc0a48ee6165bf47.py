import random
import time
import itertools

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
        
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)
    
    # Helper to check validity and count palette
    def get_metrics(colors):
        badness = 0
        palette = set(colors)
        for k4e in k4s:
            distinct = len(set(colors[e] for e in k4e))
            if distinct < 5:
                badness += (5 - distinct)
        return badness, len(palette)

    def is_valid(colors):
        for k4e in k4s:
            if len(set(colors[e] for e in k4e)) < 5:
                return False
        return True

    # Phase 1: Find a valid coloring with minimum palette
    # The seed (i+j)%17 has palette 17 and is valid.
    # We want to reduce it.
    # Try structured inits and local search.
    
    best_colors = None
    best_palette = 17 # Seed palette
    
    # Try a few structured inits
    inits = []
    # (i+j) % 17
    inits.append([(i + j) % 17 for (i, j) in edges])
    # (i*j) % 17 (Note: 0 is a color here, but i,j >= 0. If i=0, color 0. 
    # K4 {0,1,2,3}: edges (0,1)->0, (0,2)->0... wait.
    # (0,1)=0, (0,2)=0, (0,3)=0, (1,2)=2, (1,3)=3, (2,3)=6. Colors: {0,2,3,6}. Size 4. Invalid.
    # So multiplicative is bad for 0. We need non-zero i,j or offset.
    # (i+1)*(j+1) % 17?
    inits.append([((i+1)*(j+1)) % 17 for (i, j) in edges])
    
    # Random init with small palette?
    rng = random.Random(42)
    for _ in range(5):
        p = rng.randint(13, 16)
        inits.append([rng.randint(0, p-1) for _ in range(E)])
        
    for init in inits:
        bad, pal = get_metrics(init)
        if bad == 0:
            if pal < best_palette:
                best_palette = pal
                best_colors = init[:]
            if best_palette == 13:
                return best_colors
    
    # If we found a valid one with pal < 17, we are in good shape.
    # If not, we start from the seed (17) and try to reduce.
    if best_colors is None:
        best_colors = [(i + j) % 17 for (i, j) in edges]
        best_palette = 17

    # Phase 2: Local search to reduce palette from best_colors
    # We will use the strategy from Inspiration 3 (efficient local search)
    # but applied to the best valid state found.
    
    # We might have multiple valid states. Let's pick the one with smallest palette.
    # If we only have 17, we try to reduce it.
    
    # Let's run a local search on the best valid state found so far.
    # If we didn't find a valid state < 17, we start from 17.
    # But wait, if we start from 17 and it's valid, we can try to reduce.
    # If we start from a random invalid one, we need to fix it first.
    # The inits above are either valid (if bad==0) or we ignore them.
    # So best_colors is always valid.
    
    current = best_colors[:]
    current_palette = set(current)
    
    # Precompute for speed
    k4s_edges = k4s
    edge_to_k4s_local = edge_to_k4s
    
    # We will use a simple local search:
    # 1. Pick a color class to eliminate.
    # 2. Try to remap its edges to other colors.
    # This is the "eliminate color" strategy from Inspiration 2, which is effective for palette reduction.
    
    rng_search = random.Random(123)
    
    for step in range(500000):
        if time.time() > deadline - 5:
            break
            
        if len(current_palette) <= best_palette:
            if len(current_palette) == best_palette:
                # Maybe we can find a better arrangement? No, palette size is the metric.
                # If we are at best_palette, we can stop if we can't go lower.
                # But maybe we can go lower.
                pass
            else:
                # We improved? This shouldn't happen if we update best_palette immediately.
                pass
        else:
            # Reset to best if we drifted? 
            # Actually, we are doing constructive search from a valid state.
            # If we break validity, we should revert.
            # The strategy below maintains validity.
            pass
        
        # If we have reached 13, stop.
        if best_palette == 13:
            break
            
        # Pick a color to eliminate
        # Prefer colors with fewer edges? Or random?
        # Let's try random first, maybe biased towards rare colors.
        colors_list = list(current_palette)
        # Sort by frequency?
        # freq = {}
        # for c in current: freq[c] = freq.get(c, 0) + 1
        # colors_list.sort(key=lambda c: freq[c])
        # target = colors_list[0]
        
        target = rng_search.choice(colors_list)
        
        # Find edges with this color
        edges_with_target = [e for e, c in enumerate(current) if c == target]
        
        if not edges_with_target:
            continue
            
        # Try to remap each edge
        new_current = current[:]
        success = True
        for e in edges_with_target:
            # Try other colors
            other_colors = [c for c in current_palette if c != target]
            rng_search.shuffle(other_colors)
            found = False
            for new_c in other_colors:
                new_current[e] = new_c
                # Check validity for K4s containing e
                valid_so_far = True
                for k4_idx in edge_to_k4s_local[e]:
                    k4e = k4s_edges[k4_idx]
                    distinct = len(set(new_current[ee] for ee in k4e))
                    if distinct < 5:
                        valid_so_far = False
                        break
                if valid_so_far:
                    found = True
                    break
            if not found:
                new_current[e] = current[e]
                success = False
                break
        
        if success:
            # Full validation just in case
            if is_valid(new_current):
                new_pal = set(new_current)
                if len(new_pal) < len(current_palette):
                    current = new_current
                    current_palette = new_pal
                    if len(current_palette) < best_palette:
                        best_palette = len(current_palette)
                        best_colors = current[:]
                        if best_palette == 13:
                            break
                else:
                    # Palette didn't shrink, maybe we can still use this state?
                    # It's an isomorphism in palette size.
                    # Let's stick to shrinking only to avoid wandering.
                    pass
            else:
                # Should not happen if local check was correct
                pass

    return best_colors
