import random
import time
import itertools

def construct():
    deadline = time.time() + 115
    N = 16
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    E = len(edges)
    edge_idx = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute all K4s and their edge lists
    k4s = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        k4_edges = [edge_idx[(a,b)], edge_idx[(a,c)], edge_idx[(a,d)],
                    edge_idx[(b,c)], edge_idx[(b,d)], edge_idx[(c,d)]]
        k4s.append(k4_edges)
    
    K = len(k4s)
    
    # For each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(E)]
    for k4_idx, k4_edges in enumerate(k4s):
        for e in k4_edges:
            edge_to_k4s[e].append(k4_idx)
    
    def compress(colors):
        mapping = {}
        next_id = 0
        new_colors = []
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def is_valid(colors):
        for k4_idx in range(K):
            k4e = k4s[k4_idx]
            distinct = len(set(colors[e] for e in k4e))
            if distinct < 5:
                return False
        return True

    # Try multiple initialization schemes and pick the one with fewest colors
    init_schemes = [
        lambda i, j: (i * j) % 17,
        lambda i, j: (i + j) % 17,
        lambda i, j: (i * j) % 19,
        lambda i, j: (i * (j + 1)) % 17,
        lambda i, j: (i + j + i*j) % 17,
    ]
    
    best_init = None
    best_init_palette = 100
    
    for scheme in init_schemes:
        colors = [scheme(i, j) for (i, j) in edges]
        colors, palette_size = compress(colors)
        if is_valid(colors):
            if palette_size < best_init_palette:
                best_init_palette = palette_size
                best_init = colors[:]
    
    # Fallback to additive if none valid
    if best_init is None:
        colors = [(i + j) % 17 for (i, j) in edges]
        colors, palette_size = compress(colors)
        best_init = colors
        best_init_palette = palette_size
    
    colors = best_init[:]
    palette_size = best_init_palette
    
    # Track color usage counts
    color_counts = [0] * palette_size
    for c in colors:
        color_counts[c] += 1
    
    best_colors = colors[:]
    best_palette = palette_size
    
    rng = random.Random(42)
    
    def k4_is_valid(k4_idx):
        k4e = k4s[k4_idx]
        return len(set(colors[e] for e in k4e)) >= 5

    # Local search: try to eliminate colors
    for iteration in range(200000):
        if time.time() > deadline - 5:
            break
        
        active_colors = [c for c in range(palette_size) if color_counts[c] > 0]
        if not active_colors:
            break
        
        target_color = rng.choice(active_colors)
        edges_with_target = [e for e in range(E) if colors[e] == target_color]
        
        if not edges_with_target:
            continue
        
        rng.shuffle(edges_with_target)
        
        success = True
        changes = []
        
        for e in edges_with_target:
            old_c = colors[e]
            candidates = [c for c in active_colors if c != old_c]
            rng.shuffle(candidates)
            
            found = False
            for nc in candidates:
                colors[e] = nc
                
                valid = True
                for k4_idx in edge_to_k4s[e]:
                    if not k4_is_valid(k4_idx):
                        valid = False
                        break
                        
                if valid:
                    color_counts[old_c] -= 1
                    color_counts[nc] += 1
                    changes.append((e, old_c, nc))
                    found = True
                    break
                else:
                    colors[e] = old_c
            
            if not found:
                for (ce, oc, ncv) in changes:
                    colors[ce] = oc
                    color_counts[oc] += 1
                    color_counts[ncv] -= 1
                success = False
                break
        
        if success:
            current_palette = sum(1 for c in range(palette_size) if color_counts[c] > 0)
            if current_palette < best_palette:
                best_palette = current_palette
                best_colors = colors[:]
                
            if best_palette <= 13:
                break
        else:
            if rng.random() < 0.05:
                e = rng.randrange(E)
                old_c = colors[e]
                active_now = [c for c in range(palette_size) if color_counts[c] > 0]
                if active_now:
                    nc = rng.choice(active_now)
                    if nc != old_c:
                        colors[e] = nc
                        color_counts[old_c] -= 1
                        color_counts[nc] += 1
    
    return best_colors
