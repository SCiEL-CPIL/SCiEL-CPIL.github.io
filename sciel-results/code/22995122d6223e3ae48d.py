"""
Optimized hierarchical Gallai coloring + local search refinement.

Key improvements:
1. Compute optimal split points via DP to maximize color count in structural phase
2. Faster local search with better move selection
3. Tighter time management
"""
import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles
_CYCLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                _CYCLES.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                _CYCLES.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))

NUM_CYCLES = len(_CYCLES)

# For each edge, precompute which cycles contain it
_EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(_CYCLES):
    for e in cycle:
        _EDGE_TO_CYCLES[e].append(ci)

def _check_edge_change(colors, edge_idx, new_color):
    """Check if changing colors[edge_idx] to new_color maintains validity."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return True
    for ci in _EDGE_TO_CYCLES[edge_idx]:
        e0, e1, e2, e3 = _CYCLES[ci]
        c0 = colors[e0] if e0 != edge_idx else new_color
        c1 = colors[e1] if e1 != edge_idx else new_color
        c2 = colors[e2] if e2 != edge_idx else new_color
        c3 = colors[e3] if e3 != edge_idx else new_color
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _compute_optimal_splits():
    """DP to find split points that maximize color count."""
    # f(n) = max colors achievable for n vertices
    # f(1)=0, f(2)=1, f(3)=3
    # f(n) = 1 + max over k of f(k) + f(n-k) for n>=4
    f = [0] * (N + 1)
    split = [0] * (N + 1)
    f[1] = 0
    f[2] = 1
    f[3] = 3
    for n in range(4, N + 1):
        best_val = -1
        best_k = 1
        for k in range(1, n):
            val = 1 + f[k] + f[n - k]
            if val > best_val:
                best_val = val
                best_k = k
        f[n] = best_val
        split[n] = best_k
    return f, split

def _build_hierarchical(vertices, start_color, splits, color_map):
    n = len(vertices)
    if n <= 1:
        return start_color
    if n == 2:
        u, v = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        return start_color + 1
    if n == 3:
        u, v, w = vertices
        color_map[(min(u,v), max(u,v))] = start_color
        color_map[(min(v,w), max(v,w))] = start_color + 1
        color_map[(min(u,w), max(u,w))] = start_color + 2
        return start_color + 3
    
    k = splits.get(n, n // 2)
    v1 = vertices[:k]
    v2 = vertices[k:]
    
    cross_color = start_color
    for u in v1:
        for v in v2:
            color_map[(min(u,v), max(u,v))] = cross_color
    
    next_color = start_color + 1
    next_color = _build_hierarchical(v1, next_color, splits, color_map)
    next_color = _build_hierarchical(v2, next_color, splits, color_map)
    return next_color

def _build_best_structural():
    """Build the optimal hierarchical coloring using DP-computed splits."""
    f, split = _compute_optimal_splits()
    
    # Also try a few alternative split strategies for diversity
    candidates = []
    
    # Optimal DP splits
    splits1 = {n: split[n] for n in range(4, N + 1)}
    candidates.append(splits1)
    
    # Try slight variations around optimal
    for delta in [-1, 1]:
        splits_var = {}
        for n in range(4, N + 1):
            k = split[n] + delta
            if 1 <= k < n:
                splits_var[n] = k
            else:
                splits_var[n] = split[n]
        candidates.append(splits_var)
    
    # Star-like: split off 1
    splits_star = {n: 1 for n in range(4, N + 1)}
    candidates.append(splits_star)
    
    # Balanced
    splits_bal = {n: n // 2 for n in range(4, N + 1)}
    candidates.append(splits_bal)
    
    best_colors = None
    best_score = 0
    
    for splits in candidates:
        color_map = {}
        next_color = _build_hierarchical(list(range(N)), 0, splits, color_map)
        
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = color_map[(i, j)]
        
        if _is_valid(colors):
            score = _count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    return best_colors, best_score

def _local_search(colors, deadline):
    """Fast local search to increase distinct color count."""
    current_score = _count_colors(colors)
    used_colors = set(colors)
    no_improve = 0
    max_no_improve = 500000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Try random edge changes, biased toward unused colors
        for _ in range(100):
            if time.time() > deadline:
                return colors, current_score
            
            edge_idx = random.randint(0, NUM_EDGES - 1)
            old_color = colors[edge_idx]
            
            if random.random() < 0.8:
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLOR)
            else:
                new_color = random.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
            
            if _check_edge_change(colors, edge_idx, new_color):
                colors[edge_idx] = new_color
                new_score = _count_colors(colors)
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                    improved = True
                    used_colors = set(colors)
                else:
                    colors[edge_idx] = old_color
            # else: invalid, skip
        
        if not improved:
            no_improve += 100
    
    return colors, current_score

def construct():
    deadline = time.time() + 80.0
    
    random.seed(42)
    
    # Step 1: Best structural coloring
    best_colors, best_score = _build_best_structural()
    
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    # Step 2: Local search refinement
    colors, score = _local_search(best_colors[:], deadline)
    
    if score > best_score:
        best_score = score
        best_colors = colors
    
    return best_colors
