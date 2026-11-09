"""
Combined solver: hierarchical Gallai partition (structural) + fast local search (refinement).

The structural insight from inspiration 1: a recursive partition of vertices where
cross-edges between parts get a single shared color, and each part is recursively
colored, yields a valid Gallai coloring. The number of colors used equals the number
of internal nodes in the partition tree.

The refinement machinery from the current program: fast incremental validity checking
via precomputed edge-to-cycle mappings, enabling rapid local search to potentially
find colorings with more distinct colors than the pure structural approach.

Strategy:
1. Generate the best hierarchical Gallai coloring (binary splits, recursively).
2. Use fast local search to try to increase the number of distinct colors while
   maintaining validity.
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

# Precompute all 4-cycles as tuples of 4 edge indices
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
    """Check if no 4-cycle has four distinct edge colors."""
    for cycle in _CYCLES:
        e0, e1, e2, e3 = cycle
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _build_hierarchical(vertices, start_color, splits, color_map):
    """
    Build a hierarchical Gallai coloring.
    vertices: list of vertex indices
    start_color: the next available color index
    splits: dict mapping len(vertices) -> split point k
    color_map: dict to store edge colors
    Returns the next available color index.
    """
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
    
    # Cross edges get one shared color
    cross_color = start_color
    for u in v1:
        for v in v2:
            color_map[(min(u,v), max(u,v))] = cross_color
    
    next_color = start_color + 1
    next_color = _build_hierarchical(v1, next_color, splits, color_map)
    next_color = _build_hierarchical(v2, next_color, splits, color_map)
    return next_color

def _try_all_splits():
    """Try all possible binary split configurations and return the best coloring."""
    best_colors = None
    best_score = 0
    
    # Try various split strategies
    split_configs = []
    
    # Strategy 1: Always split into roughly equal halves
    splits1 = {}
    for n in range(4, N+1):
        splits1[n] = n // 2
    split_configs.append(splits1)
    
    # Strategy 2: Split off one vertex at a time (star-like)
    splits2 = {}
    for n in range(4, N+1):
        splits2[n] = 1
    split_configs.append(splits2)
    
    # Strategy 3: Split off two vertices
    splits3 = {}
    for n in range(4, N+1):
        splits3[n] = 2
    split_configs.append(splits3)
    
    # Strategy 4: Mixed - small splits for small n, balanced for large
    splits4 = {}
    for n in range(4, N+1):
        if n <= 5:
            splits4[n] = 2
        elif n <= 8:
            splits4[n] = 3
        else:
            splits4[n] = n // 2
    split_configs.append(splits4)
    
    # Strategy 5: From inspiration 1
    splits5 = {
        4: 3, 5: 3, 6: 3, 7: 3, 8: 4, 9: 3, 10: 4,
        11: 5, 12: 6, 13: 6, 14: 7, 15: 6, 16: 7
    }
    split_configs.append(splits5)
    
    # Strategy 6: Try to maximize number of internal nodes
    # Each internal node with n>=4 children contributes 1 color for cross + recursive
    # For binary splits, number of colors = number of internal nodes
    # To maximize, we want as many internal nodes as possible
    # This means splitting off small groups
    splits6 = {}
    for n in range(4, N+1):
        splits6[n] = min(3, n-1)  # Split off at most 3
    split_configs.append(splits6)
    
    # Strategy 7: Split off 1 for n<=6, then balanced
    splits7 = {}
    for n in range(4, N+1):
        if n <= 6:
            splits7[n] = 1
        else:
            splits7[n] = n // 2
    split_configs.append(splits7)
    
    # Strategy 8: Split off 2 for n<=8, then balanced
    splits8 = {}
    for n in range(4, N+1):
        if n <= 8:
            splits8[n] = 2
        else:
            splits8[n] = n // 2
    split_configs.append(splits8)
    
    for splits in split_configs:
        color_map = {}
        next_color = _build_hierarchical(list(range(N)), 0, splits, color_map)
        
        # Convert to flat array
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[_edge_idx(i, j)] = color_map[(i, j)]
        
        if _is_valid(colors):
            score = _count_colors(colors)
            if score > best_score:
                best_score = score
                best_colors = colors[:]
    
    return best_colors, best_score

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    random.seed(12345)
    
    # Step 1: Generate best hierarchical coloring
    best_colors, best_score = _try_all_splits()
    
    if best_colors is None:
        # Fallback: all zeros
        best_colors = [0] * NUM_EDGES
        best_score = 1
    
    # Step 2: Local search to try to increase color count
    colors = best_colors[:]
    current_score = best_score
    used_colors = set(colors)
    
    no_improve_count = 0
    max_no_improve = 300000
    iterations = 0
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        iterations += 1
        if iterations % 5000 == 0 and time.time() > deadline:
            break
        
        improved = False
        
        # Try random edge changes
        for _ in range(200):
            if time.time() > deadline:
                break
            
            edge_idx = random.randint(0, NUM_EDGES - 1)
            old_color = colors[edge_idx]
            
            # Try a new random color, preferring unused colors
            if random.random() < 0.7:
                # Try an unused color
                unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLOR)
            else:
                new_color = random.randint(0, MAX_COLOR)
            
            if new_color == old_color:
                continue
            
            # Check if this change maintains validity
            if _check_edge_change(colors, edge_idx, new_color):
                colors[edge_idx] = new_color
                new_score = _count_colors(colors)
                if new_score > current_score:
                    current_score = new_score
                    no_improve_count = 0
                    improved = True
                    used_colors = set(colors)
                else:
                    colors[edge_idx] = old_color
            else:
                pass  # Revert (already unchanged)
        
        if not improved:
            no_improve_count += 200
    
    if current_score > best_score:
        best_score = current_score
        best_colors = colors[:]
    
    return best_colors
