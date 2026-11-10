"""Improved program for edge-colouring K16 with K4 five-colour constraint."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    """Return linear index for edge (i,j) in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all K4 edge sets
_K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    _K4_EDGES.append([
        _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
        _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
    ])
NUM_K4 = len(_K4_EDGES)

# Precompute for each edge, which K4s it belongs to
_EDGE_K4S = [[] for _ in range(NUM_EDGES)]
for k4_idx, edges in enumerate(_K4_EDGES):
    for e in edges:
        _EDGE_K4S[e].append(k4_idx)

def _count_violations(coloring):
    """Count number of K4s with fewer than 5 distinct colours."""
    violations = 0
    for edges in _K4_EDGES:
        c0, c1, c2, c3, c4, c5 = (coloring[e] for e in edges)
        # Count distinct colours
        distinct = 1
        if c1 != c0: distinct += 1
        if c2 != c0 and c2 != c1: distinct += 1
        if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
        if distinct < 5:
            violations += 1
    return violations

def _is_valid(coloring):
    return _count_violations(coloring) == 0

def _count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial coloring: (i+j) mod 17
    coloring = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # Verify initial validity
    if not _is_valid(coloring):
        # Try other initial colorings
        for mod in [13, 14, 15, 16, 18, 19, 20]:
            coloring = [(i + j) % mod for i in range(N) for j in range(i + 1, N)]
            if _is_valid(coloring):
                break
        # If still invalid, use random
        if not _is_valid(coloring):
            random.seed(123)
            while True:
                coloring = [random.randint(0, 16) for _ in range(NUM_EDGES)]
                if _is_valid(coloring):
                    break
    
    best = coloring[:]
    best_colors = _count_colors(best)
    best_violations = 0
    
    # Phase 1: Greedy color merging - try to merge colors one by one
    # Try to reduce from current count down
    for target_count in range(best_colors - 1, 12, -1):
        if time.time() > deadline:
            break
        
        current = best[:]
        current_colors = best_colors
        
        # Get list of used colors
        used = sorted(set(current))
        
        # Try to merge pairs of colors
        improved = True
        while improved and time.time() < deadline:
            improved = False
            used = sorted(set(current))
            
            for i in range(len(used)):
                for j in range(i + 1, len(used)):
                    if time.time() > deadline:
                        break
                    
                    c1, c2 = used[i], used[j]
                    # Try merging c2 into c1
                    new_coloring = current[:]
                    for idx, c in enumerate(new_coloring):
                        if c == c2:
                            new_coloring[idx] = c1
                    
                    if _is_valid(new_coloring):
                        nc = _count_colors(new_coloring)
                        if nc < current_colors:
                            current = new_coloring
                            current_colors = nc
                            best = current[:]
                            best_colors = nc
                            improved = True
                            break
                if improved:
                    break
        
        if not _is_valid(current):
            current = best[:]
            current_colors = best_colors
    
    # Phase 2: Local search - single edge recoloring to reduce color count
    # Strategy: pick a color that appears least, try to reassign its edges to other colors
    random.seed(42)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        current = best[:]
        current_colors = best_colors
        used = sorted(set(current))
        
        if len(used) <= 13:
            break
        
        # Count occurrences of each color
        color_counts = {}
        for c in current:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Try to eliminate the least common color
        # Sort colors by frequency (ascending)
        sorted_colors = sorted(used, key=lambda c: color_counts[c])
        
        improved = False
        for target_color in sorted_colors[:3]:  # Try top 3 least common
            if time.time() > deadline:
                break
            
            edges_with_target = [i for i, c in enumerate(current) if c == target_color]
            
            # Try reassigning each edge to a different color
            for ei in edges_with_target:
                if time.time() > deadline:
                    break
                
                # Try all other colors
                for new_color in used:
                    if new_color == target_color:
                        continue
                    
                    old_color = current[ei]
                    current[ei] = new_color
                    
                    # Quick check: only need to check K4s containing edge ei
                    valid = True
                    for k4_idx in _EDGE_K4S[ei]:
                        edges = _K4_EDGES[k4_idx]
                        c0, c1, c2, c3, c4, c5 = (current[e] for e in edges)
                        distinct = 1
                        if c1 != c0: distinct += 1
                        if c2 != c0 and c2 != c1: distinct += 1
                        if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                        if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
                        if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
                        if distinct < 5:
                            valid = False
                            break
                    
                    if valid:
                        nc = _count_colors(current)
                        if nc < current_colors:
                            best = current[:]
                            best_colors = nc
                            current_colors = nc
                            improved = True
                            break
                        else:
                            # Revert
                            current[ei] = old_color
                    else:
                        current[ei] = old_color
                
                if improved:
                    break
            
            if improved:
                break
        
        if not improved:
            # Try random perturbations
            if time.time() > deadline:
                break
            
            # Randomly pick an edge and change its color
            ei = random.randint(0, NUM_EDGES - 1)
            old_color = current[ei]
            new_color = random.choice(used)
            
            if new_color != old_color:
                current[ei] = new_color
                # Check if still valid
                valid = True
                for k4_idx in _EDGE_K4S[ei]:
                    edges = _K4_EDGES[k4_idx]
                    c0, c1, c2, c3, c4, c5 = (current[e] for e in edges)
                    distinct = 1
                    if c1 != c0: distinct += 1
                    if c2 != c0 and c2 != c1: distinct += 1
                    if c3 != c0 and c3 != c1 and c3 != c2: distinct += 1
                    if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: distinct += 1
                    if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: distinct += 1
                    if distinct < 5:
                        valid = False
                        break
                
                if valid:
                    nc = _count_colors(current)
                    if nc < best_colors:
                        best = current[:]
                        best_colors = nc
                else:
                    current[ei] = old_color
    
    # Phase 3: Final polish - try to merge remaining colors with more careful search
    if time.time() < deadline - 5:
        current = best[:]
        current_colors = best_colors
        used = sorted(set(current))
        
        # Try all pairs of colors for merging
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                if time.time() > deadline:
                    break
                
                c1, c2 = used[i], used[j]
                new_coloring = current[:]
                for idx, c in enumerate(new_coloring):
                    if c == c2:
                        new_coloring[idx] = c1
                
                if _is_valid(new_coloring):
                    nc = _count_colors(new_coloring)
                    if nc < current_colors:
                        current = new_coloring
                        current_colors = nc
                        best = current[:]
                        best_colors = nc
                        used = sorted(set(current))
    
    return best
# EVOLVE-BLOCK-END
