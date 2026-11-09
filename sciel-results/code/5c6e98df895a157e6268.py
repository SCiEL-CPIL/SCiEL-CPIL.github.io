"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_K4 = 1820

# Pre-compute edge indices
EDGE_IDX = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        EDGE_IDX[(i, j)] = idx
        idx += 1

# Pre-compute K4s and their edge indices
K4_EDGES = []
for quad in combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        EDGE_IDX[(a, b)], EDGE_IDX[(a, c)], EDGE_IDX[(a, d)],
        EDGE_IDX[(b, c)], EDGE_IDX[(b, d)], EDGE_IDX[(c, d)]
    ]
    K4_EDGES.append(tuple(edges))

# Pre-compute for each edge, which K4s it belongs to
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k_idx, edges in enumerate(K4_EDGES):
    for e in edges:
        EDGE_TO_K4[e].append(k_idx)

def _count_distinct_6(c0, c1, c2, c3, c4, c5):
    return len({c0, c1, c2, c3, c4, c5})

def _compute_violations(colors):
    """Compute violation status and count."""
    k4_status = [False] * NUM_K4
    total_v = 0
    for k_idx, edges in enumerate(K4_EDGES):
        e0, e1, e2, e3, e4, e5 = edges
        if _count_distinct_6(colors[e0], colors[e1], colors[e2], colors[e3], colors[e4], colors[e5]) < 5:
            k4_status[k_idx] = True
            total_v += 1
    return k4_status, total_v

def _construct():
    deadline = time.time() + 110
    random.seed(42)
    
    # Try multiple initial constructions
    best = None
    best_violations = float('inf')
    best_k4_status = None
    
    # Try (i+j) % k for various k
    for k in range(13, 25):
        if time.time() > deadline - 5:
            break
        cand = [(i + j) % k for i in range(N) for j in range(i + 1, N)]
        ks, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_k4_status = ks
        if tv == 0:
            break
    
    # Also try random starts
    for _ in range(200):
        if time.time() > deadline - 3:
            break
        max_c = random.randint(13, 20)
        cand = [random.randint(0, max_c - 1) for _ in range(NUM_EDGES)]
        ks, tv = _compute_violations(cand)
        if tv < best_violations:
            best_violations = tv
            best = cand
            best_k4_status = ks
        if tv == 0:
            break
    
    if best is None:
        return [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    current = list(best)
    current_violations = best_violations
    current_k4_status = list(best_k4_status)
    
    # Track color frequencies
    MAX_COLOR = 25
    color_freq = [0] * MAX_COLOR
    for c in current:
        if c < MAX_COLOR:
            color_freq[c] += 1
    current_colors = sum(1 for f in color_freq if f > 0)
    
    # Local search
    for step in range(100000):
        if time.time() > deadline - 2:
            break
        
        if current_violations == 0:
            # Phase 1: Reduce colors
            # Find edges whose color appears only once
            single_use_colors = [c for c in range(MAX_COLOR) if color_freq[c] == 1]
            
            if not single_use_colors:
                # No single-use colors, try to merge colors
                # Find a color with minimum frequency > 1
                min_freq = min(f for f in color_freq if f > 0)
                if min_freq == 1:
                    continue
                # Try changing one edge of a rare color to another existing color
                rare_color = [c for c in range(MAX_COLOR) if color_freq[c] == min_freq][0]
                # Find an edge with this color
                edges_with_rare = [e for e in range(NUM_EDGES) if current[e] == rare_color]
                if not edges_with_rare:
                    continue
                e_idx = random.choice(edges_with_rare)
                old_c = rare_color
                # Choose target color: one that's already used
                used_colors = [c for c in range(MAX_COLOR) if color_freq[c] > 0 and c != old_c]
                if not used_colors:
                    continue
                new_c = random.choice(used_colors)
            else:
                # Change a single-use color edge to an existing color
                old_c = random.choice(single_use_colors)
                # Find an edge with this color
                edges_with_old = [e for e in range(NUM_EDGES) if current[e] == old_c]
                if not edges_with_old:
                    continue
                e_idx = random.choice(edges_with_old)
                used_colors = [c for c in range(MAX_COLOR) if color_freq[c] > 0 and c != old_c]
                if not used_colors:
                    continue
                new_c = random.choice(used_colors)
            
            # Check if move is valid (doesn't create violations)
            affected_k4s = EDGE_TO_K4[e_idx]
            valid = True
            new_violations = current_violations
            
            for k_idx in affected_k4s:
                edges = K4_EDGES[k_idx]
                e0, e1, e2, e3, e4, e5 = edges
                c0 = current[e0] if e0 != e_idx else new_c
                c1 = current[e1] if e1 != e_idx else new_c
                c2 = current[e2] if e2 != e_idx else new_c
                c3 = current[e3] if e3 != e_idx else new_c
                c4 = current[e4] if e4 != e_idx else new_c
                c5 = current[e5] if e5 != e_idx else new_c
                
                new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                new_status = new_distinct < 5
                
                if current_k4_status[k_idx] and not new_status:
                    new_violations -= 1
                elif not current_k4_status[k_idx] and new_status:
                    new_violations += 1
                    valid = False
                    break
            
            if not valid or new_violations > 0:
                continue
            
            # Check color count
            new_color_count = current_colors
            if color_freq[old_c] == 1:
                new_color_count -= 1
            if color_freq[new_c] == 0:
                new_color_count += 1
            
            if new_color_count < current_colors:
                # Apply move
                current[e_idx] = new_c
                color_freq[old_c] -= 1
                color_freq[new_c] += 1
                current_colors = new_color_count
                current_violations = new_violations
                
                # Update K4 statuses
                for k_idx in affected_k4s:
                    edges = K4_EDGES[k_idx]
                    e0, e1, e2, e3, e4, e5 = edges
                    current_k4_status[k_idx] = _count_distinct_6(
                        current[e0], current[e1], current[e2], current[e3], current[e4], current[e5]
                    ) < 5
        else:
            # Phase 2: Reduce violations
            # Pick a violated K4 and change one of its edges
            violated_k4s = [k for k in range(NUM_K4) if current_k4_status[k]]
            if not violated_k4s:
                current_violations = 0
                continue
                
            k_idx = random.choice(violated_k4s)
            edges = K4_EDGES[k_idx]
            e_idx = random.choice(edges)
            old_c = current[e_idx]
            
            # Try different new colors
            improved = False
            for _ in range(20):
                new_c = random.randint(0, 19)
                if new_c == old_c:
                    continue
                
                valid = True
                new_violations = current_violations
                
                for k2 in EDGE_TO_K4[e_idx]:
                    e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                    c0 = current[e0] if e0 != e_idx else new_c
                    c1 = current[e1] if e1 != e_idx else new_c
                    c2 = current[e2] if e2 != e_idx else new_c
                    c3 = current[e3] if e3 != e_idx else new_c
                    c4 = current[e4] if e4 != e_idx else new_c
                    c5 = current[e5] if e5 != e_idx else new_c
                    
                    new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                    new_status = new_distinct < 5
                    
                    if current_k4_status[k2] and not new_status:
                        new_violations -= 1
                    elif not current_k4_status[k2] and new_status:
                        new_violations += 1
                        valid = False
                        break
                
                if valid and new_violations < current_violations:
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    color_freq[new_c] += 1
                    current_violations = new_violations
                    
                    for k2 in EDGE_TO_K4[e_idx]:
                        e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                        current_k4_status[k2] = _count_distinct_6(
                            current[e0], current[e1], current[e2], current[e3], current[e4], current[e5]
                        ) < 5
                    improved = True
                    break
            
            if not improved:
                # Try a random edge instead
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_c = current[e_idx]
                new_c = random.randint(0, 19)
                if new_c == old_c:
                    continue
                
                valid = True
                new_violations = current_violations
                
                for k2 in EDGE_TO_K4[e_idx]:
                    e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                    c0 = current[e0] if e0 != e_idx else new_c
                    c1 = current[e1] if e1 != e_idx else new_c
                    c2 = current[e2] if e2 != e_idx else new_c
                    c3 = current[e3] if e3 != e_idx else new_c
                    c4 = current[e4] if e4 != e_idx else new_c
                    c5 = current[e5] if e5 != e_idx else new_c
                    
                    new_distinct = _count_distinct_6(c0, c1, c2, c3, c4, c5)
                    new_status = new_distinct < 5
                    
                    if current_k4_status[k2] and not new_status:
                        new_violations -= 1
                    elif not current_k4_status[k2] and new_status:
                        new_violations += 1
                        valid = False
                        break
                
                if valid and new_violations < current_violations:
                    current[e_idx] = new_c
                    color_freq[old_c] -= 1
                    color_freq[new_c] += 1
                    current_violations = new_violations
                    
                    for k2 in EDGE_TO_K4[e_idx]:
                        e0, e1, e2, e3, e4, e5 = K4_EDGES[k2]
                        current_k4_status[k2] = _count_distinct_6(
                            current[e0], current[e1], current[e2], current[e3], current[e4], current[e5]
                        ) < 5
    
    return current

def construct():
    return _construct()
# EVOLVE-BLOCK-END
