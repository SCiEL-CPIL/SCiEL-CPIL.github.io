"""Improved local search for K16 edge-colouring with >=5 colours per K4."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_N = 16
_NUM_EDGES = _N * (_N - 1) // 2  # 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (_N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all K4s and their edge indices
_K4S = list(combinations(range(_N), 4))
_K4_EDGE_IDXS = []
for quad in _K4S:
    eidxs = []
    for a in range(4):
        for b in range(a + 1, 4):
            eidxs.append(_edge_idx(quad[a], quad[b]))
    _K4_EDGE_IDXS.append(eidxs)

_NUM_K4S = len(_K4S)  # 1820

# For each edge, which K4s contain it
_EDGE_TO_K4S = [[] for _ in range(_NUM_EDGES)]
for k4_idx, eidxs in enumerate(_K4_EDGE_IDXS):
    for ei in eidxs:
        _EDGE_TO_K4S[ei].append(k4_idx)

# For each K4, which position each of its 6 edges is at
_K4_EDGE_POS = []
for k4_idx, eidxs in enumerate(_K4_EDGE_IDXS):
    pos_map = {}
    for p, ei in enumerate(eidxs):
        pos_map[ei] = p
    _K4_EDGE_POS.append(pos_map)


def _distinct_count(vals):
    """Count distinct values in a list of 6."""
    v0 = vals[0]
    d = 1
    if vals[1] != v0: d += 1
    if vals[2] != v0 and vals[2] != vals[1]: d += 1
    if vals[3] != v0 and vals[3] != vals[1] and vals[3] != vals[2]: d += 1
    if vals[4] != v0 and vals[4] != vals[1] and vals[4] != vals[2] and vals[4] != vals[3]: d += 1
    if vals[5] != v0 and vals[5] != vals[1] and vals[5] != vals[2] and vals[5] != vals[3] and vals[5] != vals[4]: d += 1
    return d


def _count_bad_k4s(colors):
    bad = 0
    for k4_idx in range(_NUM_K4S):
        eidxs = _K4_EDGE_IDXS[k4_idx]
        vals = [colors[eidxs[p]] for p in range(6)]
        if _distinct_count(vals) < 5:
            bad += 1
    return bad


def _count_colors_used(colors):
    return len(set(colors.tolist()))


def _delta_bad_for_change(colors, edge_idx, new_color):
    """Compute change in bad K4 count if we set colors[edge_idx] = new_color."""
    old_color = colors[edge_idx]
    if old_color == new_color:
        return 0
    
    delta = 0
    for k4_idx in _EDGE_TO_K4S[edge_idx]:
        eidxs = _K4_EDGE_IDXS[k4_idx]
        pos = _K4_EDGE_POS[k4_idx][edge_idx]
        
        # Get current 6 values
        vals = [colors[eidxs[p]] for p in range(6)]
        distinct_before = _distinct_count(vals)
        bad_before = 1 if distinct_before < 5 else 0
        
        # Apply change
        vals[pos] = new_color
        distinct_after = _distinct_count(vals)
        bad_after = 1 if distinct_after < 5 else 0
        
        delta += bad_after - bad_before
    
    return delta


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Start with the known valid 17-coloring
    colors = np.array([(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
    
    best_colors = colors.copy()
    best_score = _count_colors_used(best_colors)  # 17
    best_bad = 0
    
    rng = random.Random(42)
    
    # Try to find valid colorings with fewer colors
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 30:
            break
        
        # Multiple restarts
        for restart in range(50):
            if time.time() > deadline - 15:
                break
            
            # Initialize: try structured initialization
            if restart == 0:
                # Use (i+j) mod target_colors
                c = np.array([(i + j) % target_colors for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
            elif restart == 1:
                # Use (i*3+j) mod target_colors
                c = np.array([(i * 3 + j) % target_colors for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
            elif restart == 2:
                # Use (i*5+j) mod target_colors
                c = np.array([(i * 5 + j) % target_colors for i in range(_N) for j in range(i + 1, _N)], dtype=np.int32)
            else:
                # Random initialization
                c = np.array([rng.randint(0, target_colors - 1) for _ in range(_NUM_EDGES)], dtype=np.int32)
            
            bad = _count_bad_k4s(c)
            
            if bad == 0:
                actual = _count_colors_used(c)
                if actual < best_score:
                    best_score = actual
                    best_colors = c.copy()
                    best_bad = 0
                break  # Found valid, try next target_colors
            
            # Local search with simulated annealing
            # Track temperature for SA
            T = 2.0
            T_min = 0.01
            cooling = 0.9995
            
            max_iters = 200000
            stuck_counter = 0
            best_local_bad = bad
            best_local_c = c.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Pick a random edge
                edge_idx = rng.randint(0, _NUM_EDGES - 1)
                old_color = c[edge_idx]
                
                # Try a new color
                new_color = rng.randint(0, target_colors - 1)
                if new_color == old_color:
                    continue
                
                delta = _delta_bad_for_change(c, edge_idx, new_color)
                
                if delta <= 0:
                    c[edge_idx] = new_color
                    bad += delta
                    stuck_counter = 0
                    
                    if bad < best_local_bad:
                        best_local_bad = bad
                        best_local_c = c.copy()
                        
                    if bad == 0:
                        actual = _count_colors_used(c)
                        if actual < best_score:
                            best_score = actual
                            best_colors = c.copy()
                            best_bad = 0
                        break
                else:
                    # Simulated annealing: accept with probability exp(-delta/T)
                    if rng.random() < np.exp(-delta / T):
                        c[edge_idx] = new_color
                        bad += delta
                        stuck_counter += 1
                    else:
                        stuck_counter += 1
                
                # Cool down
                T = max(T_min, T * cooling)
                
                # If stuck too long, restart from best local
                if stuck_counter > 5000:
                    c = best_local_c.copy()
                    bad = best_local_bad
                    T = 2.0
                    stuck_counter = 0
            
            # After local search, check if we improved
            if bad == 0:
                actual = _count_colors_used(c)
                if actual < best_score:
                    best_score = actual
                    best_colors = c.copy()
                    best_bad = 0
                break  # Found valid for this target
            
            # If we got close, try to polish
            if bad <= 5:
                # Intensive local search on the best local solution
                c = best_local_c.copy()
                bad = best_local_bad
                T = 1.0
                for it in range(100000):
                    if time.time() > deadline - 10:
                        break
                    if bad == 0:
                        actual = _count_colors_used(c)
                        if actual < best_score:
                            best_score = actual
                            best_colors = c.copy()
                            best_bad = 0
                        break
                    
                    edge_idx = rng.randint(0, _NUM_EDGES - 1)
                    old_color = c[edge_idx]
                    new_color = rng.randint(0, target_colors - 1)
                    if new_color == old_color:
                        continue
                    
                    delta = _delta_bad_for_change(c, edge_idx, new_color)
                    
                    if delta <= 0:
                        c[edge_idx] = new_color
                        bad += delta
                    else:
                        if rng.random() < np.exp(-delta / T):
                            c[edge_idx] = new_color
                            bad += delta
                    
                    T = max(0.01, T * 0.9999)
                
                if bad == 0:
                    actual = _count_colors_used(c)
                    if actual < best_score:
                        best_score = actual
                        best_colors = c.copy()
                        best_bad = 0
                    break
    
    # Final: try to reduce colors used in the best valid coloring
    if best_bad == 0:
        # Try to minimize colors used via local search
        c = best_colors.copy()
        bad = 0
        used = _count_colors_used(c)
        
        for it in range(100000):
            if time.time() > deadline - 5:
                break
            
            # Try to remap a color to an unused one to reduce count
            color_counts = {}
            for col in c:
                color_counts[col] = color_counts.get(col, 0) + 1
            
            # Find the least-used color
            if not color_counts:
                break
            min_color = min(color_counts, key=lambda x: color_counts[x])
            
            # Try to change all edges with min_color to another color
            edges_with_min = [i for i in range(_NUM_EDGES) if c[i] == min_color]
            
            if not edges_with_min:
                break
            
            # Try changing one edge at a time to a different used color
            for ei in edges_with_min:
                if time.time() > deadline - 5:
                    break
                
                # Find what colors are used by other edges in K4s containing ei
                # Just try random remapping
                new_color = rng.randint(0, target_colors - 1)
                if new_color == min_color:
                    continue
                
                delta = _delta_bad_for_change(c, ei, new_color)
                if delta <= 0:
                    c[ei] = new_color
                    bad += delta
            
            # Check if we reduced colors
            new_used = _count_colors_used(c)
            if new_used < used:
                used = new_used
                best_score = used
                best_colors = c.copy()
    
    # Final validation
    if _count_bad_k4s(best_colors) > 0:
        # Fallback to original 17-coloring
        return [(i + j) % 17 for i in range(_N) for j in range(i + 1, _N)]
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
