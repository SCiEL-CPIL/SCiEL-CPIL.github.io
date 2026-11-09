import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2

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
                _CYCLES.append([_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)])
                _CYCLES.append([_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)])
                _CYCLES.append([_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)])

NUM_CYCLES = len(_CYCLES)

# Precompute: for each edge, which cycles contain it
_edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(_CYCLES):
    for e in cycle:
        _edge_to_cycles[e].append(ci)

# Convert cycles to numpy array for fast checking
_cycles_arr = np.array(_CYCLES, dtype=np.int32)

def _is_valid_fast(colors):
    """Fast validity check using numpy."""
    # colors is a numpy array
    # For each cycle, check if all 4 colors are distinct
    # Get colors for all edges in all cycles at once
    cycle_colors = colors[_cycles_arr]  # shape (NUM_CYCLES, 4)
    # Check if any cycle has 4 distinct colors
    # A cycle has 4 distinct colors iff:
    # c0 != c1, c0 != c2, c0 != c3, c1 != c2, c1 != c3, c2 != c3
    c0 = cycle_colors[:, 0]
    c1 = cycle_colors[:, 1]
    c2 = cycle_colors[:, 2]
    c3 = cycle_colors[:, 3]
    
    all_distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
    return not np.any(all_distinct)

def _find_violations_fast(colors):
    """Return indices of violating cycles."""
    cycle_colors = colors[_cycles_arr]
    c0 = cycle_colors[:, 0]
    c1 = cycle_colors[:, 1]
    c2 = cycle_colors[:, 2]
    c3 = cycle_colors[:, 3]
    
    all_distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & (c1 != c2) & (c1 != c3) & (c2 != c3)
    return np.where(all_distinct)[0]

def _check_edge_change(colors, edge_idx, new_color):
    """Check if changing edge_idx to new_color keeps the coloring valid.
    Only checks cycles containing edge_idx."""
    old_color = colors[edge_idx]
    if new_color == old_color:
        return True
    
    colors[edge_idx] = new_color
    # Check only cycles containing this edge
    for ci in _edge_to_cycles[edge_idx]:
        e0, e1, e2, e3 = _CYCLES[ci]
        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
        if len({c0, c1, c2, c3}) == 4:
            colors[edge_idx] = old_color
            return False
    
    colors[edge_idx] = old_color
    return True

def construct():
    deadline = time.time() + 85.0
    
    random.seed(42)
    
    # Start with all zeros
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    current_score = 1
    
    # Strategy: iteratively try to add new colors
    # For each new color, try to assign it to some edges while maintaining validity
    
    # First, try structured initial colorings
    structured_patterns = []
    
    # Pattern 1: (i+j) % k
    for k in range(2, 25):
        test = np.zeros(NUM_EDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                test[_edge_idx(i, j)] = (i + j) % k
        structured_patterns.append(test)
    
    # Pattern 2: (i*j) % k
    for k in range(2, 25):
        test = np.zeros(NUM_EDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                test[_edge_idx(i, j)] = (i * j) % k
        structured_patterns.append(test)
    
    # Pattern 3: (i-j) % k (using absolute difference for symmetry)
    for k in range(2, 25):
        test = np.zeros(NUM_EDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                test[_edge_idx(i, j)] = (j - i) % k
        structured_patterns.append(test)
    
    # Pattern 4: random with small palette
    for seed in range(10):
        rng = np.random.RandomState(seed)
        test = rng.randint(0, 4, size=NUM_EDGES).astype(np.int32)
        structured_patterns.append(test)
    
    # Try each structured pattern
    for pattern in structured_patterns:
        if time.time() > deadline:
            break
        if _is_valid_fast(pattern):
            score = len(set(pattern.tolist()))
            if score > current_score:
                current_score = score
                colors = pattern.copy()
    
    # Local search phase
    # Try to increase the number of distinct colors
    
    no_improve = 0
    max_no_improve = 200000
    step = 0
    
    while time.time() < deadline and no_improve < max_no_improve:
        step += 1
        
        # Try to add a new color or swap to increase diversity
        # Pick a random edge and try to change its color
        edge_idx = random.randint(0, NUM_EDGES - 1)
        old_color = colors[edge_idx]
        
        # Try a random new color
        new_color = random.randint(0, 119)
        
        if new_color == old_color:
            continue
        
        # Check if this change is valid
        if _check_edge_change(colors, edge_idx, new_color):
            colors[edge_idx] = new_color
            new_score = len(set(colors.tolist()))
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
            else:
                # Revert if no improvement in score
                colors[edge_idx] = old_color
                no_improve += 1
        else:
            no_improve += 1
    
    return colors.tolist()
