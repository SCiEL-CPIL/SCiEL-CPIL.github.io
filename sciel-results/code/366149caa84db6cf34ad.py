"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _precompute_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = (_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a))
                    e2 = (_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a))
                    e3 = (_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a))
                    cycles.append(e1)
                    cycles.append(e2)
                    cycles.append(e3)
    return cycles

_CYCLES = _precompute_cycles()

def _is_valid(colors):
    for cycle in _CYCLES:
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_colors = [0] * NUM_EDGES
    best_score = 0
    
    rng = random.Random(42)
    
    # Phase 1: Greedy with random restarts
    for trial in range(500):
        if time.time() > deadline - 10:
            break
        
        colors = [0] * NUM_EDGES
        
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        for ei in edge_order:
            if time.time() > deadline - 10:
                break
            found = False
            for c in range(NUM_COLORS - 1, -1, -1):
                colors[ei] = c
                if _is_valid(colors):
                    found = True
                    break
            if not found:
                colors[ei] = 0
        
        score = _count_colors(colors)
        if score > best_score:
            best_score = score
            best_colors = colors[:]
    
    # Phase 2: Local search
    if time.time() < deadline - 5:
        colors = best_colors[:]
        score = _count_colors(colors)
        
        for iteration in range(100000):
            if time.time() > deadline - 2:
                break
            
            ei = rng.randint(0, NUM_EDGES - 1)
            old_c = colors[ei]
            
            new_c = rng.randint(0, NUM_COLORS - 1)
            if new_c == old_c:
                continue
            
            colors[ei] = new_c
            if _is_valid(colors):
                new_score = _count_colors(colors)
                if new_score >= score:
                    score = new_score
                    if score > best_score:
                        best_score = score
                        best_colors = colors[:]
                else:
                    colors[ei] = old_c
            else:
                colors[ei] = old_c
    
    return best_colors
# EVOLVE-BLOCK-END
