"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_idx(i, j):
    """Index for edge (i,j) with i<j in lexicographic order."""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _precompute_cycles():
    """Precompute all 4-subsets and their three cycles' edge indices."""
    quads = list(itertools.combinations(range(N), 4))
    quad_cycles = []
    for quad in quads:
        a, b, c, d = quad
        cyc1 = [_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(a,d)]
        cyc2 = [_edge_idx(a,b), _edge_idx(b,d), _edge_idx(c,d), _edge_idx(a,c)]
        cyc3 = [_edge_idx(a,c), _edge_idx(b,c), _edge_idx(b,d), _edge_idx(a,d)]
        quad_cycles.append((cyc1, cyc2, cyc3))
    return quad_cycles

_CYCLE_CACHE = None

def _get_cycles():
    global _CYCLE_CACHE
    if _CYCLE_CACHE is None:
        _CYCLE_CACHE = _precompute_cycles()
    return _CYCLE_CACHE

def _fast_check(colors):
    """Fast check: no 4-cycle has 4 distinct colors."""
    for cyc1, cyc2, cyc3 in _get_cycles():
        if len({colors[cyc1[0]], colors[cyc1[1]], colors[cyc1[2]], colors[cyc1[3]]}) == 4:
            return False
        if len({colors[cyc2[0]], colors[cyc2[1]], colors[cyc2[2]], colors[cyc2[3]]}) == 4:
            return False
        if len({colors[cyc3[0]], colors[cyc3[1]], colors[cyc3[2]], colors[cyc3[3]]}) == 4:
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin out of 90s
    
    best_colors = [0] * NUM_EDGES
    best_score = 1
    
    rng = random.Random(42)
    
    def try_candidate(colors):
        nonlocal best_colors, best_score
        if time.time() > deadline:
            return
        if _fast_check(colors):
            score = _count_distinct(colors)
            if score > best_score:
                best_score = score
                best_colors = list(colors)
    
    # Strategy 1: 2D linear (i*p + j*q) mod M
    for M in [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        for p in range(M):
            for q in range(M):
                if p == 0 and q == 0:
                    continue
                colors = [0] * NUM_EDGES
                for i in range(N):
                    for j in range(i + 1, N):
                        colors[_edge_idx(i, j)] = (i * p + j * q) % M
                try_candidate(colors)
    
    # Strategy 2: 2D with interaction (i*p + j*q + i*j*r) mod M
    for M in [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        for p in range(M):
            for q in range(M):
                for r in range(M):
                    if time.time() > deadline:
                        break
                    colors = [0] * NUM_EDGES
                    for i in range(N):
                        for j in range(i + 1, N):
                            colors[_edge_idx(i, j)] = (i * p + j * q + i * j * r) % M
                    try_candidate(colors)
    
    # Strategy 3: Additive f(i)+f(j) mod M with random f
    for M in [16, 17, 19, 20, 23, 25, 29, 32, 37, 41, 49, 64, 100, 120]:
        if time.time() > deadline:
            break
        for trial in range(500):
            if time.time() > deadline:
                break
            f_vals = [rng.randint(0, M - 1) for _ in range(N)]
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    colors[_edge_idx(i, j)] = (f_vals[i] + f_vals[j]) % M
            try_candidate(colors)
    
    # Strategy 4: Multiplicative f(i)*f(j) mod M
    for M in [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        for trial in range(200):
            if time.time() > deadline:
                break
            f_vals = [rng.randint(1, M - 1) for _ in range(N)]
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    colors[_edge_idx(i, j)] = (f_vals[i] * f_vals[j]) % M
            try_candidate(colors)
    
    # Strategy 5: XOR
    for trial in range(3000):
        if time.time() > deadline:
            break
        f_vals = [rng.randint(0, 15) for _ in range(N)]
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_idx(i, j)] = f_vals[i] ^ f_vals[j]
        try_candidate(colors)
    
    # Strategy 6: Mixed linear a*f(i) + b*f(j) mod M
    for M in [16, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        if time.time() > deadline:
            break
        for a in range(1, min(M, 8)):
            for b in range(1, min(M, 8)):
                if time.time() > deadline:
                    break
                for trial in range(30):
                    if time.time() > deadline:
                        break
                    f_vals = [rng.randint(0, M - 1) for _ in range(N)]
                    colors = [0] * NUM_EDGES
                    for i in range(N):
                        for j in range(i + 1, N):
                            colors[_edge_idx(i, j)] = (a * f_vals[i] + b * f_vals[j]) % M
                    try_candidate(colors)
    
    # Strategy 7: Direct formulas with constants
    for M in range(2, 120):
        if time.time() > deadline:
            break
        for r in range(min(M, 5)):
            # i+j+r
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    colors[_edge_idx(i, j)] = (i + j + r) % M
            try_candidate(colors)
            # i*j+r
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    colors[_edge_idx(i, j)] = (i * j + r) % M
            try_candidate(colors)
            # i*j+i+j+r
            colors = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    colors[_edge_idx(i, j)] = (i * j + i + j + r) % M
            try_candidate(colors)
    
    # Strategy 8: Quadratic forms
    for M in range(2, 120):
        if time.time() > deadline:
            break
        for a in range(min(M, 6)):
            for b in range(min(M, 6)):
                for c in range(min(M, 6)):
                    if time.time() > deadline:
                        break
                    colors = [0] * NUM_EDGES
                    for i in range(N):
                        for j in range(i + 1, N):
                            colors[_edge_idx(i, j)] = (a*i*i + b*j*j + c*i*j) % M
                    try_candidate(colors)
    
    # Strategy 9: Local search from best
    if time.time() < deadline - 5:
        current = list(best_colors)
        for iteration in range(100000):
            if time.time() > deadline:
                break
            ei = rng.randint(0, NUM_EDGES - 1)
            old_c = current[ei]
            new_c = rng.randint(0, 119)
            current[ei] = new_c
            if _fast_check(current):
                score = _count_distinct(current)
                if score > best_score:
                    best_score = score
                    best_colors = list(current)
            else:
                current[ei] = old_c
    
    return best_colors

# EVOLVE-BLOCK-END
