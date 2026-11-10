import time
import random

N = 16

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles on a,b,c,d
                    # a-b-c-d-a
                    cycles.append((_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a)))
                    # a-b-d-c-a
                    cycles.append((_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a)))
                    # a-c-b-d-a
                    cycles.append((_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a)))
    return cycles

def _is_valid(colors, cycles):
    for e1, e2, e3, e4 in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    cycles = _build_cycles()
    
    best_colors = [0] * (N*(N-1)//2)
    best_score = 1
    
    # Try algebraic colorings
    # Form: (A*i + B*j) % K
    # We want to maximize K.
    # Let's try a few forms and high Ks.
    
    forms = [
        lambda i, j, K: (i + j) % K,
        lambda i, j, K: (i - j) % K,
        lambda i, j, K: (i * j) % K,
        lambda i, j, K: (i + 2*j) % K,
        lambda i, j, K: (2*i + j) % K,
        lambda i, j, K: (i * j + i + j) % K,
    ]
    
    # Try K from 20 down to 1
    # If a form with K is valid, score is at least K (if all 0..K-1 used)
    # Actually score is number of distinct colors.
    
    for K in range(20, 0, -1):
        if time.time() > deadline:
            break
        for f in forms:
            colors = [0] * (N*(N-1)//2)
            for i in range(N):
                for j in range(i+1, N):
                    idx = _edge_idx(i, j)
                    colors[idx] = f(i, j, K)
            
            if _is_valid(colors, cycles):
                score = _count_colors(colors)
                if score > best_score:
                    best_score = score
                    best_colors = colors[:]
    
    # If we have time, do a random local search to try to improve
    if time.time() < deadline - 10.0:
        # Random restart local search
        # Start from best_colors
        current_colors = best_colors[:]
        current_score = best_score
        
        # Simple random perturbation
        # Pick a random edge, change its color to a random value 0..119
        # Check validity. If valid and score increases, keep.
        # If valid but score same, keep with some probability? No, just keep if valid.
        # If invalid, revert.
        
        # This is slow if done naively.
        # Given the small N, maybe just try to add colors to the existing valid set?
        # The algebraic solution is likely near optimal for this specific constraint class.
        
        pass

    return best_colors

# EVOLVE-BLOCK-START
def construct():
    # Use the helper defined above
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    start_time = time.time()
    deadline = start_time + 50.0
    
    cycles = _build_cycles()
    
    best_colors = [0] * (N*(N-1)//2)
    best_score = 1
    
    forms = [
        lambda i, j, K: (i + j) % K,
        lambda i, j, K: (i - j) % K,
        lambda i, j, K: (i * j) % K,
        lambda i, j, K: (i + 2*j) % K,
        lambda i, j, K: (2*i + j) % K,
        lambda i, j, K: (i * j + i + j) % K,
        lambda i, j, K: (i + j + i*j) % K,
    ]
    
    for K in range(25, 0, -1):
        if time.time() > deadline:
            break
        for f in forms:
            colors = [0] * (N*(N-1)//2)
            for i in range(N):
                for j in range(i+1, N):
                    idx = _edge_idx(i, j)
                    colors[idx] = f(i, j, K)
            
            if _is_valid(colors, cycles):
                score = _count_colors(colors)
                if score > best_score:
                    best_score = score
                    best_colors = colors[:]
    
    return best_colors
