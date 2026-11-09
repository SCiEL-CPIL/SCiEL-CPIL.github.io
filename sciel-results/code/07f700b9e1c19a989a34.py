"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

# EVOLVE-BLOCK-START
def _edge_idx(i, j):
    """Map vertex pair (i,j) with i<j to edge index."""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def _all_quadruples():
    """Generate all 4-vertex subsets and their 3 cycle edge sets."""
    from itertools import combinations
    quads = []
    for a, b, c, d in combinations(range(N), 4):
        # Three 4-cycles on {a,b,c,d}:
        # Cycle 1: a-b-c-d-a -> edges (a,b),(b,c),(c,d),(d,a)
        # Cycle 2: a-b-d-c-a -> edges (a,b),(b,d),(d,c),(c,a)
        # Cycle 3: a-c-b-d-a -> edges (a,c),(c,b),(b,d),(d,a)
        c1 = [(_edge_idx(a,b), _edge_idx(b,c), _edge_idx(c,d), _edge_idx(d,a))]
        c2 = [(_edge_idx(a,b), _edge_idx(b,d), _edge_idx(d,c), _edge_idx(c,a))]
        c3 = [(_edge_idx(a,c), _edge_idx(c,b), _edge_idx(b,d), _edge_idx(d,a))]
        quads.append((c1[0], c2[0], c3[0]))
    return quads

def _count_distinct_colors(colors):
    return len(set(colors))

def _is_valid(colors):
    """Check if no 4-cycle has 4 distinct colors."""
    for c1, c2, c3 in _ALL_QUADS:
        for cyc in (c1, c2, c3):
            if len(set(colors[e] for e in cyc)) == 4:
                return False
    return True

# Precompute all quadruples
_ALL_QUADS = _all_quadruples()

def _violation_count(colors):
    """Count number of violating 4-cycles."""
    count = 0
    for c1, c2, c3 in _ALL_QUADS:
        for cyc in (c1, c2, c3):
            if len(set(colors[e] for e in cyc)) == 4:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    # Start with a valid base: color by (i+j) mod K for some K
    # This gives a valid coloring since any 4-cycle will have repeated (i+j) mod K patterns
    # Actually, let's use a simpler valid base: all same color
    best_colors = [0] * NUM_EDGES
    best_distinct = 1
    
    # Try structured initializations
    random.seed(42)
    
    # Strategy: use a "block" coloring then perturb
    # Divide 16 vertices into 4 blocks of 4
    # Color edge (i,j) based on block pair and position
    
    # Let's try: color = (block_i * 4 + block_j) if block_i != block_j, else (i+j) mod 4
    # This uses limited colors but is likely valid
    
    def _block_coloring():
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                bi = i // 4
                bj = j // 4
                if bi == bj:
                    # Within same block: use position-based
                    colors[_edge_idx(i,j)] = (i + j) % 4
                else:
                    # Between blocks: use block pair
                    colors[_edge_idx(i,j)] = 4 + bi * 4 + bj
        return colors
    
    # Try multiple structured inits
    inits = []
    inits.append([0] * NUM_EDGES)
    inits.append(_block_coloring())
    
    # Another: color by (i*j) mod 16
    def _mult_coloring():
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[_edge_idx(i,j)] = (i * j) % 16
        return colors
    inits.append(_mult_coloring())
    
    # Color by (i+j) mod 16
    def _sum_coloring():
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[_edge_idx(i,j)] = (i + j) % 16
        return colors
    inits.append(_sum_coloring())
    
    # Color by (j-i) mod 16
    def _diff_coloring():
        colors = [0] * NUM_EDGES
        for i in range(N):
            for j in range(i+1, N):
                colors[_edge_idx(i,j)] = (j - i) % 16
        return colors
    inits.append(_diff_coloring())
    
    for init in inits:
        distinct = _count_distinct_colors(init)
        if _is_valid(init):
            if distinct > best_distinct:
                best_distinct = distinct
                best_colors = init[:]
    
    # Local search: try to increase distinct colors
    # Pick a random edge, try changing its color to a new unused color
    # Check if still valid; if so, keep it; else revert
    # Repeat until deadline
    
    used_colors = set(best_colors)
    
    # Precompute for speed: for each edge, which 4-cycles contain it
    # Actually, let's just do random perturbations with validation
    
    np.random.seed(42)
    
    for _ in range(500000):
        if time.time() > deadline:
            break
        
        # Try to introduce a new color on a random edge
        edge = np.random.randint(NUM_EDGES)
        new_color = np.random.randint(MAX_COLOR + 1)
        
        if new_color == best_colors[edge]:
            continue
        
        old_color = best_colors[edge]
        best_colors[edge] = new_color
        
        # Quick check: only need to check 4-cycles containing this edge
        # But for simplicity, check all (or we can optimize)
        # Let's do a full check for correctness
        if _is_valid(best_colors):
            distinct = _count_distinct_colors(best_colors)
            if distinct > best_distinct:
                best_distinct = distinct
            # Keep the change
        else:
            # Revert
            best_colors[edge] = old_color
    
    # Try to add more distinct colors by trying all unused colors on random edges
    all_colors = set(range(MAX_COLOR + 1))
    used = set(best_colors)
    unused = all_colors - used
    
    for _ in range(200000):
        if time.time() > deadline:
            break
        if not unused:
            break
        
        edge = np.random.randint(NUM_EDGES)
        new_color = np.random.choice(list(unused))
        
        old_color = best_colors[edge]
        best_colors[edge] = new_color
        
        if _is_valid(best_colors):
            distinct = _count_distinct_colors(best_colors)
            if distinct > best_distinct:
                best_distinct = distinct
                unused = all_colors - set(best_colors)
        else:
            best_colors[edge] = old_color
    
    return best_colors
# EVOLVE-BLOCK-END
