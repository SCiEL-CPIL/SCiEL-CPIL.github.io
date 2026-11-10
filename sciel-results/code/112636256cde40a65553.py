"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools

N = 16
NUM_PAIRS = N * (N - 1) // 2  # 120
MAX_COLOR = 119

# Precompute all 4-cycles: for each set of 4 vertices, the 3 possible 4-cycles
# A 4-cycle on vertices {a,b,c,d} can be: a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
# Each cycle consists of 4 edges.
# We need to check that no 4-cycle has 4 distinct edge colors.

# Let's precompute the list of 4-cycles as tuples of edge indices
# Edge index for (i,j) with i<j: we use a mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Lexicographic ordering of pairs (i,j) with 0<=i<j<16
    # Count pairs before (i,j):
    # For each k < i: there are (15-k) pairs starting with k
    # Then for pairs starting with i: (i+1, i+2, ..., j-1) -> j-1-i pairs
    cnt = 0
    for k in range(i):
        cnt += (N - 1 - k)
    cnt += (j - 1 - i)
    return cnt

# Verify: edge_idx(0,1) should be 0
# k=0: cnt += 15, then j-1-i = 1-1-0=0, so 15? That's wrong.

# Let me recalculate. Pairs are ordered lexicographically:
# (0,1), (0,2), ..., (0,15), (1,2), (1,3), ..., (1,15), ..., (14,15)
# Index of (i,j): sum_{k=0}^{i-1} (N-1-k) + (j-1-i)
# For (0,1): sum_{k=0}^{-1} = 0, + (1-1-0) = 0. Correct.
# For (0,2): 0 + (2-1-0) = 1. Correct.
# For (1,2): sum_{k=0}^{0} (15-k) = 15, + (2-1-1)=0, total 15. 
# But (0,15) is index 14, so (1,2) should be 15. Correct.

def edge_index(i, j):
    if i > j:
        i, j = j, i
    idx = 0
    for k in range(i):
        idx += (N - 1 - k)
    idx += (j - 1 - i)
    return idx

# Precompute all 4-cycles
# For each combination of 4 vertices (a,b,c,d) with a<b<c<d
# The three 4-cycles are:
# 1. a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
# 2. a-b-d-c-a: edges (a,b), (b,d), (d,c), (c,a)
# 3. a-c-b-d-a: edges (a,c), (c,b), (b,d), (d,a)

all_cycles = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    cycle1 = [edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a)]
    # Cycle 2: a-b-d-c-a
    cycle2 = [edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a)]
    # Cycle 3: a-c-b-d-a
    cycle3 = [edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a)]
    all_cycles.append(cycle1)
    all_cycles.append(cycle2)
    all_cycles.append(cycle3)

def count_colors(colors):
    return len(set(colors))

def is_valid(colors):
    for cycle in all_cycles:
        c = [colors[e] for e in cycle]
        if len(set(c)) == 4:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a coloring of K_16 edges with colors in 0..119 such that
    no 4-cycle has 4 distinct edge colors. Maximize number of distinct colors.
    """
    best_colors = None
    best_count = 0
    
    # Strategy: Use a structured approach with randomization and local search.
    # Start with a base pattern and use hill-climbing to add more colors.
    
    # Initialize with a simple valid coloring
    # Use a bipartite-like structure or modular arithmetic
    
    # Try multiple random initializations and local search
    random.seed(42)
    
    for attempt in range(50):
        # Random initialization with limited colors
        num_init_colors = random.randint(2, 6)
        colors = [random.randint(0, num_init_colors - 1) for _ in range(NUM_PAIRS)]
        
        # Ensure validity first
        if not is_valid(colors):
            # Fix by reducing to 2 colors if needed
            colors = [0] * NUM_PAIRS
            # Add some structure
            for i in range(N):
                for j in range(i + 1, N):
                    idx = edge_index(i, j)
                    colors[idx] = (i + j) % 2
        
        current_count = count_colors(colors)
        
        # Local search: try to change edges to new colors
        for iteration in range(2000):
            # Pick a random edge
            e = random.randint(0, NUM_PAIRS - 1)
            old_color = colors[e]
            
            # Try to assign a new color
            # First try a completely new color
            if current_count < MAX_COLOR:
                # Find a color not currently used
                used = set(colors)
                new_color = random.randint(0, MAX_COLOR)
                while new_color in used:
                    new_color = random.randint(0, MAX_COLOR)
                    if new_color > MAX_COLOR:
                        new_color = 0
                # Check if assigning new_color to e keeps validity
                colors[e] = new_color
                if is_valid(colors):
                    current_count += 1
                    continue
                else:
                    colors[e] = old_color
            
            # Try changing to an existing color
            other_color = random.randint(0, MAX_COLOR)
            if other_color == old_color:
                continue
            colors[e] = other_color
            if is_valid(colors):
                # Check if this reduced color count
                new_count = count_colors(colors)
                if new_count >= current_count:
                    current_count = new_count
                else:
                    colors[e] = old_color
            else:
                colors[e] = old_color
        
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
        
        if best_count >= 10:  # Good enough for now
            break
    
    if best_colors is None:
        # Fallback: all zeros
        return [0] * NUM_PAIRS
    
    return best_colors
# EVOLVE-BLOCK-END
