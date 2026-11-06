"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Returns a list of 120 integer colors for the edges of K_16.
    Uses a local search approach to maximize distinct colors while
    maintaining the C4-rainbow-free property.
    """
    # Deadline guard
    deadline = time.time() + 55  # 55s safety margin
    
    # Edge index mapping
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        # Lexicographic order: (0,1),(0,2),...,(0,15),(1,2),(1,3),...
        # Index = i*(N-1) - i*(i+1)//2 + (j-i-1)
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    def get_edge(i, j):
        if i > j:
            i, j = j, i
        return edge_idx(i, j)
    
    # Precompute all 4-cycles (4-vertex subsets and their 3 cycles)
    # For 4 vertices a<b<c<d, the three 4-cycles are:
    # Cycle 1: a-b-c-d-a -> edges (a,b),(b,c),(c,d),(d,a)
    # Cycle 2: a-b-d-c-a -> edges (a,b),(b,d),(d,c),(c,a)
    # Cycle 3: a-c-b-d-a -> edges (a,c),(c,b),(b,d),(d,a)
    four_cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    c1 = [get_edge(a,b), get_edge(b,c), get_edge(c,d), get_edge(d,a)]
                    # Cycle 2: a-b-d-c-a
                    c2 = [get_edge(a,b), get_edge(b,d), get_edge(d,c), get_edge(c,a)]
                    # Cycle 3: a-c-b-d-a
                    c3 = [get_edge(a,c), get_edge(c,b), get_edge(b,d), get_edge(d,a)]
                    four_cycles.append((c1, c2, c3))
    
    # Start with the simple valid coloring: color(i,j) = i
    colors = [0] * 120
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i, j)] = i
    
    # Helper to check if a coloring is valid (no rainbow 4-cycles)
    def is_valid(col):
        for c1, c2, c3 in four_cycles:
            if len(set([col[c1[0]], col[c1[1]], col[c1[2]], col[c1[3]]])) == 4:
                return False
            if len(set([col[c2[0]], col[c2[1]], col[c2[2]], col[c2[3]]])) == 4:
                return False
            if len(set([col[c3[0]], col[c3[1]], col[c3[2]], col[c3[3]]])) == 4:
                return False
        return True
    
    # Helper to count distinct colors
    def count_distinct(col):
        return len(set(col))
    
    best_colors = colors[:]
    best_count = count_distinct(best_colors)
    
    # Local search: try changing individual edges to new colors
    random.seed(42)
    edge_list = list(range(120))
    
    while time.time() < deadline:
        # Shuffle edges
        random.shuffle(edge_list)
        
        improved = False
        for ei in edge_list:
            if time.time() > deadline:
                break
            
            old_color = colors[ei]
            
            # Try all possible new colors
            for new_color in range(120):
                if new_color == old_color:
                    continue
                
                colors[ei] = new_color
                
                # Check validity
                if is_valid(colors):
                    new_count = count_distinct(colors)
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                        improved = True
                        break
                else:
                    colors[ei] = old_color
                    continue
            
            # Reset if we didn't keep the change
            if colors[ei] != old_color and not improved:
                # Check if this specific change was kept
                if colors[ei] == old_color:
                    pass
            # If we broke out of the new_color loop with an improvement, keep it
            # Otherwise, reset
            if colors[ei] != old_color:
                # This change was accepted (improved), so keep it
                pass
            else:
                # No improvement, keep old color
                pass
        
        if not improved:
            # Try random restarts with different initial colorings
            # Use a different seed or structure
            break
    
    return best_colors
# EVOLVE-BLOCK-END
