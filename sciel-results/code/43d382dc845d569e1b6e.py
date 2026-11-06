import time
import sys
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit for safety (90s kill)
    
    # Precompute edge index mapping
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        # Row i has edges to i+1, i+2, ..., N-1
        # Previous rows: 0 has N-1, 1 has N-2, ..., i-1 has N-i
        # So offset = sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
        # Wait, let me recalculate:
        # Row 0: N-1 edges (to 1..N-1)
        # Row 1: N-2 edges (to 2..N-1)
        # ...
        # Row i: N-1-i edges (to i+1..N-1)
        # Total before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
        # Position in row i: j - i - 1
        offset = i * (N - 1) - i * (i - 1) // 2
        return offset + (j - i - 1)
    
    # Start with binary partition Gallai coloring
    color_matrix = [[0] * N for _ in range(N)]
    color_id = 0
    
    def assign_color(i, j, c):
        color_matrix[i][j] = c
        color_matrix[j][i] = c
    
    def recurse(vertices):
        nonlocal color_id
        if len(vertices) <= 1:
            return
        if time.time() > deadline - 10:
            return
        mid = len(vertices) // 2
        left = vertices[:mid]
        right = vertices[mid:]
        cross_color = color_id
        color_id += 1
        for u in left:
            for v in right:
                assign_color(u, v, cross_color)
        recurse(left)
        recurse(right)
    
    recurse(list(range(N)))
    
    # Flatten initial
    initial_flat = [color_matrix[i][j] for i in range(N) for j in range(i+1, N)]
    initial_colors = set(initial_flat)
    next_color = max(initial_colors) + 1 if initial_colors else 0
    
    edge_list = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Optimize check: precompute for each edge (i,j), the list of (k,l) pairs to check
    # For edge (i,j), we need to check all pairs (k,l) from V\{i,j}
    # There are C(14,2) = 91 such pairs
    # For each, we check 3 cycles
    
    def check_valid(cm, i, j, new_color):
        """Check if recoloring edge (i,j) to new_color keeps all 4-cycles non-rainbow."""
        others = [v for v in range(N) if v != i and v != j]
        for idx_k in range(len(others)):
            k = others[idx_k]
            for idx_l in range(idx_k + 1, len(others)):
                l = others[idx_l]
                # Three 4-cycles on {i, j, k, l}
                # Cycle 1: i-k-j-l-i
                c1 = cm[i][k]
                c2 = cm[k][j]
                c3 = new_color  # cm[j][l] would be cm[j][l], but we're checking j-l edge
                # Wait, I need to be more careful.
                # The three cycles on vertices {i,j,k,l} are:
                # 1) i-k-j-l-i: edges (i,k), (k,j), (j,l), (l,i)
                # 2) i-k-l-j-i: edges (i,k), (k,l), (l,j), (j,i)
                # 3) i-l-k-j-i: edges (i,l), (l,k), (k,j), (j,i)
                
                # Cycle 1: (i,k), (k,j), (j,l), (l,i)
                colors1 = [cm[i][k], cm[k][j], cm[j][l], cm[l][i]]
                if len(set(colors1)) == 4:
                    return False
                
                # Cycle 2: (i,k), (k,l), (l,j), (j,i)
                colors2 = [cm[i][k], cm[k][l], cm[l][j], new_color]
                if len(set(colors2)) == 4:
                    return False
                
                # Cycle 3: (i,l), (l,k), (k,j), (j,i)
                colors3 = [cm[i][l], cm[l][k], cm[k][j], new_color]
                if len(set(colors3)) == 4:
                    return False
        return True
    
    best_count = len(initial_colors)
    best_flat = initial_flat[:]
    
    # Greedy recoloring with multiple passes and randomization
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        # Shuffle edge order
        random.shuffle(edge_list)
        
        for (i, j) in edge_list:
            if time.time() > deadline - 3:
                break
            
            old_color = color_matrix[i][j]
            new_color = next_color
            
            color_matrix[i][j] = new_color
            color_matrix[j][i] = new_color
            
            if check_valid(color_matrix, i, j, new_color):
                next_color += 1
                current_count = next_color - 1
                if current_count > best_count:
                    best_count = current_count
                    best_flat = [color_matrix[a][b] for a in range(N) for b in range(a+1, N)]
            else:
                color_matrix[i][j] = old_color
                color_matrix[j][i] = old_color
    
    return best_flat

if __name__ == "__main__":
    c = construct()
    print(len(set(c)))
