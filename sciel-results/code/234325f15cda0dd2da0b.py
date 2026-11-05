import random
import time
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Precompute K4 edges
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return 15*i - i*(i-1)//2 + (j - i - 1)
        
    k4_edges = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d),
                        edge_idx(c, d)
                    ]
                    k4_edges.append(tuple(edges))
                    
    # Initial valid coloring generator
    def get_initial_coloring(seed):
        random.seed(seed)
        # Shuffle vertex indices to break symmetry
        verts = list(range(16))
        random.shuffle(verts)
        # Map original vertex v to new index pos
        # Actually, just permuting vertices changes the structure of (i+j)%17?
        # (i+j)%17 is not invariant under arbitrary permutation.
        # So shuffling vertices creates a different valid coloring.
        
        colors = [0] * 120
        # We need to map edge (i,j) to color based on permuted indices
        # Let perm[v] be the new index of vertex v.
        # Color of edge (u,v) in original graph is (perm[u] + perm[v]) % 17
        perm = [0]*16
        for i, v in enumerate(verts):
            perm[v] = i
            
        for i in range(16):
            for j in range(i+1, 16):
                colors[edge_idx(i, j)] = (perm[i] + perm[j]) % 17
        return colors

    best_colors = get_initial_coloring(42)
    best_count = len(set(best_colors))
    
    max_color = 17
    
    def rebuild_k4s_using(col_list):
        new_k4s_using = [set() for _ in range(max_color)]
        for k4_idx, edges in enumerate(k4_edges):
            distinct_colors_in_k4 = set()
            for e in edges:
                c = col_list[e]
                distinct_colors_in_k4.add(c)
            for c in distinct_colors_in_k4:
                if c < max_color:
                    new_k4s_using[c].add(k4_idx)
        return new_k4s_using

    def is_valid_subset(k4_indices, col_list):
        for k4_idx in k4_indices:
            edges = k4_edges[k4_idx]
            c0 = col_list[edges[0]]
            c1 = col_list[edges[1]]
            c2 = col_list[edges[2]]
            c3 = col_list[edges[3]]
            c4 = col_list[edges[4]]
            c5 = col_list[edges[5]]
            mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
            if mask.bit_count() < 5:
                return False
        return True

    for restart in range(100):
        if time.time() > deadline:
            break
            
        current_colors = get_initial_coloring(restart * 12345 + 777)
        current_active = set(current_colors)
        current_k4s_using = rebuild_k4s_using(current_colors)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            cols_list = list(current_active)
            random.shuffle(cols_list)
            
            for i in range(len(cols_list)):
                for j in range(i+1, len(cols_list)):
                    c1 = cols_list[i]
                    c2 = cols_list[j]
                    if c1 not in current_active or c2 not in current_active:
                        continue
                        
                    affected = current_k4s_using[c1] | current_k4s_using[c2]
                    
                    new_colors = current_colors.copy()
                    for k in range(120):
                        if new_colors[k] == c2:
                            new_colors[k] = c1
                    
                    if is_valid_subset(affected, new_colors):
                        current_colors = new_colors
                        current_active.remove(c2)
                        current_k4s_using[c1] = current_k4s_using[c1] | current_k4s_using[c2]
                        current_k4s_using[c2] = set()
                        
                        improved = True
                        if len(current_active) < best_count:
                            best_count = len(current_active)
                            best_colors = current_colors.copy()
                        break
                if improved:
                    break
                        
    return best_colors
