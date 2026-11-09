import time
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def build_multiway(vertices, next_color, randomize=False, max_depth=0):
        """Build coloring using multi-way hierarchical clustering.
        Each internal node splits into k children (2<=k<=4), assigning one color
        to all cross-edges between different children.
        """
        n = len(vertices)
        if n <= 1:
            return next_color
        if n == 2:
            u, v = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            return next_color + 1
        if n == 3:
            u, v, w = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            colors[v][w] = next_color + 1
            colors[w][v] = next_color + 1
            colors[u][w] = next_color + 2
            colors[w][u] = next_color + 2
            return next_color + 3
        
        # Choose number of children (2-4 for n>=4)
        if randomize:
            if n <= 4:
                k_children = 2
            elif n <= 6:
                k_children = random.choice([2, 3])
            else:
                k_children = random.choice([2, 3, 4])
        else:
            # Deterministic: use 3-way splits when beneficial
            if n <= 4:
                k_children = 2
            elif n <= 6:
                k_children = 3
            elif n <= 9:
                k_children = 3
            else:
                k_children = 4
        
        # Partition vertices into k_children groups
        if randomize:
            # Random partition
            shuffled = list(vertices)
            random.shuffle(shuffled)
            base_size = n // k_children
            remainder = n % k_children
            groups = []
            idx = 0
            for g in range(k_children):
                size = base_size + (1 if g < remainder else 0)
                groups.append(shuffled[idx:idx+size])
                idx += size
        else:
            # Contiguous partition (deterministic)
            base_size = n // k_children
            remainder = n % k_children
            groups = []
            idx = 0
            for g in range(k_children):
                size = base_size + (1 if g < remainder else 0)
                groups.append(vertices[idx:idx+size])
                idx += size
        
        # Assign one color to all cross-edges between different groups
        cross_color = next_color
        for i in range(k_children):
            for j in range(i + 1, k_children):
                for u in groups[i]:
                    for v in groups[j]:
                        colors[u][v] = cross_color
                        colors[v][u] = cross_color
        
        next_color += 1
        
        # Recurse into each group
        for group in groups:
            if time.time() > deadline:
                return next_color
            next_color = build_multiway(group, next_color, randomize, max_depth + 1)
        
        return next_color
    
    def is_valid_recolor(u, v, new_color):
        # Check all C4s containing edge (u,v)
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                if x >= w:
                    continue
                c1 = colors[u][w]
                c2 = colors[w][x]
                c3 = colors[x][v]
                if new_color != c1 and new_color != c2 and new_color != c3:
                    return False
        return True
    
    def local_search(colors, deadline):
        edge_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        used_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
        all_colors = set(range(120))
        unused_colors = list(all_colors - used_colors)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(edge_list)
            for (u, v) in edge_list:
                if time.time() > deadline:
                    break
                if not unused_colors:
                    break
                
                old_color = colors[u][v]
                
                random.shuffle(unused_colors)
                
                for new_color in unused_colors:
                    if is_valid_recolor(u, v, new_color):
                        colors[u][v] = new_color
                        colors[v][u] = new_color
                        unused_colors.remove(new_color)
                        used_colors.add(new_color)
                        improved = True
                        break
        
        return len(used_colors)
    
    best_score = 0
    best_result = None
    
    # Run multiple attempts with different strategies
    attempts = 15
    for i in range(attempts):
        if time.time() > deadline:
            break
        
        colors = [[0] * N for _ in range(N)]
        
        # Mix deterministic and randomized multi-way splits
        rand_flag = (i > 0)
        
        build_multiway(list(range(N)), 0, rand_flag)
        
        # Local search
        current_score = local_search(colors, deadline)
        
        if current_score > best_score:
            best_score = current_score
            best_result = []
            for r in range(N):
                for c in range(r+1, N):
                    best_result.append(colors[r][c])
    
    if best_result is None:
        # Fallback
        colors = [[0] * N for _ in range(N)]
        build_multiway(list(range(N)), 0, False)
        local_search(colors, deadline)
        best_result = []
        for r in range(N):
            for c in range(r+1, N):
                best_result.append(colors[r][c])
                
    return best_result
