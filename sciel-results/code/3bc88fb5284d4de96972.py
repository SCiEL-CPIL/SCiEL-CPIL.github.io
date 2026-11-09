import time
import random

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Greedy construction
    # n: number of vertices
    # colors: list of colors for edges (i, j) with i < j
    # We build the graph incrementally.
    
    def get_color(colors, n, i, j):
        if i > j: i, j = j, i
        # Index calculation for lexicographic order
        # Edges for vertex 0: (0,1)..(0, n-1) -> n-1 edges
        # Edges for vertex 1: (1,2)..(1, n-1) -> n-2 edges
        # ...
        # Edges for vertex i: (i, i+1)..(i, n-1) -> n-1-i edges
        # Offset for vertex i is sum_{k=0}^{i-1} (n-1-k)
        # This is O(n) to compute if done naively.
        # For n=20, it's fine.
        
        # Precompute offsets for speed?
        # For now, simple index.
        # Total edges so far is N = n*(n-1)/2.
        # But we are adding a new vertex, so n is increasing.
        # It's better to store colors in a 2D list or dict during construction.
        
        pass

    # Better approach: Store colors in a 2D list `color_matrix`.
    # color_matrix[i][j] is the color of edge (i, j).
    
    n = 1
    # color_matrix is n x n
    color_matrix = [[-1] * n for _ in range(n)]
    
    # Counters for common neighbors
    # common_neighbors[c][i][j] = number of vertices k such that color(i,k)=c and color(j,k)=c
    # This is O(n^2) memory per color. 5 * 30^2 = 4500 ints. Fine.
    
    max_n_target = 29
    current_n = 1
    
    while current_n < max_n_target:
        if time.time() - start_time > timeout:
            break
            
        # Try to add vertex current_n
        new_n = current_n + 1
        new_color_matrix = [row[:] for row in color_matrix]
        for i in range(new_n):
            if i < new_n:
                new_color_matrix[i].append(-1)
        new_color_matrix.append([-1] * new_n)
        
        # Update common neighbors for existing edges?
        # No, we just need to check if adding new edges creates C4s.
        
        # We need to assign colors to edges (new_n-1, i) for i in 0..new_n-2.
        
        # Heuristic: For each edge (new_n-1, i), choose color c that minimizes the number of new "at-risk" pairs.
        # An at-risk pair is a pair (u, v) such that they have exactly 1 common neighbor in color c.
        # If they have 2, it's a C4.
        
        # We maintain `common_count[c][u][v]`
        # Initially, for the existing graph, we have these counts.
        # When we add vertex v_new with edges to u, we update counts for pairs (u, w).
        
        # To be efficient, we will simulate the addition.
        
        # Let's use a simpler check:
        # For a candidate color c for edge (v_new, u):
        # It creates a C4 if there exists w such that color(u,w)=c and color(v_new,w)=c.
        # But v_new is new, so color(v_new,w) is not set yet for w < u?
        # We are assigning sequentially.
        
        # Let's assign colors for edges (v_new, 0), (v_new, 1), ...
        
        # State:
        # colors_assigned: list of colors for (v_new, 0), ..., (v_new, i-1)
        # common_counts: updated for the new vertex?
        
        # This is getting complex to implement correctly in a short time.
        
        # Alternative: Use the known n=25 construction if I can find it online or derive it.
        # Since I can't, I will use a random restart greedy search.
        
        # Let's implement a robust greedy search that restarts if stuck.
        
        pass
    
    # Fallback: Return the seed solution if search fails or times out.
    # But we should try to do better.
    
    # Let's implement a simple greedy that works.
    
    def greedy_construct(target_n, seed):
        random.seed(seed)
        n = 1
        # color_matrix[i][j]
        cm = [[-1] * n for _ in range(n)]
        
        # common[c][i][j]
        common = [[[0]*n for _ in range(n)] for _ in range(5)]
        
        for new_v in range(1, target_n):
            if time.time() - start_time > timeout:
                break
                
            # Extend cm
            for i in range(n):
                cm[i].append(-1)
            cm.append([-1] * (n+1))
            
            # Extend common
            for c in range(5):
                for i in range(n):
                    common[c][i].append(0)
                common[c].append([0] * (n+1))
                
            n = n + 1
            
            # Assign colors for edges (new_v-1, i) for i in 0..new_v-2
            # Order: random or sorted by degree?
            # Let's try random order to break symmetry.
            neighbors = list(range(new_v - 1))
            random.shuffle(neighbors)
            
            assigned_colors = {}
            
            for u in neighbors:
                best_c = -1
                best_score = float('inf')
                
                for c in range(5):
                    # Check if color c creates a C4 immediately
                    # A C4 is formed if there is w such that color(u,w)=c and color(new_v-1, w)=c.
                    # We have assigned colors for some w.
                    # Let's check against already assigned w's.
                    valid = True
                    risk = 0
                    
                    for w in assigned_colors:
                        if cm[u][w] == c:
                            if assigned_colors[w] == c:
                                valid = False
                                break
                            else:
                                # They share u? No.
                                # u and w are both connected to new_v-1?
                                # No, u is connected to new_v-1 with color c.
                                # w is connected to new_v-1 with color assigned_colors[w].
                                # If assigned_colors[w] == c, then new_v-1, u, w form a path?
                                # No, C4 requires 4 vertices.
                                # Vertices: new_v-1, u, w, x.
                                # Edges: (nv, u)=c, (nv, w)=c, (u, w)=? , (u, x)=c, (w, x)=c.
                                # 
                                # Immediate C4 check:
                                # If we assign color c to (nv, u), and we have already assigned color c to (nv, w),
                                # then if color(u, w) is NOT c, do we have a C4?
                                # No.
                                # We have a C4 if there exists x such that color(u,x)=c and color(w,x)=c.
                                # This x is in the existing graph.
                                # So we need to check common counts.
                                
                                pass
                                
                    # Simplified: 
                    # If we assign c to (nv, u), we must ensure that for all w already assigned c to (nv, w),
                    # the number of common neighbors of u and w in color c (in the existing graph) is 0.
                    # Because if there is x, then nv-u-x-w-nv is a C4?
                    # Edges: (nv,u)=c, (u,x)=c, (x,w)=c, (w,nv)=c.
                    # Yes.
                    
                    conflict = False
                    for w in assigned_colors:
                        if assigned_colors[w] == c:
                            if common[c][u][w] > 0:
                                conflict = True
                                break
                    if conflict:
                        continue
                    
                    # Score: number of new at-risk pairs
                    # For each w (not yet assigned or assigned different color),
                    # if we assign c to (nv, u), we increase common[c][u][w] by 1?
                    # No, common[c][u][w] counts x such that (u,x)=c and (w,x)=c.
                    # Adding (nv, u)=c adds nv as a common neighbor of u and w?
                    # Only if (w, nv)=c.
                    # If w is not yet assigned, or assigned different color, then nv is NOT a common neighbor of u and w in color c.
                    # So common[c][u][w] does NOT change.
                    
                    # However, for pairs (u, z) where z is not nv,
                    # adding (nv, u)=c does not change common[c][u][z].
                    
                    # Wait, the "at-risk" is about future vertices.
                    # The heuristic is: minimize the number of pairs (u, w) such that common[c][u][w] becomes 1.
                    # Initially common[c][u][w] is 0.
                    # If we assign c to (nv, u), and later assign c to (nv, w),
                    # then nv becomes a common neighbor.
                    # So we want to avoid assigning c to many u's that are "close" to each other.
                    
                    # Simple score: Sum of common[c][u][w] for all w?
                    # If common[c][u][w] is high, u and w are already risky.
                    
                    score = 0
                    for w in range(n-1):
                        if w == u: continue
                        # If w is already assigned color c, we already checked for conflict.
                        # If w is not assigned c, we look at existing common count.
                        # If common[c][u][w] >= 1, then u and w are already at risk.
                        score += common[c][u][w]
                    
                    if score < best_score:
                        best_score = score
                        best_c = c
                        
                if best_c == -1:
                    return None # Failed
                
                assigned_colors[u] = best_c
                cm[new_v-1][u] = best_c
                cm[u][new_v-1] = best_c
                
                # Update common counts?
                # Adding (nv, u) with color best_c.
                # For any w, nv is a common neighbor of u and w in color best_c IF (w, nv) is best_c.
                # We don't know (w, nv) yet for unassigned w.
                # So we cannot update common counts accurately until all edges from nv are assigned.
                
            # After all edges from nv are assigned, update common counts.
            # For each pair (u, w) in 0..n-2:
            # If cm[nv-1][u] == c and cm[nv-1][w] == c, then nv is a common neighbor of u and w in color c.
            # So common[c][u][w] += 1.
            # Also common[c][w][u] += 1.
            
            for c in range(5):
                for u in range(n-1):
                    if cm[nv-1][u] != c: continue
                    for w in range(u+1, n-1):
                        if cm[nv-1][w] == c:
                            common[c][u][w] += 1
                            common[c][w][u] += 1
        
        # Convert cm to list
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(cm[i][j])
        return {'n': n, 'colors': colors}

    best_result = None
    for seed in range(100):
        res = greedy_construct(26, seed)
        if res:
            if best_result is None or res['n'] > best_result['n']:
                best_result = res
                if best_result['n'] >= 26:
                    break
        if time.time() - start_time > timeout:
            break
            
    if best_result:
        return best_result
        
    # Fallback to seed
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

if __name__ == '__main__':
    construct()
