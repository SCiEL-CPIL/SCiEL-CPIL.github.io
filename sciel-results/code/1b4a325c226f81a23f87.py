import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave margin for overhead
    
    n = 18
    # Initialize with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = [[0]*n for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u][v] = 1
                    adj[v][u] = 1
    
    def is_k4_free():
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    # Check common neighbors for triangle
                    common = [k for k in range(n) if k != i and k != j and adj[i][k] and adj[j][k]]
                    # Check if common neighbors contain a triangle
                    for a in range(len(common)):
                        for b in range(a+1, len(common)):
                            if adj[common[a]][common[b]]:
                                return False
        return True

    def max_triangle_free_size():
        # Exact branch and bound
        # State: frozenset of vertices
        # If triangle-free, return len
        # Else pick a triangle, branch on removing each vertex
        
        # Cache for memoization? n=18, 2^18 states is too many. 
        # But pruning is strong.
        
        def has_triangle(vertices):
            v_list = list(vertices)
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    if adj[v_list[i]][v_list[j]]:
                        for k in range(j+1, len(v_list)):
                            if adj[v_list[i]][v_list[k]] and adj[v_list[j]][v_list[k]]:
                                return (v_list[i], v_list[j], v_list[k])
            return None

        def solve(vertices):
            if len(vertices) <= 2:
                return len(vertices)
            
            tri = has_triangle(vertices)
            if tri is None:
                return len(vertices)
            
            # Branch on removing one of the 3 vertices in the triangle
            best = 0
            for v in tri:
                new_vertices = vertices - {v}
                res = solve(new_vertices)
                if res > best:
                    best = res
            return best

        return solve(frozenset(range(n)))

    current_score = max_triangle_free_size()
    best_score = current_score
    best_adj = [row[:] for row in adj]
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        
        # Decide to add or remove
        # We can bias towards removing edges if score is high, or adding if low?
        # Let's just flip randomly.
        
        old_val = adj[u][v]
        adj[u][v] = 1 - old_val
        adj[v][u] = adj[u][v]
        
        # Check K4-free
        if old_val == 0:
            # We added an edge. Check if it creates a K4.
            # This is equivalent to checking if common neighbors of u and v contain a triangle.
            common = [k for k in range(n) if k != u and k != v and adj[u][k] and adj[v][k]]
            creates_k4 = False
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    if adj[common[i]][common[j]]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            if creates_k4:
                # Revert
                adj[u][v] = old_val
                adj[v][u] = old_val
                continue
        # If old_val == 1, we removed an edge. Always K4-free.
        
        # Calculate new score
        # Optimization: if we removed an edge, the max triangle-free size can only decrease or stay same?
        # No, removing an edge makes it *easier* to be triangle-free? 
        # Wait. G[S] is triangle-free if no triangle.
        # Removing an edge from G might destroy a triangle, making a previously non-triangle-free set triangle-free.
        # So the max size can *increase*.
        # Adding an edge creates more triangles, so max size can *decrease*.
        
        # We want to minimize the score.
        # So adding edges is generally good for reducing the score.
        # But we are constrained by K4-free.
        
        new_score = max_triangle_free_size()
        
        if new_score < best_score:
            best_score = new_score
            best_adj = [row[:] for row in adj]
        else:
            # Revert if not better? Or keep if equal?
            # Let's revert to stay in the basin? Or random walk?
            # If we revert, we do local search around the best.
            # If we keep, we do simulated annealing.
            # Given the small time, let's try to stay near the best.
            # Revert if worse.
            adj[u][v] = old_val
            adj[v][u] = old_val

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
