import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Fallback: n=11 cyclic coloring
    def get_fallback():
        n = 11
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    # Greedy Randomized Construction
    def try_greedy(n, time_limit):
        # edge_color[i][j] is -1 if unassigned, 0-4 if assigned
        edge_color = [[-1] * n for _ in range(n)]
        
        # forbidden[c][u][v] = number of w such that edge(u,w)==c and edge(v,w)==c
        # If we assign edge(u,v)=c, we need to ensure forbidden[c][u][v] < 2
        # Actually, the constraint is that NO pair of vertices can have >=2 common neighbors in color c.
        # So if forbidden[c][u][v] == 2, we cannot assign edge(u,v)=c? 
        # No, the definition of C4 in the color class: 
        # A C4 in color c exists if there are vertices a,b,c,d such that edges (a,b),(b,c),(c,d),(d,a) are all color c.
        # This is equivalent to saying that in the graph formed by edges of color c, there is no C4.
        # A graph has no C4 if and only if for every pair of vertices (u,v), they have at most 1 common neighbor in that graph.
        # So, if we assign edge(u,v)=c, we must check that u and v do not already have 2 common neighbors in color c.
        # Wait, the common neighbors of u and v in color c are vertices w such that edge(u,w)=c and edge(v,w)=c.
        # So yes, we need to maintain count_common[c][u][v].
        # If count_common[c][u][v] >= 2, we cannot assign edge(u,v)=c.
        
        # However, assigning edge(u,v)=c also affects the common neighbor counts of OTHER pairs.
        # Specifically, if we assign edge(u,v)=c:
        # For any w, if edge(u,w)=c, then v and w now have u as a common neighbor in c. So count_common[c][v][w] increases by 1.
        # If edge(v,w)=c, then u and w now have v as a common neighbor in c. So count_common[c][u][w] increases by 1.
        
        # We want to assign edges in an order. A simple order is lexicographic (i,j).
        # But maybe a random order or a specific structured order is better?
        # Let's try lexicographic first, with random color selection among valid ones.
        
        count_common = [[[0] * n for _ in range(n)] for _ in range(5)]
        
        # List of edges to assign
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # We will iterate through edges.
        # To add randomness, we can shuffle the edges? 
        # But the structure of the graph matters. 
        # Let's stick to lexicographic for determinism in the "greedy" part, 
        # but randomize the color choice.
        # Actually, a random permutation of edges might allow escaping local traps?
        # No, greedy construction is sequential. The order matters.
        # Let's try a few different edge orderings or just lexicographic with random colors.
        
        for u, v in edges:
            if time.time() > time_limit:
                return None
            
            # Find valid colors for edge (u, v)
            valid_colors = []
            for c in range(5):
                if count_common[c][u][v] < 2:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: Choose color that minimizes the increase in "danger"
            # Danger = sum of increases in count_common for pairs that become >= 2?
            # Or simply choose randomly from valid colors.
            # Let's try a simple heuristic:
            # For each valid color c, calculate the number of pairs (x, y) that will have their count_common increase.
            # Specifically, for color c:
            # Neighbors of u in c: Nu = {w | edge_color[u][w] == c}
            # Neighbors of v in c: Nv = {w | edge_color[v][w] == c}
            # Assigning (u,v)=c will:
            # 1. Increase count_common[c][v][w] for all w in Nu.
            # 2. Increase count_common[c][u][w] for all w in Nv.
            
            # We want to avoid creating new "forbidden" states (count_common == 2).
            # So we prefer colors that cause the fewest transitions from 1 to 2.
            
            best_color = -1
            min_cost = float('inf')
            
            for c in valid_colors:
                cost = 0
                # Check neighbors of u
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if edge_color[u][w] == c:
                        # count_common[c][v][w] will increase
                        if count_common[c][v][w] == 1:
                            cost += 1
                # Check neighbors of v
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if edge_color[v][w] == c:
                        # count_common[c][u][w] will increase
                        if count_common[c][u][w] == 1:
                            cost += 1
                
                # Add some randomness
                cost += random.random() * 0.1
                
                if cost < min_cost:
                    min_cost = cost
                    best_color = c
            
            # Assign best_color
            c = best_color
            edge_color[u][v] = c
            edge_color[v][u] = c
            
            # Update count_common
            for w in range(n):
                if w == u or w == v:
                    continue
                if edge_color[u][w] == c:
                    count_common[c][v][w] += 1
                    count_common[c][w][v] += 1
                if edge_color[v][w] == c:
                    count_common[c][u][w] += 1
                    count_common[c][w][u] += 1
        
        # If we reached here, we have a valid coloring
        res_colors = []
        for i in range(n):
            for j in range(i + 1, n):
                res_colors.append(edge_color[i][j])
        return res_colors

    # Try n from 29 down to 11
    # Give more time to larger n? 
    # n=29 has ~400 edges. Greedy is O(n^3) per edge? No, O(n) per edge update. Total O(n^3).
    # 29^3 is ~24k operations. Very fast.
    # We can try multiple random restarts for each n.
    
    for n in range(29, 10, -1):
        if time.time() > deadline:
            break
        
        # Try a few restarts for this n
        # Time per restart: (deadline - time.time()) / 10 ?
        # Let's just try until time runs out or we find one.
        
        found = False
        while not found and time.time() < deadline:
            # Give each attempt a small time slice, e.g., 0.5s
            t_start = time.time()
            t_end = min(deadline, t_start + 0.5)
            
            result = try_greedy(n, t_end)
            if result is not None:
                return {'n': n, 'colors': result}
            
            # If we used up the slice, loop again
            # If we failed quickly, try again immediately
            
    # Fallback
    return get_fallback()
