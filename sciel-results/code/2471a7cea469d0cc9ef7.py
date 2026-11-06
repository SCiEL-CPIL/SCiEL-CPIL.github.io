import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115 seconds safety margin
    
    # Fallback solution n=11 (from seed)
    n_fallback = 11
    colors_fallback = [min(j - i, n_fallback - j + i) - 1 for i in range(n_fallback) for j in range(i + 1, n_fallback)]
    best_n = n_fallback
    best_colors = colors_fallback[:]

    def solve_csp(n, time_limit):
        """
        Solve for n using Backtracking with Forward Checking.
        We assign colors to edges in a specific order.
        To prune effectively, we track for each color c, and for each pair of vertices (u, v),
        the number of common neighbors they already have in color c.
        If this count reaches 2, we have a C4 in color c, which is forbidden.
        
        Actually, it's more efficient to track:
        For each color c, and each vertex v, the set of neighbors N_c(v).
        When assigning edge (u, v) color c:
        We must check if u and v have >= 2 common neighbors in c.
        Common neighbors = N_c(u) ∩ N_c(v).
        
        To make this fast:
        We maintain an array common_counts[c][u][v] = number of common neighbors of u and v in color c.
        When we add edge (u, v) with color c:
        For every w in N_c(u) ∩ N_c(v), the count for pair (w, ?) doesn't change directly in a simple way?
        
        Let's refine the data structure.
        For a fixed color c:
        When we add edge (u, v):
        1. Check if |N_c(u) ∩ N_c(v)| >= 2. If yes, conflict.
        2. Update structures:
           For each w in N_c(u):
               v becomes a new common neighbor of w and ...? No.
               w is a neighbor of u. u is now a neighbor of v.
               So w and v have u as a common neighbor.
               So common_count_c[w][v] += 1 and common_count_c[v][w] += 1.
           For each w in N_c(v):
               u becomes a new common neighbor of w and v? No.
               w is a neighbor of v. v is now a neighbor of u.
               So w and u have v as a common neighbor.
               So common_count_c[w][u] += 1 and common_count_c[u][w] += 1.
        
        This allows O(deg(u) + deg(v)) update and O(1) conflict check (just check common_count_c[u][v]).
        Wait, the conflict check is: do u and v have >= 2 common neighbors?
        Yes, if common_count_c[u][v] >= 2 BEFORE adding the edge?
        No, common_count_c[u][v] counts common neighbors.
        If common_count_c[u][v] >= 2, then adding edge (u,v) creates a C4?
        No, the C4 is u-w-v-x-u.
        The condition "no two vertices with two common neighbors in one color" means:
        For any distinct u, v, |N_c(u) ∩ N_c(v)| <= 1.
        
        So, before assigning color c to edge (u, v), we must ensure that
        |N_c(u) ∩ N_c(v)| <= 1.
        This is exactly common_count_c[u][v] <= 1.
        
        If common_count_c[u][v] == 1, we CAN assign color c to (u, v).
        If common_count_c[u][v] >= 2, we CANNOT assign color c to (u, v).
        
        After assigning color c to (u, v):
        We update the common counts for pairs involving u or v.
        Specifically:
        For each w in N_c(u):
            # w is connected to u. u is now connected to v.
            # So w and v share u as a common neighbor.
            common_count_c[w][v] += 1
            common_count_c[v][w] += 1
            # Check if this creates a violation for pair (w, v)?
            # If common_count_c[w][v] becomes 2, then w and v have 2 common neighbors.
            # This is a violation! But wait, the violation is only checked when we ASSIGN an edge.
            # If we create a state where w and v have 2 common neighbors, that's a C4 w-u-v-x-w.
            # This C4 exists regardless of whether the edge (w, v) is colored or not.
            # The problem statement says: "No colour class may contain a four-cycle".
            # A four-cycle in a graph is a cycle of length 4.
            # If w, u, v, x are vertices, and edges (w,u), (u,v), (v,x), (x,w) exist in color c, that's a C4.
            # The condition "no two vertices with two common neighbors" is equivalent to "no C4" in a simple graph.
            # So if we create a situation where w and v have 2 common neighbors (say u and x),
            # then the cycle w-u-v-x-w is a C4 in color c.
            # This is a violation.
            # So, when we add edge (u, v) with color c, we must ensure that for all w in N_c(u),
            # the pair (w, v) does NOT already have another common neighbor.
            # i.e., common_count_c[w][v] must be 0 BEFORE we increment it.
            # If common_count_c[w][v] is 1, incrementing it to 2 creates a C4.
            # So the check is: for all w in N_c(u) ∪ N_c(v), common_count_c[w][other] must be 0?
            # Let's be precise.
            # Adding (u, v) with color c.
            # For any w in N_c(u):
            #   w and v now share u.
            #   If w and v already shared another neighbor x (i.e., common_count_c[w][v] >= 1),
            #   then after adding (u,v), common_count_c[w][v] becomes >= 2.
            #   This forms a C4 w-u-v-x-w.
            #   So we MUST have common_count_c[w][v] == 0 for all w in N_c(u).
            # Similarly, for any w in N_c(v):
            #   w and u now share v.
            #   We MUST have common_count_c[w][u] == 0 for all w in N_c(v).
            
        This is a very strong constraint! It means that when adding an edge (u, v) in color c,
        the neighborhoods N_c(u) and N_c(v) must be "disjoint" in terms of their connections to each other?
        No, it means:
        1. N_c(u) and N_c(v) must be disjoint? No.
           If w is in N_c(u) ∩ N_c(v), then w is a common neighbor.
           The condition "common_count_c[w][v] == 0 for all w in N_c(u)" implies:
           If w is in N_c(u), then w and v have no common neighbors.
           In particular, if u is a neighbor of v, then w and v share u.
           So common_count_c[w][v] must be 0.
           But if w is in N_c(u) ∩ N_c(v), then w and v share u AND w.
           So common_count_c[w][v] >= 1 (because of u).
           So we CANNOT have any w in N_c(u) ∩ N_c(v).
           Therefore, N_c(u) and N_c(v) must be DISJOINT.
           
        Let's double check this logic.
        If N_c(u) and N_c(v) are disjoint, then u and v have 0 common neighbors.
        Condition: |N_c(u) ∩ N_c(v)| <= 1.
        So they can share AT MOST 1 common neighbor.
        
        Let's re-evaluate the update step.
        Suppose we add (u, v) with color c.
        Let S_u = N_c(u) and S_v = N_c(v).
        We require |S_u ∩ S_v| <= 1.
        
        After adding (u, v):
        New N_c(u) = S_u ∪ {v}
        New N_c(v) = S_v ∪ {u}
        
        For any w in S_u:
        w is connected to u. u is connected to v.
        So w and v share u.
        Do w and v share any other neighbor?
        If w is in S_v, then w is connected to v. So w and v share w? No, w is a vertex, not a neighbor of itself.
        Wait, "common neighbor" means a vertex x such that (w,x) and (v,x) are edges.
        If w in S_v, then (w,v) is an edge.
        Is w a common neighbor of w and v? No.
        But if w in S_u ∩ S_v, then (w,u) and (w,v) are edges.
        So u and v share w.
        
        Let's look at the pair (w, v) where w in S_u.
        Common neighbors of w and v in the NEW graph:
        - u is a common neighbor (since (w,u) and (u,v) exist).
        - Are there any others?
        - If x is a common neighbor, then (w,x) and (v,x) exist.
        - Before adding (u,v), did w and v have common neighbors?
          Let K = N_c_old(w) ∩ N_c_old(v).
          After adding (u,v), the new common neighbors are K ∪ {u} (if u was not in K).
          We require |K ∪ {u}| <= 1.
          Since u is in S_u, and w is in S_u, (w,u) exists.
          Is u in K? K = N_c_old(w) ∩ N_c_old(v).
          u is in N_c_old(w) (since w in S_u).
          Is u in N_c_old(v)?
          If u in N_c_old(v), then (u,v) already existed? No, we are adding it.
          So u is NOT in N_c_old(v).
          Therefore u is NOT in K.
          So new common neighbors = K ∪ {u}.
          Size = |K| + 1.
          We require |K| + 1 <= 1 => |K| = 0.
          So N_c_old(w) ∩ N_c_old(v) must be empty.
          i.e., w and v had NO common neighbors before adding (u,v).
        
        This must hold for ALL w in S_u.
        Similarly, for ALL w in S_v, N_c_old(w) ∩ N_c_old(u) must be empty.
        
        Also, we need to check the pair (u, v) itself.
        Common neighbors of u and v in new graph:
        S_u ∩ S_v.
        We require |S_u ∩ S_v| <= 1.
        
        So the checks for adding (u, v) with color c are:
        1. |N_c(u) ∩ N_c(v)| <= 1.
        2. For all w in N_c(u), N_c(w) ∩ N_c(v) is empty.
        3. For all w in N_c(v), N_c(w) ∩ N_c(u) is empty.
        
        This is a very strong local constraint. It should prune the search space drastically.
        
        Data Structures:
        - colors: list of length n*(n-1)/2, initialized to -1.
        - adj: list of 5 sets, adj[c][v] = set of neighbors of v in color c.
        - common: list of 5 dictionaries or 2D arrays?
          common[c][u][v] = number of common neighbors of u and v in color c.
          Since n <= 29, we can use a 3D list or a list of 2D arrays.
          common = [ [[0]*n for _ in range(n)] for _ in range(5) ]
          
        When adding edge (u, v) with color c:
        1. Check |adj[c][u] ∩ adj[c][v]| <= 1.
           If not, return False.
        2. For w in adj[c][u]:
           If common[c][w][v] > 0: return False.
        3. For w in adj[c][v]:
           If common[c][w][u] > 0: return False.
           
        If all checks pass:
        1. colors[idx] = c
        2. adj[c][u].add(v)
        3. adj[c][v].add(u)
        4. Update common counts:
           # u and v now have common neighbors S_u ∩ S_v.
           # But we already checked that.
           # We need to update common[c][x][y] for pairs affected.
           
           # The addition of (u, v) increases common neighbor counts for:
           # - Pairs (w, v) for w in adj[c][u]: common[c][w][v] += 1
           # - Pairs (w, u) for w in adj[c][v]: common[c][w][u] += 1
           
           for w in adj[c][u]:
               common[c][w][v] += 1
               common[c][v][w] += 1
           for w in adj[c][v]:
               common[c][w][u] += 1
               common[c][u][w] += 1
               
        This is O(deg(u) + deg(v)).
        
        Backtracking:
        We need to undo these changes.
        We can store the changes in a stack or just reverse the operations.
        Since we are doing DFS, we can just reverse the loops.
        
        Order of edges:
        We need to assign colors to edges in a specific order.
        A good order is to process edges in a way that constrains the graph early.
        Maybe process edges (i, j) where i is small?
        Or use a dynamic order?
        For simplicity, let's use a fixed order: lexicographic (i, j).
        However, dynamic ordering (MRV) is usually better for CSP.
        Let's try MRV: pick the edge with the fewest available colors.
        An edge (u, v) has color c available if the checks pass.
        Checking availability for all 5 colors is O(n).
        So MRV is O(n * E) per step, which is O(n^3).
        For n=29, E=406, n^3 ~ 24k. 406 * 24k ~ 10M ops per step. Too slow for Python.
        
        Let's stick to a fixed order but a smart one.
        Maybe process edges from a "center" vertex outwards?
        Or just lexicographic.
        
        Let's try lexicographic first.
        
        Randomization:
        To avoid getting stuck in the same local minima, we can randomize the color order.
        For each edge, try colors in a random order.
        
        We will implement a recursive backtracking solver.
        To prevent stack overflow and allow time checking, we'll use a stack-based iterative DFS.
        But Python recursion is easier. n=29, depth=406. Python default recursion limit is 1000.
        We can increase it or use iterative.
        Let's use iterative DFS with an explicit stack.
        
        State:
        - current edge index
        - colors array
        - adj sets
        - common counts
        
        Since we need to backtrack, we need to save the state at each step.
        Saving the whole common array (5 * 29 * 29 = 4205 ints) at each of 406 steps is 1.7M ints.
        This is acceptable memory-wise.
        But copying is slow.
        
        Alternative: Log the changes.
        When we make a move, log:
        - edge_idx, color
        - list of (w, v, delta) for common updates
        Then to backtrack, reverse the log.
        
        Let's implement the iterative DFS with change logging.
        
        """
        num_edges = n * (n - 1) // 2
        if num_edges == 0:
            return []
            
        # Edge list
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                
        # Map (i, j) to index
        edge_idx = {}
        for idx, (i, j) in enumerate(edges):
            edge_idx[(i, j)] = idx
            edge_idx[(j, i)] = idx
            
        # State
        colors = [-1] * num_edges
        adj = [set() for _ in range(5)]
        # common[c][u][v]
        common = [ [[0]*n for _ in range(n)] for _ in range(5) ]
        
        # Stack for DFS
        # Each item: (edge_pos, color_to_try, changes_log)
        # We start at edge 0, trying color 0.
        
        # To speed up, we can use a fixed order of edges but randomize color order.
        # Or randomize edge order?
        # Let's randomize edge order slightly? No, fixed order is better for caching?
        # Let's try lexicographic edge order, but random color order.
        
        # We will use a recursive function with sys.setrecursionlimit
        # It's simpler and likely fast enough if pruning is good.
        
        sys.setrecursionlimit(10000)
        
        def backtrack(pos):
            if time.time() > time_limit:
                return False
                
            if pos == num_edges:
                return True
                
            u, v = edges[pos]
            idx = pos
            
            # Try colors in random order
            colors_to_try = [0, 1, 2, 3, 4]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Check if color c is valid for edge (u, v)
                valid = True
                
                # 1. Check |adj[c][u] ∩ adj[c][v]| <= 1
                # Since sets, intersection is fast for small sets.
                # deg is at most n-1=28.
                inter = adj[c][u] & adj[c][v]
                if len(inter) > 1:
                    valid = False
                
                if valid:
                    # 2. Check common neighbors for w in adj[c][u]
                    for w in adj[c][u]:
                        if common[c][w][v] > 0:
                            valid = False
                            break
                            
                if valid:
                    # 3. Check common neighbors for w in adj[c][v]
                    for w in adj[c][v]:
                        if common[c][w][u] > 0:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Apply move
                colors[idx] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                changes = []
                for w in adj[c][u]:
                    common[c][w][v] += 1
                    common[c][v][w] += 1
                    changes.append((w, v, 1))
                    changes.append((v, w, 1))
                    
                for w in adj[c][v]:
                    common[c][w][u] += 1
                    common[c][u][w] += 1
                    changes.append((w, u, 1))
                    changes.append((u, w, 1))
                    
                # Recurse
                if backtrack(pos + 1):
                    return True
                    
                # Undo move
                for w, x, delta in changes:
                    common[c][w][x] -= delta
                    common[c][x][w] -= delta
                
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                colors[idx] = -1
                
            return False
            
        # Run backtracking
        # We might need multiple restarts if it fails quickly?
        # Or just run once with random color shuffling.
        
        if backtrack(0):
            return colors
        else:
            return None

    # Try to find solutions for larger n
    # We start from a known good n and go up?
    # Or start from high n and go down?
    # The task says "Maximize n".
    # Known best is 26.
    # Let's try 29, 28, 27, 26...
    # If we find a valid one, return it.
    # If we time out, return the best found (which might be the fallback).
    
    # However, CSP might be very fast for small n and very slow for large n.
    # Let's allocate time slices.
    
    # We'll try n=12, 13, ..., 29.
    # If we find a valid coloring for n, we update best_n and best_colors.
    # We continue to try larger n if time permits.
    
    # To ensure we don't spend all time on n=12 (which is easy), we should give less time to smaller n?
    # Or just let it run. If n=12 is solved in 0.1s, we move to 13.
    
    for n in range(12, 30):
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this n
        # Give it the remaining time minus a safety margin for the next few n's?
        # Or just a fixed slice?
        # Let's give it a slice that grows with n?
        # Or just try it with the remaining time.
        # If it fails, we move to the next n.
        
        # We'll give each n a chance to solve within the remaining time.
        # But if n=12 takes 10s, we might not have time for 29.
        # n=12 should be very fast.
        
        # Let's try to solve n with a time limit of min(remaining_time, 5 seconds)
        # If it solves, great. If not, we try the next n.
        # This ensures we explore larger n's.
        
        time_limit = min(deadline - 2, time.time() + 5)
        
        # CSP solver
        result = solve_csp(n, time_limit)
        
        if result is not None:
            best_n = n
            best_colors = result[:]
            # We found a valid coloring for n.
            # Should we continue to n+1?
            # Yes, we want to maximize n.
            # But we need to be careful with time.
            # If we found n=26, we should try n=27 with remaining time.
            
            # If we found a large n, we might want to spend more time on it?
            # No, we just want to find ANY valid coloring for n.
            # Once found, we move to n+1.
            
        else:
            # Failed to find coloring for n within time_limit.
            # We can try again with more time?
            # Or move to n+1?
            # If n is hard, n+1 is likely harder.
            # So if we fail for n, we should probably stop trying larger n's?
            # Not necessarily. Maybe n=15 is hard but n=16 is easy? Unlikely.
            # Usually, if you can't solve n, you can't solve n+1.
            # So we should break?
            # But what if n=12 is hard due to bad random seed, but n=13 is easy?
            # Let's try a few more if we fail.
            # Or just break.
            # Given the monotonicity of difficulty, breaking is safer.
            # But we might have failed due to time limit, not impossibility.
            # Let's try the next n with a smaller time limit?
            # No, let's just break. If we can't solve n, we likely can't solve n+1.
            # However, the fallback is n=11.
            # If we fail for n=12, we return n=11.
            
            # Wait, the current program tries n=12..26 sequentially.
            # If it fails for 12, it tries 13?
            # In the current program, `solve_n_sa` is called for each n.
            # If it fails, it moves to the next n.
            # This is bad if 12 is hard.
            
            # For CSP, if it fails, it's likely a dead end.
            # Let's break.
            break

    return {'n': best_n, 'colors': best_colors}

# Test locally
if __name__ == "__main__":
    res = construct()
    print(res['n'])
