import time
import random

def construct():
    """
    Constructs a 4-coloring of K_n without monochromatic triangles.
    Tries to maximize n, targeting n=50.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for 120s limit
    
    # We will try to solve for n=50 down to n=32
    # If we find a solution for 50, we return it.
    # If not, we return the best found, or fallback to 32.
    
    best_result = None
    
    # Try n=50
    # We use a backtracking search with bitmasks.
    # To increase the chance of finding a solution quickly, we use randomized restarts.
    
    def solve_n(n, time_limit):
        """
        Attempts to find a valid 4-coloring for K_n.
        Returns {'n': n, 'colors': [...]} or None.
        """
        if n <= 1:
            return {'n': n, 'colors': []}
            
        # adj[c][v] is a bitmask of vertices u such that edge (v,u) has color c
        # We build the graph vertex by vertex.
        # We maintain the state for the current vertex v being added.
        
        # To handle the large search space, we use a recursive backtracker
        # with a node limit and time check.
        
        max_nodes = 5000000 # 5 million nodes per attempt
        nodes_visited = 0
        
        # We represent the current partial coloring for the new vertex v
        # as a list of colors assigned to edges (v, u) for u in 0..v-1.
        
        # We use a global state for the fixed vertices 0..v-1
        adj = [[0] * n for _ in range(4)]
        
        def backtrack(v, current_colors):
            nonlocal nodes_visited
            if time.time() > time_limit:
                return False
            
            if v == n:
                # Success
                # Reconstruct colors list
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        # Find color
                        for c in range(4):
                            if adj[c][i] & (1 << j):
                                colors.append(c)
                                break
                # Store in nonlocal
                nonlocal result_colors
                result_colors = colors
                return True
            
            nodes_visited += 1
            if nodes_visited > max_nodes:
                return False
            
            # Assign colors to edges (v, u) for u in 0..v-1
            # We do this recursively for each u
            # To optimize, we can order u such that the most constrained are first?
            # For now, we iterate u from 0 to v-1.
            
            # We need to try all 4^v possibilities? No, we prune.
            # We iterate through possible colors for edge (v, u)
            
            # Helper to check if color c is valid for (v, u)
            def is_valid(v, u, c):
                # Check if (v, u) with color c creates a triangle
                # Triangle would be (v, u, w) where w < u (since we are building v, and u < v)
                # Actually, w can be any vertex < v.
                # If (v, w) is color c and (u, w) is color c, then triangle.
                # adj[c][v] contains w such that (v, w) is color c.
                # adj[c][u] contains w such that (u, w) is color c.
                # Intersection must be empty.
                return (adj[c][v] & adj[c][u]) == 0
            
            # We will use a recursive function to assign colors for u = 0..v-1
            def assign_u(u):
                nonlocal nodes_visited
                if time.time() > time_limit:
                    return False
                if u == v:
                    # All edges for v are assigned
                    return backtrack(v + 1, current_colors)
                
                # Try colors 0, 1, 2, 3 for edge (v, u)
                # Heuristic: Try colors in a randomized order?
                # Or fixed order? Fixed order is deterministic.
                # Let's try a random order to escape local minima in the search tree?
                # No, backtracking is exact. Random order just changes which branch is explored first.
                
                # Optimization: Check if any color is possible
                # If no color is possible for (v, u), return False
                
                # We need to know which colors are currently assigned to (v, w) for w < u
                # This is stored in adj[c][v]
                
                for c in range(4):
                    if is_valid(v, u, c):
                        # Assign color c
                        adj[c][v] |= (1 << u)
                        adj[c][u] |= (1 << v)
                        
                        if assign_u(u + 1):
                            return True
                        
                        # Unassign
                        adj[c][v] &= ~(1 << u)
                        adj[c][u] &= ~(1 << v)
                
                return False
            
            return assign_u(0)
        
        # We wrap the search in a restart loop
        # If the first attempt fails (due to node limit or time), we can restart with a different random seed?
        # But the search is deterministic. Restarting with the same logic yields the same result.
        # To make restarts useful, we can randomize the order of colors tried.
        
        # Let's modify assign_u to use a shuffled color order
        color_order = [0, 1, 2, 3]
        
        def assign_u_random(u):
            nonlocal nodes_visited
            if time.time() > time_limit:
                return False
            if u == v:
                return backtrack(v + 1, current_colors)
            
            # Shuffle colors
            random.shuffle(color_order)
            
            for c in color_order:
                if (adj[c][v] & adj[c][u]) == 0:
                    adj[c][v] |= (1 << u)
                    adj[c][u] |= (1 << v)
                    
                    if assign_u_random(u + 1):
                        return True
                    
                    adj[c][v] &= ~(1 << u)
                    adj[c][u] &= ~(1 << v)
            
            return False
        
        # We need to reset adj and nodes_visited for each restart
        # But adj is modified in place. We need to deep copy or reset.
        # Since n is small (50), resetting is cheap.
        
        result_colors = None
        
        # Try a few restarts
        for attempt in range(10):
            # Reset state
            adj = [[0] * n for _ in range(4)]
            nodes_visited = 0
            result_colors = None
            
            # Redefine backtrack to use the current adj
            # Python closures capture variables by reference, so we need to be careful.
            # It's better to pass adj as an argument or use a class.
            
            # Let's refactor to a class or pass state explicitly to avoid closure issues in recursive calls
            # Actually, the nested function `backtrack` uses `adj` from the outer scope.
            # If we reassign `adj` in the loop, the closure will see the new `adj`.
            
            # However, `backtrack` calls `assign_u_random` which calls `backtrack`.
            # This recursion depth is n (50). Python recursion limit is 1000. OK.
            
            # We need to ensure `is_valid` and `assign_u_random` use the correct `adj`.
            
            # Let's just run the search.
            
            # To avoid recursion overhead, we can use an iterative stack, but recursion is fine for n=50.
            
            # One issue: `nodes_visited` is reset, but the recursion might hit the time limit.
            
            if backtrack(0, []):
                return {'n': n, 'colors': result_colors}
                
            # If we hit node limit or time, we restart
            # But if we hit time limit, we should stop completely
            if time.time() > time_limit:
                break
        
        return None

    # Try n=50
    res = solve_n(50, deadline)
    if res:
        return res
        
    # If 50 failed, try 32 (known valid)
    # The seed program has a valid 32-vertex coloring.
    # We can just call the seed logic or return a simple valid one.
    # Let's use the seed logic for 32 to be safe.
    
    def construct_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
        
    return construct_32()
