import time
import random
from itertools import combinations

def _is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable using backtracking with unit propagation.
    Returns True if 2-colorable, False otherwise.
    """
    # Preprocess edges into sets for faster lookup?
    # For n=15, simple lists are fine.
    
    # Unit propagation and backtracking
    # State: list of colors, -1 for uncolored
    color = [-1] * n
    
    # Adjacency: for each vertex, list of edges containing it
    # This helps in propagation
    vertex_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            vertex_edges[v].append(i)
    
    def propagate(start_color, start_vertex):
        """
        Propagate constraints.
        Returns False if contradiction, True otherwise.
        Modifies 'color' in place.
        """
        stack = [(start_vertex, start_color)]
        while stack:
            v, c = stack.pop()
            if color[v] != -1:
                if color[v] != c:
                    return False
                else:
                    continue
            color[v] = c
            # Check all edges containing v
            for e_idx in vertex_edges[v]:
                e = edges[e_idx]
                # Count colors in this edge
                c0 = 0
                c1 = 0
                uncolored = []
                for u in e:
                    if color[u] == 0:
                        c0 += 1
                    elif color[u] == 1:
                        c1 += 1
                    else:
                        uncolored.append(u)
                
                # Contradiction checks
                if c0 == 5:
                    return False
                if c1 == 5:
                    return False
                
                # Propagation
                if c0 == 4 and len(uncolored) == 1:
                    # The remaining one must be 1
                    stack.append((uncolored[0], 1))
                elif c1 == 4 and len(uncolored) == 1:
                    # The remaining one must be 0
                    stack.append((uncolored[0], 0))
        return True

    def backtrack(idx):
        """
        Recursive backtracking.
        idx is the index of the vertex to assign.
        """
        if idx == n:
            return True
        
        # Try color 0
        saved_color = color[idx]
        # We need to save the whole color array? 
        # Or just undo changes?
        # For simplicity and small n, we can copy the array.
        # But copying 15 ints is cheap.
        
        # Optimization: Try the color that leads to contradiction first? No, standard order.
        
        # Try 0
        if propagate(0, idx):
            if backtrack(idx + 1):
                return True
        color[idx] = -1 # Reset for next try? 
        # Wait, propagate modifies color. We need to reset all changes made by propagate.
        # This is tricky with in-place modification.
        
        # Better approach: Copy the state before propagation.
        
    # Let's use a cleaner recursive structure with state copying.
    # n=15, max depth 15. Copying list of 15 is negligible.
    
    def solve(color_state):
        # Find first uncolored vertex
        v = -1
        for i in range(n):
            if color_state[i] == -1:
                v = i
                break
        if v == -1:
            return True # All colored, valid
        
        # Try 0
        new_state_0 = color_state[:]
        if propagate_state(new_state_0, v, 0):
            if solve(new_state_0):
                return True
        
        # Try 1
        new_state_1 = color_state[:]
        if propagate_state(new_state_1, v, 1):
            if solve(new_state_1):
                return True
        
        return False

    def propagate_state(state, v, c):
        stack = [(v, c)]
        while stack:
            cv, cc = stack.pop()
            if state[cv] != -1:
                if state[cv] != cc:
                    return False
                else:
                    continue
            state[cv] = cc
            for e_idx in vertex_edges[cv]:
                e = edges[e_idx]
                c0 = 0
                c1 = 0
                uncolored = []
                for u in e:
                    if state[u] == 0:
                        c0 += 1
                    elif state[u] == 1:
                        c1 += 1
                    else:
                        uncolored.append(u)
                
                if c0 == 5 or c1 == 5:
                    return False
                
                if c0 == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 1))
                elif c1 == 4 and len(uncolored) == 1:
                    stack.append((uncolored[0], 0))
        return True

    initial_state = [-1] * n
    return solve(initial_state)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    n = 15
    all_edges = list(combinations(range(n), 5))
    
    best_result = None
    best_m = 127 # Seed is 126, so anything < 127 is better
    
    # We want to minimize M.
    # Start from M=49 (known minimum) and go up.
    # If we find one, we return it.
    # To be safe, we can try a few M values.
    
    # Heuristic: Try M=49, 50, 51...
    # If we fail to find one quickly, increase M.
    
    for m in range(49, 127):
        if time.time() > deadline:
            break
            
        # Try a few random instances for this M
        # Number of tries depends on time remaining
        # Let's try 100 instances per M
        found = False
        for _ in range(100):
            if time.time() > deadline:
                break
            random.shuffle(all_edges)
            edges = [list(e) for e in all_edges[:m]]
            
            if not _is_2_colorable(n, edges):
                return {'n': n, 'edges': edges}
        
        # If we didn't find one in 100 tries, move to next M
        # This ensures we don't get stuck on M=49 if it's rare
        # But we should keep track of the best found so far?
        # No, we return the first one found.
        # Since we iterate M from low to high, the first one found is the smallest M.
        
    # If we didn't find any (unlikely), return the seed
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
