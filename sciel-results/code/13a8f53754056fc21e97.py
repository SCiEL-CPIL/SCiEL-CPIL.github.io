import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try to find a solution for n=26, then 25, etc.
    # We know 26 is possible.
    for n in range(26, 10, -1):
        if time.time() > deadline:
            break
        colors = solve_n(n, deadline)
        if colors is not None:
            # Verify solution just in case
            if verify(n, colors):
                return {'n': n, 'colors': colors}
            
    # Fallback to seed solution
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def verify(n, colors):
    # colors is a list of length n*(n-1)//2
    # Map index to (i,j)
    # idx = i*n - (i+1)*i//2 + (j-i-1) ? No, lexicographic.
    # i=0: 0,1 ... 0,n-1
    # i=1: 1,2 ... 1,n-1
    # Index calculation:
    # For i, start index is sum_{k=0}^{i-1} (n-k) = i*n - i*(i-1)//2
    # offset is j - i - 1
    
    # Build adjacency matrix for each color
    adj = [ [[False]*n for _ in range(n)] for _ in range(5) ]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            adj[c][i][j] = True
            adj[c][j][i] = True
            idx += 1
            
    # Check for C4 in each color
    for c in range(5):
        # A C4 exists if there are u,v,w,z such that 
        # adj[u][v], adj[v][w], adj[w][z], adj[z][u] are all True
        # And u!=v!=w!=z!=u
        # Also diagonals u-w and v-z can be anything.
        # We can check by iterating pairs of edges.
        # Or simpler: for each pair of vertices (u,v), find common neighbors.
        # If two common neighbors w, z exist, and w!=z, then u-v-w-z-u is a C4?
        # No, u and v are connected. w and z are connected to u and v.
        # The cycle is u-w-v-z-u.
        # Edges: (u,w), (w,v), (v,z), (z,u).
        # All must be in color c.
        # So for each color c:
        #   For each pair (u,v) such that adj[c][u][v] is True:
        #      Find common neighbors of u and v in color c.
        #      If there are at least 2 common neighbors w, z, then we have a C4.
        #      Wait, if w and z are common neighbors, then:
        #      u-w, w-v, v-z, z-u are all edges in color c.
        #      This forms a C4 u-w-v-z-u.
        #      Yes.
        
        for u in range(n):
            for v in range(u+1, n):
                if adj[c][u][v]:
                    common = []
                    for k in range(n):
                        if k != u and k != v:
                            if adj[c][u][k] and adj[c][v][k]:
                                common.append(k)
                    if len(common) >= 2:
                        return False
    return True

def solve_n(n, deadline):
    # Total edges
    m = n * (n - 1) // 2
    # colors array, -1 for unassigned
    colors = [-1] * m
    
    # Helper to get index
    def get_idx(i, j):
        if i > j: i, j = j, i
        # Sum of edges for 0..i-1
        # 0: n-1
        # 1: n-2
        # ...
        # i: n-1-i
        # Sum = i*(n-1) - i*(i-1)//2
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    # Symmetry breaking
    # Fix x[0,1]=0, x[0,2]=1, x[0,3]=2, x[0,4]=3
    # This is valid if n >= 5
    if n >= 5:
        colors[get_idx(0, 1)] = 0
        colors[get_idx(0, 2)] = 1
        colors[get_idx(0, 3)] = 2
        colors[get_idx(0, 4)] = 3
        
    # List of unassigned edges
    unassigned = [k for k in range(m) if colors[k] == -1]
    
    # Randomize order for restarts?
    # Let's use a deterministic order first, but maybe shuffle if stuck.
    
    def check_conflict(idx, color_val):
        # Check if assigning colors[idx] = color_val causes a C4
        # Get i, j from idx
        # Reverse of get_idx
        # This is tricky to invert directly. 
        # Better to iterate i,j in the main loop and pass i,j.
        pass
        
    # Refactored solver with i,j tracking
    
    # To make it faster, we precompute the neighbors for each edge?
    # No, dynamic check is O(n^2).
    
    def is_consistent(i, j, c):
        # Check if setting colors[i,j] = c is consistent with current assignments
        # Only need to check 4-sets containing (i,j)
        # 4-set {i,j,w,z}
        # Cycles involving (i,j):
        # 1. i-j-w-z-i : edges (i,j), (j,w), (w,z), (z,i)
        # 2. i-j-z-w-i : edges (i,j), (j,z), (z,w), (w,i)
        
        for w in range(n):
            if w == i or w == j: continue
            for z in range(w+1, n):
                if z == i or z == j: continue
                
                # Check cycle 1: i-j-w-z-i
                # Edges: (i,j) [being set to c], (j,w), (w,z), (z,i)
                # We need to check if (j,w), (w,z), (z,i) are all c.
                # If any is unassigned, no conflict yet.
                # If any is assigned to different color, no conflict for this specific cycle.
                # If all are c, conflict.
                
                idx_jw = get_idx(j, w)
                idx_wz = get_idx(w, z)
                idx_zi = get_idx(z, i)
                
                c_jw = colors[idx_jw]
                c_wz = colors[idx_wz]
                c_zi = colors[idx_zi]
                
                if c_jw == c and c_wz == c and c_zi == c:
                    return False
                # If one is unassigned, it might be set to c later, but for now, 
                # we only check if a conflict is *already formed* or *forced*.
                # Standard backtracking: we only check if the current partial assignment 
                # already contains a C4. 
                # If c_jw, c_wz, c_zi are all c, we have a C4.
                # If one is -1, we don't know yet.
                
                # Check cycle 2: i-j-z-w-i
                # Edges: (i,j) [c], (j,z), (z,w), (w,i)
                idx_jz = get_idx(j, z)
                idx_zw = get_idx(z, w)
                idx_wi = get_idx(w, i)
                
                c_jz = colors[idx_jz]
                c_zw = colors[idx_zw]
                c_wi = colors[idx_wi]
                
                if c_jz == c and c_zw == c and c_wi == c:
                    return False
                    
        return True

    # Backtracking function
    # depth is number of assigned edges? No, we pick from unassigned.
    # We'll use a recursive function with a stack or just recursion.
    # Recursion depth m=325. Python limit is ~1000. OK.
    
    def backtrack(unassigned_list):
        if not unassigned_list:
            return True
            
        if time.time() > deadline:
            return False
            
        # Heuristic: Pick the first edge? 
        # Better: Pick an edge that is "most constrained".
        # For simplicity, pick the first one in the list.
        # But we can shuffle unassigned_list at each level? No, that's expensive.
        # Let's just pick the first one.
        
        edge_idx = unassigned_list[0]
        rest = unassigned_list[1:]
        
        # Recover i, j from edge_idx
        # This is inefficient to do every time.
        # Let's store (i, j, idx) in the list.
        
        return False # Placeholder

    # Let's rewrite using a list of tuples (i, j, idx)
    
    def solve():
        unassigned = []
        for i in range(n):
            for j in range(i+1, n):
                idx = get_idx(i, j)
                if colors[idx] == -1:
                    unassigned.append((i, j, idx))
                    
        # Shuffle to break symmetry further?
        # random.shuffle(unassigned)
        
        def bt(unassigned):
            if not unassigned:
                return True
            if time.time() > deadline:
                return False
                
            # Pick a variable
            # Heuristic: Most Constrained Variable (MCV)
            # Count unassigned neighbors in 4-sets?
            # Too complex. Let's try static order or random.
            # Let's try picking the first one.
            
            i, j, idx = unassigned[0]
            rest = unassigned[1:]
            
            # Value ordering: 0, 1, 2, 3, 4
            for c in range(5):
                if is_consistent(i, j, c):
                    colors[idx] = c
                    if bt(rest):
                        return True
                    colors[idx] = -1
                    
            return False
            
        return bt(unassigned)
        
    # Run multiple restarts with different random shuffles if first fails
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Reset colors except fixed ones
        fixed = []
        if n >= 5:
            fixed = [
                (0, 1, get_idx(0,1), 0),
                (0, 2, get_idx(0,2), 1),
                (0, 3, get_idx(0,3), 2),
                (0, 4, get_idx(0,4), 3)
            ]
            
        colors = [-1] * m
        for i, j, idx, c in fixed:
            colors[idx] = c
            
        unassigned = []
        for i in range(n):
            for j in range(i+1, n):
                idx = get_idx(i, j)
                if colors[idx] == -1:
                    unassigned.append((i, j, idx))
                    
        if attempt > 0:
            random.shuffle(unassigned)
            
        # Call bt
        # We need to pass colors and unassigned to bt
        # Let's wrap bt in a closure that uses the local colors
        
        # Re-defining bt inside loop is slow. 
        # Let's just make one solver function that takes colors and unassigned.
        
        def bt(unassigned, colors):
            if not unassigned:
                return True
            if time.time() > deadline:
                return False
                
            i, j, idx = unassigned[0]
            rest = unassigned[1:]
            
            for c in range(5):
                if is_consistent(i, j, c, colors):
                    colors[idx] = c
                    if bt(rest, colors):
                        return True
                    colors[idx] = -1
                    
            return False
            
        # Modify is_consistent to take colors as arg
        def is_consistent(i, j, c, colors):
            for w in range(n):
                if w == i or w == j: continue
                for z in range(w+1, n):
                    if z == i or z == j: continue
                    
                    idx_jw = get_idx(j, w)
                    idx_wz = get_idx(w, z)
                    idx_zi = get_idx(z, i)
                    
                    if colors[idx_jw] == c and colors[idx_wz] == c and colors[idx_zi] == c:
                        return False
                    
                    idx_jz = get_idx(j, z)
                    idx_zw = get_idx(z, w)
                    idx_wi = get_idx(w, i)
                    
                    if colors[idx_jz] == c and colors[idx_zw] == c and colors[idx_wi] == c:
                        return False
            return True

        if bt(unassigned, colors):
            return colors
            
    return None

# Optimization: The above is_consistent is O(n^2) per call.
# With n=26, n^2=676.
# Depth 325.
# 325 * 676 * 5 (values) * branching factor.
# It might be too slow if branching factor is high.
# But symmetry breaking helps.

# Let's add a simple propagation: 
# If an edge is forced to a value, assign it.
# This is complex. Let's stick to the simple backtracking with restarts.

# One more check: The seed solution is n=11.
# If we fail to find 26, we return 11.
# This guarantees score 1.0.
# If we find 26, score is 26/11 = 2.36.

# Let's refine the code to be robust.
