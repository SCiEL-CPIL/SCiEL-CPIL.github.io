import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 256
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * (N + 1)
    
    # Build graph
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
    # Identify isolated vertices
    isolated = []
    active = []
    for i in range(1, N + 1):
        if adj[i] == 0:
            isolated.append(i)
        else:
            active.append(i)
            
    best_set = list(isolated)
    
    # Find connected components in active graph
    visited = [False] * (N + 1)
    components = []
    
    for i in active:
        if not visited[i]:
            # BFS
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                v = queue.pop(0)
                comp.append(v)
                # Get neighbors in active
                neighbors_mask = adj[v]
                # Iterate through bits
                while neighbors_mask:
                    lsb = neighbors_mask & -neighbors_mask
                    u = lsb.bit_length() - 1
                    neighbors_mask ^= lsb
                    if not visited[u]:
                        visited[u] = True
                        queue.append(u)
            components.append(comp)
            
    # Solve MIS for each component
    for comp in components:
        if time.time() > deadline:
            break
            
        # Create mask for this component
        comp_mask = 0
        for v in comp:
            comp_mask |= (1 << v)
            
        # State for B&B
        best_size_comp = 0
        best_set_comp = []
        path = []
        
        def color_bound(C_mask):
            # Greedy coloring bound
            # Extract vertices
            verts = []
            m = C_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
                
            if not verts:
                return 0
                
            # Sort by degree in C_mask descending
            # Degree in C_mask is (adj[v] & C_mask).bit_count()
            verts.sort(key=lambda v: (adj[v] & C_mask).bit_count(), reverse=True)
            
            colors = [0] * (N + 1)
            max_color = 0
            
            for v in verts:
                used_colors = set()
                neighbors = adj[v] & C_mask
                while neighbors:
                    lsb = neighbors & -neighbors
                    u = lsb.bit_length() - 1
                    neighbors ^= lsb
                    if colors[u] > 0:
                        used_colors.add(colors[u])
                        
                c = 1
                while c in used_colors:
                    c += 1
                colors[v] = c
                if c > max_color:
                    max_color = c
                    
            return max_color
        
        def solve(C_mask, current_size):
            nonlocal best_size_comp, best_set_comp, path, deadline
            
            if time.time() > deadline:
                return
                
            if C_mask == 0:
                if current_size > best_size_comp:
                    best_size_comp = current_size
                    best_set_comp = path[:]
                return
                
            bound = color_bound(C_mask)
            
            if current_size + bound <= best_size_comp:
                return
                
            # If bound is 1, C_mask is an independent set
            if bound == 1:
                # Add all vertices in C_mask
                verts_to_add = []
                m = C_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    verts_to_add.append(v)
                    m ^= lsb
                path.extend(verts_to_add)
                if len(path) > best_size_comp:
                    best_size_comp = len(path)
                    best_set_comp = path[:]
                path.pop(len(verts_to_add)) # Remove added vertices
                return
                
            # Extract vertices and sort by degree in C_mask
            verts = []
            m = C_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
                
            verts.sort(key=lambda v: (adj[v] & C_mask).bit_count(), reverse=True)
            
            # Branch and Bound
            # We iterate through vertices.
            # For each v, we try including it, then exclude it (by removing from C_mask).
            for v in verts:
                if time.time() > deadline:
                    return
                    
                # Include v
                path.append(v)
                new_C = C_mask & ~adj[v] & ~(1 << v)
                solve(new_C, current_size + 1)
                path.pop()
                
                # Exclude v
                C_mask = C_mask & ~(1 << v)
                
                # Optional: If C_mask becomes empty, break
                if C_mask == 0:
                    break
                    
                # Pruning check after excluding v?
                # The bound check at the start of the next iteration or recursive call handles this.
                # But we can check if current_size + color_bound(C_mask) <= best_size_comp
                # This might be expensive to call repeatedly.
                # The standard algorithm relies on the bound at the entry of solve.
                # However, inside the loop, we are effectively in a "state" where v is excluded.
                # The next iteration of the for-loop will pick a different vertex from the reduced C_mask.
                # The bound check at the start of solve() is the main pruning.
                # But wait, the `for` loop is inside `solve`.
                # The `solve` function is called for the "Include" branch.
                # The "Exclude" branch is handled by continuing the loop with reduced C_mask.
                # So the bound check at the start of `solve` is sufficient for the "Include" branches.
                # For the "Exclude" path, we just continue looping.
                # We should check if the remaining C_mask can beat the best.
                # bound_rem = color_bound(C_mask)
                # if current_size + bound_rem <= best_size_comp: break
                # This is a good pruning step.
                rem_bound = color_bound(C_mask)
                if current_size + rem_bound <= best_size_comp:
                    break
                    
        solve(comp_mask, 0)
        
        best_set.extend(best_set_comp)
        
    return best_set
