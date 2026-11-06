import time
import random
import math

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
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
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _index_to_edge(n, idx):
    """Convert lexicographic edge index to (i, j) pair."""
    i = 0
    remaining = idx
    while i < n - 1:
        edges_from_i = n - 1 - i
        if remaining < edges_from_i:
            j = i + 1 + remaining
            return (i, j)
        remaining -= edges_from_i
        i += 1
    return (n - 2, n - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i]
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                # Only count triangles where i < j < k to avoid double counting
                # But bitset approach counts all common neighbors.
                # We need to be careful. 
                # Standard trick: sum over edges (u,v) of common neighbors w > v? 
                # Or just count all and divide by 3? No, each triangle is counted 3 times (once per vertex).
                # Actually, for each edge (i,j) in color c, the number of triangles containing it is popcount(adj[c][i] & adj[c][j]).
                # Total triangles = sum_{c} sum_{i<j, color(i,j)=c} popcount(adj[c][i] & adj[c][j]) / 3 ?
                # No, each triangle {u,v,w} has 3 edges. It is counted once for edge (u,v), once for (u,w), once for (v,w).
                # So summing over all edges counts each triangle 3 times.
                # However, iterating over all pairs (i,j) is O(n^2).
                # The current loop iterates i, then j in neighbors.
                # Let's stick to a simpler, slightly slower but correct count for small n, 
                # or optimize the count.
                # For n=50, n^2 is 2500. 4 colors. 
                # Let's just count by checking all triples? No, O(n^3) is too slow.
                # Let's use the edge-based count.
                pass
    
    # Re-implementing count efficiently
    count = 0
    for c in range(4):
        for i in range(n):
            # Iterate j > i
            # We can just sum popcount(adj[c][i] & adj[c][j]) for all i<j and divide by 3?
            # Let's do:
            for j in range(i+1, n):
                if adj[c][i] & (1 << j): # if edge (i,j) exists in color c
                    common = adj[c][i] & adj[c][j]
                    count += common.bit_count()
    return count // 3

def _find_one_mono_triangle(adj, n):
    """Find one monochromatic triangle, return (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            for j in range(i+1, n):
                if adj[c][i] & (1 << j):
                    common = adj[c][i] & adj[c][j]
                    if common:
                        k_bit = common & (-common)
                        k = k_bit.bit_length() - 1
                        return (i, j, k, c)
    return None

def _local_search(colors, n, time_budget):
    """
    Local search to minimize monochromatic triangles.
    Returns (colors, tri_count).
    """
    m = n * (n - 1) // 2
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    
    if tri_count == 0:
        return colors, 0

    start_time = time.time()
    T = 2.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute edge indices for faster access? 
    # We need to map (u,v) to index.
    # index = u*(n-1) - u*(u+1)//2 + (v-u-1)
    
    while time.time() - start_time < time_budget and _time_left() > 5.0:
        if tri_count == 0:
            break
            
        # Find a random edge involved in a triangle, or just random edge
        # Random edge is simpler
        ri = random.randint(0, m - 1)
        u, v = _index_to_edge(n, ri)
        old_c = colors[ri]
        
        # Find a neighbor w such that (u,v,w) is a mono triangle in old_c
        # Or just pick a random new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Calculate delta
        # Triangles removed: edges (u,v) in old_c. Triangles containing (u,v) in old_c are counted in tri_count.
        # When we change (u,v) to new_c:
        # We remove all triangles of color old_c containing (u,v).
        # We add all triangles of color new_c containing (u,v).
        
        # Number of triangles of color c containing edge (u,v) is popcount(adj[c][u] & adj[c][v]).
        # Note: This counts triangles where w is any neighbor.
        
        rem = (adj[old_c][u] & adj[old_c][v]).bit_count()
        add = (adj[new_c][u] & adj[new_c][v]).bit_count()
        
        # Wait, the triangle count formula `count // 3` implies we are summing over edges.
        # So `rem` is the contribution of edge (u,v) to the sum.
        # `add` is the new contribution.
        # Delta = add - rem.
        
        delta = add - rem
        
        if delta <= 0 or (T > T_min and random.random() < math.exp(-delta / T)):
            # Accept
            colors[ri] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            tri_count += delta
            
        T *= cooling

    return colors, tri_count

def construct():
    """
    Construct a 4-coloring of K_n without monochromatic triangles.
    """
    if _time_left() <= 0:
        return _seed_32()

    # Start with the known valid n=32 coloring
    seed_res = _seed_32()
    n = 32
    colors = seed_res['colors']
    best_n = n
    best_colors = colors[:]

    # Try to extend to n=50
    for target_n in range(33, 51):
        if _time_left() < 5.0:
            break
            
        # Extend colors list to target_n
        # New edges are (i, target_n-1) for i in 0..target_n-2
        # And edges among new vertices? No, we add one vertex at a time.
        
        new_vertex = target_n - 1
        new_edges_count = new_vertex # edges (0, nv), (1, nv), ..., (nv-1, nv)
        
        # Initialize new edge colors greedily/randomly
        # We need to insert them at the correct positions in the lexicographic list.
        # Lexicographic order: 
        # (0,1), (0,2), ..., (0, n-1)
        # (1,2), ..., (1, n-1)
        # ...
        # (n-2, n-1)
        
        # When adding vertex v = n-1 to a graph of size n-1:
        # The new edges are (0, v), (1, v), ..., (v-1, v).
        # In the list for size n, these edges are interspersed.
        # Specifically, edge (i, v) for i < v.
        # The index of (i, v) in K_n is:
        # i * (n - 1) - i * (i + 1) // 2 + (v - i - 1)
        # Since v = n - 1, v - i - 1 = n - 1 - i - 1 = n - i - 2.
        
        # It is easier to rebuild the list.
        new_colors = [0] * (target_n * (target_n - 1) // 2)
        
        # Copy old colors
        # For each pair (i, j) in old graph (0 <= i < j < n):
        # If j < new_vertex, it stays at the same relative position?
        # No, the indices change.
        
        # Let's just generate the new list from scratch based on the old one.
        # We have a function to get color of (i,j) from the list?
        # Or we can store the coloring as a 2D matrix internally for efficiency during extension.
        
        # Let's switch to a 2D matrix representation for the search phase.
        # colors_mat[i][j] = color
        # Then convert to list at the end.
        
        if not hasattr(construct, '_mat'):
            # Initialize matrix from seed list
            mat = [[0]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx]
                    mat[i][j] = c
                    mat[j][i] = c
                    idx += 1
            construct._mat = mat
            construct._n = n
        else:
            mat = construct._mat
            # Resize matrix
            new_mat = [[0]*target_n for _ in range(target_n)]
            for i in range(n):
                for j in range(n):
                    new_mat[i][j] = mat[i][j]
            construct._mat = new_mat
            mat = new_mat
            construct._n = target_n
            
        # Assign colors to new edges (i, new_vertex)
        for i in range(new_vertex):
            # Greedy choice: pick color that creates fewest new mono triangles
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                # Count triangles formed with (i, new_vertex) of color c
                # Triangle (i, new_vertex, k) is mono if color(i, new_vertex)=c, color(i,k)=c, color(new_vertex,k)=c.
                # We haven't set color(new_vertex, k) for k < i yet? 
                # We are iterating i from 0 to new_vertex-1.
                # When we set color(i, new_vertex), the edges (k, new_vertex) for k > i are not set yet.
                # Edges (k, new_vertex) for k < i are already set.
                # So we can only check triangles (i, new_vertex, k) where k < i.
                # This is a sequential dependency.
                
                cost = 0
                for k in range(i):
                    if mat[i][k] == c and mat[k][new_vertex] == c:
                        cost += 1
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            mat[i][new_vertex] = best_c
            mat[new_vertex][i] = best_c
            
        # Now we have a coloring for target_n.
        # Convert to list for local search? 
        # Or run local search on the matrix?
        # Local search on matrix is faster for updates.
        
        # Let's implement a local search step on the matrix.
        # We need to count triangles.
        # For speed, we'll do a few iterations of local search on the matrix.
        
        # Function to count triangles in matrix
        def count_tris_mat(mat, size):
            count = 0
            for c in range(4):
                for i in range(size):
                    for j in range(i+1, size):
                        if mat[i][j] == c:
                            # Count k > j such that mat[i][k]==c and mat[j][k]==c
                            # This is O(n^3). Too slow for n=50? 50^3 = 125k. 4 colors. 500k ops. 
                            # In Python, 500k ops is ~0.05s. 
                            # If we do this in a loop, it might be slow.
                            # Let's use bitsets.
                            pass
            # Let's stick to the list-based SA which is proven to work in the inspiration, 
            # but we need to convert mat to list.
            
        colors_list = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                colors_list.append(mat[i][j])
                
        # Run local search with a small time budget
        # We want to quickly check if we can fix it.
        time_budget = min(2.0, _time_left() / 10.0)
        new_colors_list, tri_count = _local_search(colors_list, target_n, time_budget)
        
        if tri_count == 0:
            # Success!
            best_n = target_n
            best_colors = new_colors_list[:]
            # Update matrix for next iteration
            mat = [[0]*target_n for _ in range(target_n)]
            idx = 0
            for i in range(target_n):
                for j in range(i+1, target_n):
                    c = new_colors_list[idx]
                    mat[i][j] = c
                    mat[j][i] = c
                    idx += 1
            construct._mat = mat
            construct._n = target_n
            colors = new_colors_list[:]
            n = target_n
        else:
            # Failed to fix. 
            # We can try a few more times with different random seeds or longer SA?
            # Or just stop extending?
            # If we can't extend from 33, we probably can't reach 50 with this method.
            # But maybe 34 works?
            # Let's try a few restarts for this target_n.
            found = False
            for restart in range(5):
                if _time_left() < 2.0:
                    break
                # Re-randomize new edges
                for i in range(new_vertex):
                    mat[i][new_vertex] = random.randint(0, 3)
                    mat[new_vertex][i] = mat[i][new_vertex]
                
                colors_list = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        colors_list.append(mat[i][j])
                        
                new_colors_list, tri_count = _local_search(colors_list, target_n, 1.0)
                if tri_count == 0:
                    found = True
                    break
                    
            if found:
                best_n = target_n
                best_colors = new_colors_list[:]
                mat = [[0]*target_n for _ in range(target_n)]
                idx = 0
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        c = new_colors_list[idx]
                        mat[i][j] = c
                        mat[j][i] = c
                        idx += 1
                construct._mat = mat
                construct._n = target_n
                colors = new_colors_list[:]
                n = target_n
            else:
                # If we fail at n=33, it's unlikely to work for higher n with this simple extension.
                # We should stop extending and return the best we have.
                # Or try a completely different initialization for n=50?
                # Given the time, returning n=32 or the best found is safer.
                # But let's try one big SA for n=50 directly if we have time?
                # No, the prompt says "Combine strategy... take structural idea of one and refinement of other".
                # The structural idea is the seed's n=32. The refinement is the SA.
                # If we can't extend, we return the seed.
                break

    return {'n': best_n, 'colors': best_colors}

# Ensure no state leakage between calls if construct is called multiple times
# Although the task implies a single call.
# But to be safe, we can reset the attribute.
# The task says "Keep the function names and signatures".
# It doesn't forbid attributes.

def construct():
    # Reset state
    if hasattr(construct, '_mat'):
        del construct._mat
    return _construct_impl()

def _construct_impl():
    if _time_left() <= 0:
        return _seed_32()

    # Start with the known valid n=32 coloring
    seed_res = _seed_32()
    n = 32
    colors = seed_res['colors']
    best_n = n
    best_colors = colors[:]

    # Try to extend to n=50
    for target_n in range(33, 51):
        if _time_left() < 5.0:
            break
            
        # Extend colors list to target_n
        new_vertex = target_n - 1
        
        # Initialize matrix from current best
        if not hasattr(construct, '_mat'):
            mat = [[0]*n for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx]
                    mat[i][j] = c
                    mat[j][i] = c
                    idx += 1
            construct._mat = mat
        else:
            mat = construct._mat
            # Resize matrix
            new_mat = [[0]*target_n for _ in range(target_n)]
            for i in range(n):
                for j in range(n):
                    new_mat[i][j] = mat[i][j]
            construct._mat = new_mat
            mat = new_mat
            
        # Assign colors to new edges (i, new_vertex)
        for i in range(new_vertex):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                cost = 0
                for k in range(i):
                    if mat[i][k] == c and mat[k][new_vertex] == c:
                        cost += 1
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            mat[i][new_vertex] = best_c
            mat[new_vertex][i] = best_c
            
        # Convert to list for local search
        colors_list = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                colors_list.append(mat[i][j])
                
        # Run local search with a small time budget
        time_budget = min(2.0, _time_left() / 10.0)
        new_colors_list, tri_count = _local_search(colors_list, target_n, time_budget)
        
        if tri_count == 0:
            best_n = target_n
            best_colors = new_colors_list[:]
            # Update matrix
            idx = 0
            for i in range(target_n):
                for j in range(i+1, target_n):
                    c = new_colors_list[idx]
                    mat[i][j] = c
                    mat[j][i] = c
                    idx += 1
            n = target_n
            colors = new_colors_list[:]
        else:
            # Try a few restarts
            found = False
            for restart in range(5):
                if _time_left() < 2.0:
                    break
                for i in range(new_vertex):
                    mat[i][new_vertex] = random.randint(0, 3)
                    mat[new_vertex][i] = mat[i][new_vertex]
                
                colors_list = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        colors_list.append(mat[i][j])
                        
                new_colors_list, tri_count = _local_search(colors_list, target_n, 1.0)
                if tri_count == 0:
                    found = True
                    break
                    
            if found:
                best_n = target_n
                best_colors = new_colors_list[:]
                idx = 0
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        c = new_colors_list[idx]
                        mat[i][j] = c
                        mat[j][i] = c
                        idx += 1
                n = target_n
                colors = new_colors_list[:]
            else:
                break

    return {'n': best_n, 'colors': best_colors}

# Re-define construct to use the impl
def construct():
    if hasattr(construct, '_mat'):
        del construct._mat
    return _construct_impl()

