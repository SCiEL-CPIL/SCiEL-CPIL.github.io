"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_mono_k222_fast(adj_masks, n, color):
    """
    Fast check for K2,2,2 in the subgraph defined by adj_masks (list of ints, bit i set if edge to i exists).
    adj_masks[i] contains bits for neighbors of i in the specified color.
    """
    # For each pair {a, b}, let common_ab = adj_masks[a] & adj_masks[b].
    # We need to find a K2,2 in the subgraph induced by common_ab.
    # A K2,2 exists in a graph if there are two vertices x, y that share at least 2 common neighbors.
    # So for each pair {a, b}:
    #   1. Get list of vertices in common_ab.
    #   2. For each pair {x, y} in that list:
    #      Check if (adj_masks[x] & adj_masks[y] & common_ab) has at least 2 bits set.
    #      If yes, we found {a,b}, {x,y}, {u,v}.
    #      We must ensure {a,b}, {x,y}, {u,v} are disjoint.
    #      Since x,y are in common_ab, and we assume adj_masks don't have self-loops,
    #      a and b are not in common_ab (unless a=b which is not possible).
    #      So {a,b} is disjoint from {x,y,u,v}.
    #      We just need to ensure {x,y} and {u,v} are disjoint?
    #      u,v are common neighbors of x,y. So u,v are in common_ab.
    #      Are u,v distinct from x,y? Yes, no self-loops.
    #      Are u,v distinct from each other? Yes, we check for >= 2 bits.
    #      So the 6 vertices are a,b,x,y,u,v.
    #      Are they all distinct?
    #      a,b distinct.
    #      x,y distinct.
    #      u,v distinct.
    #      x,y not in {a,b} because x,y in common_ab implies x,y are neighbors of a and b.
    #      If x=a, then a is neighbor of a, impossible.
    #      So {a,b} disjoint from {x,y,u,v}.
    #      Thus we have 6 distinct vertices.
    
    for a in range(n):
        mask_a = adj_masks[a]
        for b in range(a + 1, n):
            # If a and b are not connected in this color, they can't form a part?
            # Wait, the definition of K2,2,2 subgraph does NOT require the internal edges of the parts to exist.
            # It only requires the 12 cross edges.
            # So {a,b} can be a part even if (a,b) is not color C.
            # However, for {a,b} to be a part, all edges from {a,b} to {x,y} and {u,v} must be color C.
            # This means x,y,u,v must be in adj_masks[a] AND adj_masks[b].
            # So common_ab = mask_a & adj_masks[b].
            
            common_ab = mask_a & adj_masks[b]
            
            # If common_ab has fewer than 4 vertices, we can't form a K2,2 (which needs 4 vertices x,y,u,v).
            # Actually, K2,2 needs 4 vertices.
            # So if bit_count(common_ab) < 4, skip.
            if common_ab.bit_count() < 4:
                continue
            
            # Iterate through pairs {x, y} in common_ab
            # To be efficient, we can just check if any pair in common_ab has >= 2 common neighbors in common_ab.
            
            # Let's extract the list of vertices in common_ab
            # Since n <= 31, we can iterate
            verts = []
            temp_mask = common_ab
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                verts.append(v)
                temp_mask ^= lsb
            
            # Check all pairs in verts
            for i in range(len(verts)):
                x = verts[i]
                mask_x = adj_masks[x]
                for j in range(i + 1, len(verts)):
                    y = verts[j]
                    # Common neighbors of x and y within common_ab
                    common_xy = mask_x & adj_masks[y] & common_ab
                    if common_xy.bit_count() >= 2:
                        return True
    return False

def _is_valid_fast(colors, n):
    """Check if neither color class contains K2,2,2 using fast bitmask method."""
    # Build adj masks for color 0 and 1
    adj0 = [0] * n
    adj1 = [0] * n
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
            
    if _has_mono_k222_fast(adj0, n, 0):
        return False
    if _has_mono_k222_fast(adj1, n, 1):
        return False
    return True

def _backtrack_search(max_n, deadline):
    """
    Backtracking search to find the maximum n such that K_n can be colored
    without monochromatic K2,2,2.
    """
    # We build the graph vertex by vertex.
    # colors[i][j] for i < j.
    # We store colors in a flat list or 2D list.
    # For speed, we maintain adj masks incrementally.
    
    best_n = 0
    best_colors = []
    
    # Recursive backtracking
    def backtrack(n, adj0, adj1, colors_list):
        nonlocal best_n, best_colors
        if time.time() > deadline:
            return False
        
        if n > best_n:
            best_n = n
            best_colors = list(colors_list)
            
        if n == max_n:
            return True # Found max_n
            
        # Add new vertex n
        # We need to assign colors to edges (n, j) for j in 0..n-1.
        # There are 2^n possibilities.
        # Use symmetry breaking: fix edge (n, 0) to 0?
        # Or just try 0 and 1.
        
        new_adj0 = [0] * (n + 1)
        new_adj1 = [0] * (n + 1)
        
        # Copy existing adj masks
        for i in range(n):
            new_adj0[i] = adj0[i]
            new_adj1[i] = adj1[i]
        
        # Try all 2^n assignments? No, that's too slow for n=15 (32768).
        # But we can prune.
        # For each assignment, check if it creates a K2,2,2.
        
        # To speed up, we can iterate through assignments and check validity incrementally?
        # Or just check the full validity of the new vertex addition.
        
        # A K2,2,2 involving vertex n must have n in one of the 3 parts.
        # This is complex to check incrementally.
        # Given n is small (<= 20), checking full validity is O(n^2 * n^2) ~ 10^4-10^5 ops.
        # 2^15 * 10^5 = 10^9. Too slow.
        
        # We need a smarter search.
        # Greedy with backtracking.
        # Try color 0 for (n, n-1), then 1.
        # If dead end, backtrack.
        
        # For simplicity and robustness within time limits, we will only go up to n=12 or 15.
        # If we can't find a valid coloring for n, we return the best found.
        
        # Let's try a simple DFS with symmetry breaking.
        # Fix colors of (n, 0), (n, 1) to break symmetry?
        # Actually, just try 0 then 1 for each edge.
        
        # Optimization: Check if the new vertex creates a K2,2,2.
        # Instead of full check, we can check only subgraphs involving n.
        # But _has_mono_k222_fast is fast enough for n<20 if called sparingly.
        
        # Let's try all 2^n assignments for small n.
        # For n=10, 2^10=1024. 1024 * fast_check(~10^3) = 10^6. OK.
        # For n=15, 2^15=32768. 32768 * 10^4 = 3*10^8. Might be tight.
        
        # We will limit the search to n=15.
        
        if n > 15:
            return False
            
        # Iterate through all 2^n color assignments for the new vertex
        for mask in range(1 << n):
            # Construct new adj masks
            valid = True
            new_adj0_local = new_adj0[:]
            new_adj1_local = new_adj1[:]
            
            # Assign colors
            for j in range(n):
                if (mask >> j) & 1:
                    # Color 1
                    new_adj1_local[n] |= (1 << j)
                    new_adj1_local[j] |= (1 << n)
                else:
                    # Color 0
                    new_adj0_local[n] |= (1 << j)
                    new_adj0_local[j] |= (1 << n)
                    
            # Check validity
            # Only need to check if K2,2,2 is formed.
            # Since the previous graph was valid, only new K2,2,2s involving n can appear.
            # We can optimize the check to only look at K2,2,2s involving n.
            # But _has_mono_k222_fast checks all.
            # For n=15, it's fast enough.
            
            if _has_mono_k222_fast(new_adj0_local, n + 1, 0) or _has_mono_k222_fast(new_adj1_local, n + 1, 1):
                continue
                
            # Update colors_list
            # colors_list is a flat list of length n*(n-1)/2
            # We need to append n new colors for edges (n, 0)...(n, n-1)
            # Lexicographic order: (0,1), (0,2)...(0,n-1), (1,2)...
            # The new edges are (0,n), (1,n), ..., (n-1, n).
            # In the flat list, these are appended at the end?
            # No, lexicographic order of (i,j) with i<j.
            # (0, n) comes before (1, n)?
            # Order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2, n-1), (0,n)? No.
            # (0,n) has i=0, j=n.
            # (1,2) has i=1, j=2.
            # (0,n) comes before (1,2) if n > 2?
            # Yes, 0 < 1.
            # So the list is not simply "append".
            # We need to insert the new colors in the correct positions.
            # This is inefficient.
            
            # Better representation: 2D array or list of lists.
            # Or, just return the mask and reconstruct at the end?
            # No, we need to pass state to deeper recursion.
            
            # Let's change representation to a 2D list of colors, or just keep the masks.
            # The output requires a flat list. We can convert at the end.
            
            # For now, let's just say we found a valid step.
            # We recurse.
            # But we need to pass the masks.
            
            # To avoid copying lists in Python (slow), we can pass the masks and update in place?
            # No, backtracking requires undo.
            
            # Given the time limit, let's simplify:
            # We will only try to find a valid coloring for n=10 or 12.
            # This is guaranteed to be fast.
            
            # If we find a valid assignment for n, we recurse.
            # We store the color of (n, j) in a temporary list.
            
            new_colors_edges = [1 if (mask >> j) & 1 else 0 for j in range(n)]
            
            # Recurse
            # We need to pass the new masks.
            # We'll make a copy.
            
            if backtrack(n + 1, new_adj0_local, new_adj1_local, colors_list + new_colors_edges):
                return True
                
        return False
        
    # Start with n=1 (no edges)
    # Actually, start with n=2, edge (0,1) color 0 (symmetry)
    # Or just start with n=0.
    
    # Let's start with n=1.
    # adj0, adj1 are empty lists.
    # colors_list is empty.
    
    # To save time, we will stop if we reach n=15.
    # And we will break if time runs out.
    
    backtrack(1, [0], [0], [])
    
    return best_n, best_colors

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try to find a valid coloring using backtracking
    n_found, colors_found = _backtrack_search(15, deadline)
    
    if n_found > 0 and colors_found:
        # Reconstruct the flat list in lexicographic order
        # colors_found is a list of edges added at each step?
        # No, in the backtrack function, `colors_list + new_colors_edges` appends the edges for the new vertex.
        # But the lexicographic order of the final graph K_n is:
        # (0,1), (0,2), ..., (0,n-1), (1,2), ..., (1,n-1), ..., (n-2, n-1)
        
        # The `colors_list` we built is:
        # Step 1 (n=1 to 2): edge (0,1). List: [c(0,1)]
        # Step 2 (n=2 to 3): edges (0,2), (1,2). List: [c(0,1), c(0,2), c(1,2)]
        # Step 3 (n=3 to 4): edges (0,3), (1,3), (2,3). List: [c(0,1), c(0,2), c(1,2), c(0,3), c(1,3), c(2,3)]
        
        # The required lexicographic order for n=4 is:
        # (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        
        # Our list is:
        # [ (0,1), (0,2), (1,2), (0,3), (1,3), (2,3) ]
        
        # This is NOT lexicographic order.
        # We need to reorder the colors.
        
        # Let's map the indices.
        # For a given n, the total number of edges is n*(n-1)/2.
        # We have a list `colors_found` of length n*(n-1)/2.
        # But the order is "by vertex addition".
        
        # We can reconstruct the color of any edge (i,j) from `colors_found`?
        # When vertex j was added (j > i), the edge (i,j) was the i-th element in the `new_colors_edges` for step j.
        # The offset for step j in `colors_found` is:
        # Sum of lengths of new_colors_edges for steps 1 to j-1.
        # Length of step k is k.
        # Sum_{k=1}^{j-1} k = (j-1)j/2.
        
        # So color(i,j) for i<j is at index:
        # offset = (j-1)j/2 + i
        
        # Let's verify for n=4.
        # Edge (0,1): j=1, i=0. Offset = 0*1/2 + 0 = 0. Index 0. Correct.
        # Edge (0,2): j=2, i=0. Offset = 1*2/2 + 0 = 1. Index 1. Correct.
        # Edge (1,2): j=2, i=1. Offset = 1*2/2 + 1 = 2. Index 2. Correct.
        # Edge (0,3): j=3, i=0. Offset = 2*3/2 + 0 = 3. Index 3. Correct.
        # Edge (1,3): j=3, i=1. Offset = 2*3/2 + 1 = 4. Index 4. Correct.
        # Edge (2,3): j=3, i=2. Offset = 2*3/2 + 2 = 5. Index 5. Correct.
        
        # So we can map `colors_found` to the lexicographic order.
        
        n = n_found
        total_edges = n * (n - 1) // 2
        final_colors = [0] * total_edges
        
        for i in range(n):
            for j in range(i + 1, n):
                # Find index in colors_found
                # j is the "step" where this edge was added?
                # Yes, edge (i,j) is added when vertex j is introduced.
                # So step = j.
                # Offset in colors_found for step j is (j-1)j/2.
                # Index in that step is i.
                idx_in_found = (j - 1) * j // 2 + i
                final_colors[i * (2 * n - i - 1) // 2 + (j - i - 1)] = colors_found[idx_in_found]
                # Wait, the target index in lexicographic order for (i,j):
                # Number of edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*n - i*(i+1)/2
                # In row i, j is the (j-i-1)-th edge.
                # Target idx = i*n - i*(i+1)/2 + j - i - 1
                
        return {'n': n, 'colors': final_colors}
        
    # Fallback: return a trivial valid coloring for n=1 or n=2
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
