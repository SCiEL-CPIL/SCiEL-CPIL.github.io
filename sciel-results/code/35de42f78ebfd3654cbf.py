"""Construct 5-colourings of K_n without monochromatic C4, maximising n."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0

    def full_check(n, colors):
        """Verify no colour class has a C4."""
        adj = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][c].add(j)
                adj[j][c].add(i)
                idx += 1
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if len(adj[i][c] & adj[j][c]) >= 2:
                        return False
        return True

    def geometric_coloring(n):
        """
        Deterministic geometric construction based on finite affine plane AG(2,5).
        Vertices 0..24 are mapped to points in GF(5)^2.
        Edge (i,j) is colored by the slope of the line through them.
        Slopes: 0, 1, 2, 3, 4 (and vertical is handled by mapping).
        Actually, in AG(2,5), there are 6 directions (slopes): 0, 1, 2, 3, 4, and infinity.
        We have 5 colors. We need to map 6 slopes to 5 colors?
        No, the Lazebnik-Woldar construction uses a different structure.
        
        Let's use the standard C4-free coloring for K_25.
        A known C4-free coloring of K_25 with 5 colors:
        Map vertices to (x,y) in GF(5)^2.
        Color of edge ((x1,y1), (x2,y2)) is the slope m = (y2-y1)/(x2-x1) mod 5.
        If x1 == x2, slope is infinity.
        There are 5 finite slopes and 1 infinite slope. Total 6 directions.
        We have 5 colors. This direct mapping gives 6 colors.
        
        However, we can merge two slopes? No, that would create C4s.
        
        Alternative: Use the construction from Lazebnik & Woldar.
        They use a group-based construction.
        
        Let's try a simpler approach:
        For n=25, we can use the following known valid coloring:
        Vertices are elements of Z_5 x Z_5.
        Color of edge (u, v) where u != v is determined by the direction.
        Since there are 6 directions, and we need 5 colors, this specific mapping doesn't work directly with 5 colors for all 25 vertices unless we use a different group or structure.
        
        Wait, the bound R_5(C4) >= 26 is known.
        The construction for 26 is more complex.
        
        Let's implement a robust local search that starts from a good random seed or a specific heuristic.
        The previous program used simulated annealing. Let's improve the local search with a better cost function and faster updates.
        
        We will use a bitset representation for adjacency to speed up common neighbor counting.
        """
        pass

    def local_search(n, max_time=10.0):
        """
        Find a valid 5-coloring of K_n using local search with bitset optimization.
        """
        total_edges = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Initialize colors randomly
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Build adjacency bitsets for each color
        # adj[c][i] is a bitset of neighbors of i in color c
        adj = [[0] * n for _ in range(5)]
        
        for idx, (i, j) in enumerate(edges):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
        
        # Precompute common neighbor counts? 
        # Instead, we calculate cost on the fly or maintain a count array.
        # Let's maintain a count array common[c][i][j] for i < j.
        # To save memory and time, we only store for i < j.
        # common[c][i][j] = popcount(adj[c][i] & adj[c][j])
        
        # Calculating full common matrix is O(5 * n^2 * (n/word_size))
        # For n=26, this is very fast.
        
        def compute_common():
            common = [[[0]*n for _ in range(n)] for _ in range(5)]
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        common[c][i][j] = bin(adj[c][i] & adj[c][j]).count('1')
                        common[c][j][i] = common[c][i][j]
            return common

        common = compute_common()
        
        def calc_cost():
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i+1, n):
                        k = common[c][i][j]
                        if k >= 2:
                            cost += k * (k - 1) // 2
            return cost

        current_cost = calc_cost()
        best_cost = current_cost
        best_colors = colors[:]
        
        temperature = 5.0
        cooling = 0.9995
        min_temp = 0.1
        
        # Local search loop
        while time.time() < start + max_time and temperature > min_temp:
            idx = random.randint(0, total_edges - 1)
            i, j = edges[idx]
            current_color = colors[idx]
            new_color = random.randint(0, 4)
            if new_color == current_color:
                continue
            
            # Calculate delta cost
            delta = 0
            
            # Remove edge (i,j) from current_color
            oc = current_color
            # Affected pairs: (i,k) for k in adj[oc][i], (j,k) for k in adj[oc][j]
            # We need to update common counts for these pairs.
            
            # To be efficient, we identify the affected pairs.
            # Pairs involving i: (i,k) where k is neighbor of i in oc.
            # Pairs involving j: (j,k) where k is neighbor of j in oc.
            
            # Let's collect affected pairs
            affected = set()
            # Neighbors of i in oc
            mask_i = adj[oc][i]
            k = 0
            while mask_i:
                if mask_i & 1:
                    if k != j:
                        affected.add((min(i,k), max(i,k)))
                mask_i >>= 1
                k += 1
            
            # Neighbors of j in oc
            mask_j = adj[oc][j]
            k = 0
            while mask_j:
                if mask_j & 1:
                    if k != i:
                        affected.add((min(j,k), max(j,k)))
                mask_j >>= 1
                k += 1
            
            # Add (i,j) itself? common[i][j] doesn't change when edge (i,j) is removed?
            # common[i][j] counts m such that (i,m) and (j,m) exist.
            # Removing (i,j) doesn't remove any m. So common[i][j] is unchanged.
            # So we don't need to include (i,j) in affected for cost calculation of oc?
            # Wait, if i and j become disconnected, does it affect other pairs?
            # No, only pairs where i or j is one of the endpoints are affected in terms of common neighbor counts.
            
            cost_before_oc = 0
            for u, v in affected:
                k_val = common[oc][u][v]
                if k_val >= 2:
                    cost_before_oc += k_val * (k_val - 1) // 2
            
            # Update adj and common for oc
            # Remove j from adj[oc][i] and i from adj[oc][j]
            adj[oc][i] &= ~(1 << j)
            adj[oc][j] &= ~(1 << i)
            
            # Update common counts
            # For each k in old adj[oc][i] (excluding j):
            #   common[oc][i][k] -= 1 (because j is no longer a common neighbor of i and k? No.)
            #   Wait. common[oc][i][k] is count of m such that (i,m) and (k,m) exist.
            #   If we remove (i,j), then j is no longer a neighbor of i.
            #   So if k is a neighbor of j, then j was a common neighbor of i and k.
            #   So common[oc][i][k] decreases by 1 for each k in adj[oc][j] (old).
            
            # Let's redo the update logic carefully.
            # Old state: (i,j) exists in oc.
            # New state: (i,j) removed from oc.
            
            # For any k:
            # If (j,k) exists in oc (old), then i and k shared j. Now they don't.
            # So common[oc][i][k] -= 1.
            # If (i,k) exists in oc (old), then j and k shared i. Now they don't.
            # So common[oc][j][k] -= 1.
            
            # We need the OLD adjacency lists to know which k are affected.
            # We already have mask_i and mask_j from above (before removal).
            
            # Iterate k in mask_i (old neighbors of i in oc)
            mask_i_old = adj[oc][i] | (1 << j) # Restore j for iteration
            k = 0
            while mask_i_old:
                if mask_i_old & 1:
                    if k != j:
                        # j and k shared i. Now they don't.
                        common[oc][j][k] -= 1
                        common[oc][k][j] -= 1
                mask_i_old >>= 1
                k += 1
                
            mask_j_old = adj[oc][j] | (1 << i)
            k = 0
            while mask_j_old:
                if mask_j_old & 1:
                    if k != i:
                        # i and k shared j. Now they don't.
                        common[oc][i][k] -= 1
                        common[oc][k][i] -= 1
                mask_j_old >>= 1
                k += 1
            
            cost_after_oc = 0
            for u, v in affected:
                k_val = common[oc][u][v]
                if k_val >= 2:
                    cost_after_oc += k_val * (k_val - 1) // 2
            delta += cost_after_oc - cost_before_oc
            
            # Add edge (i,j) to new_color
            nc = new_color
            affected_nc = set()
            
            # Neighbors of i in nc
            mask_i_nc = adj[nc][i]
            k = 0
            while mask_i_nc:
                if mask_i_nc & 1:
                    if k != j:
                        affected_nc.add((min(i,k), max(i,k)))
                mask_i_nc >>= 1
                k += 1
                
            # Neighbors of j in nc
            mask_j_nc = adj[nc][j]
            k = 0
            while mask_j_nc:
                if mask_j_nc & 1:
                    if k != i:
                        affected_nc.add((min(j,k), max(j,k)))
                mask_j_nc >>= 1
                k += 1
            
            cost_before_nc = 0
            for u, v in affected_nc:
                k_val = common[nc][u][v]
                if k_val >= 2:
                    cost_before_nc += k_val * (k_val - 1) // 2
            
            # Update adj and common for nc
            adj[nc][i] |= (1 << j)
            adj[nc][j] |= (1 << i)
            
            # For each k in old adj[nc][i]:
            # i and k gain j as common neighbor?
            # If (j,k) exists in nc, then i and k now share j.
            # So common[nc][i][k] += 1 for each k in adj[nc][j] (old).
            
            mask_j_nc_old = adj[nc][j] & ~(1 << i) # Exclude i
            k = 0
            while mask_j_nc_old:
                if mask_j_nc_old & 1:
                    common[nc][i][k] += 1
                    common[nc][k][i] += 1
                mask_j_nc_old >>= 1
                k += 1
                
            mask_i_nc_old = adj[nc][i] & ~(1 << j)
            k = 0
            while mask_i_nc_old:
                if mask_i_nc_old & 1:
                    common[nc][j][k] += 1
                    common[nc][k][j] += 1
                mask_i_nc_old >>= 1
                k += 1
            
            cost_after_nc = 0
            for u, v in affected_nc:
                k_val = common[nc][u][v]
                if k_val >= 2:
                    cost_after_nc += k_val * (k_val - 1) // 2
            delta += cost_after_nc - cost_before_nc
            
            # Update colors
            colors[idx] = new_color
            
            # Acceptance
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            else:
                # Undo
                # This is complex. It's safer to keep a copy of the state or use a stack.
                # Given the time constraint, let's just restart if we get stuck?
                # Or, simpler: don't undo, but this breaks the SA logic.
                # Let's implement a simple undo by storing the previous state of affected variables.
                # This is getting too complex for the time limit.
                # Alternative: Use a simpler heuristic like first-improvement hill climbing.
                # If delta > 0, reject and try another edge.
                # We need to revert the changes.
                
                # Revert oc
                adj[oc][i] |= (1 << j)
                adj[oc][j] |= (1 << i)
                
                # Revert common for oc
                mask_j_old = adj[oc][j] & ~(1 << i)
                k = 0
                while mask_j_old:
                    if mask_j_old & 1:
                        common[oc][i][k] += 1
                        common[oc][k][i] += 1
                    mask_j_old >>= 1
                    k += 1
                    
                mask_i_old = adj[oc][i] & ~(1 << j)
                k = 0
                while mask_i_old:
                    if mask_i_old & 1:
                        common[oc][j][k] += 1
                        common[oc][k][j] += 1
                    mask_i_old >>= 1
                    k += 1
                    
                # Revert nc
                adj[nc][i] &= ~(1 << j)
                adj[nc][j] &= ~(1 << i)
                
                mask_j_nc_old = adj[nc][j] | (1 << i)
                k = 0
                while mask_j_nc_old:
                    if mask_j_nc_old & 1:
                        if k != i:
                            common[nc][i][k] -= 1
                            common[nc][k][i] -= 1
                    mask_j_nc_old >>= 1
                    k += 1
                    
                mask_i_nc_old = adj[nc][i] | (1 << j)
                k = 0
                while mask_i_nc_old:
                    if mask_i_nc_old & 1:
                        if k != j:
                            common[nc][j][k] -= 1
                            common[nc][k][j] -= 1
                    mask_i_nc_old >>= 1
                    k += 1
                    
                colors[idx] = current_color
            
            temperature *= cooling
            
        if best_cost == 0:
            return best_colors
        return None

    # Try to find valid coloring for increasing n
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # We will try n=26 first, then 25, etc.
    # To increase the chance of finding n=26, we run multiple restarts for n=26.
    
    for n_try in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 10:
            break
        
        # For n=26, give it more time and multiple restarts
        if n_try == 26:
            restarts = 5
            time_per_restart = 15.0
        elif n_try == 25:
            restarts = 3
            time_per_restart = 10.0
        else:
            restarts = 2
            time_per_restart = 5.0
            
        found = False
        for r in range(restarts):
            if time.time() > deadline - 5:
                break
            res = local_search(n_try, max_time=time_per_restart)
            if res is not None:
                if full_check(n_try, res):
                    best_n = n_try
                    best_colors = res
                    found = True
                    break
        
        if found:
            break

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
