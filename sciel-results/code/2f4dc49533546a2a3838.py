import time
import random
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    
    Strategy:
    1. Fast incremental local search using bitsets.
    2. Start with random initialization.
    3. Use O(1) cost updates for edge flips.
    4. Try n=28, 27, 26 in order.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    
    def solve_fast(n, time_limit):
        """
        Fast local search with incremental cost updates.
        """
        total_edges = n * (n - 1) // 2
        if total_edges == 0:
            return []
        
        # Precompute edge endpoints
        edge_u = [0] * total_edges
        edge_v = [0] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_u[idx] = i
                edge_v[idx] = j
                idx += 1
        
        # Precompute for each vertex and color, the list of edges incident to that vertex
        # vertex_edges[u][c] is a list of edge indices where edge_u or edge_v is u and color is c
        # Actually, for incremental updates, we need to know which edges are affected by a flip.
        # When we flip edge e=(u,v) from c_old to c_new:
        # - It affects C4 counts for pairs (u, w) and (v, w) for all w.
        # - Specifically, the common neighbors of u and v in color c change.
        
        # Data structures:
        # neighbors[c][u]: bitmask of neighbors of u in color c
        # pair_count[c][u][v]: number of common neighbors of u and v in color c.
        # But storing pair_count for all pairs is O(n^2 * 5) which is fine for n=28 (5*28*28/2 ~ 2000 ints).
        # However, updating pair_count efficiently is the key.
        
        # Let's use a different representation for speed in Python.
        # neighbors[c] is a list of n integers (bitmasks).
        # cost = sum over all c, u < v of (1 if popcount(neighbors[c][u] & neighbors[c][v]) >= 2 else 0)
        
        # Incremental update logic:
        # Flip edge e=(u,v) from c_old to c_new.
        # 1. Remove (u,v) from neighbors[c_old][u] and neighbors[c_old][v].
        #    This changes common neighbors for all pairs (u, w) and (v, w) in color c_old.
        #    Specifically, for any w != u, v:
        #    - If w is a neighbor of u in c_old (and w != v), then (u, w) share v as a common neighbor?
        #      No. The pair (u, w) has common neighbors in c_old. One of them was v (since v was neighbor of u and w? No, v is neighbor of u, but is v neighbor of w? Not necessarily).
        #      
        #      Let's re-evaluate.
        #      A C4 is defined by two vertices u, v having >= 2 common neighbors.
        #      Let S_c(u, v) = {w | (u,w) in E_c and (v,w) in E_c}.
        #      Cost contribution of pair (u,v) in color c is 1 if |S_c(u,v)| >= 2.
        #      
        #      When we remove edge (u,v) from color c_old:
        #      - It removes v from S_c_old(u, w) for all w such that (u,w) is in E_c_old? No.
        #      - It removes u from S_c_old(v, w) for all w such that (v,w) is in E_c_old? No.
        #      - It affects S_c_old(x, y) if x or y is u or v.
        #      
        #      Pairs affected in c_old:
        #      1. Pair (u, v): They were connected, but now they are not. Do they still share neighbors?
        #         The edge (u,v) itself is not a "common neighbor". A common neighbor w is a vertex such that (u,w) and (v,w) are edges.
        #         Removing edge (u,v) does NOT change S_c_old(u, v). The set of common neighbors w is unchanged.
        #         Wait, the definition of C4 in a graph usually implies a cycle of length 4.
        #         Vertices u, v, w, x. Edges (u,v), (v,w), (w,x), (x,u).
        #         This is equivalent to u and x having two common neighbors v and w.
        #         So yes, we look at pairs of vertices (u, x) and their common neighbors.
        #         The edge (u,v) being present or not does not directly change the set of common neighbors of u and x, UNLESS x=v or u=v.
        #         
        #      Let's trace carefully.
        #      Let the edge being flipped be (u, v) with color c_old.
        #      We remove this edge from color c_old.
        #      This changes the neighbor sets:
        #      N_c_old(u) = N_c_old(u) \ {v}
        #      N_c_old(v) = N_c_old(v) \ {u}
        #      
        #      Which pairs (a, b) have their common neighbor count changed?
        #      A pair (a, b) has common neighbors S(a, b) = N(a) & N(b).
        #      If a is not u and not v, N(a) is unchanged. If b is not u and not v, N(b) is unchanged.
        #      So only pairs where a in {u, v} or b in {u, v} are affected.
        #      
        #      Affected pairs in c_old:
        #      1. (u, w) for all w != u, v.
        #         S(u, w) = N(u) & N(w).
        #         New N(u) = N(u) \ {v}.
        #         So v is removed from N(u).
        #         Does v belong to S(u, w)? i.e., is v in N(w)?
        #         If (v, w) is an edge in c_old, then v was in N(u) and v was in N(w), so v was in S(u, w).
        #         Now v is not in N(u), so v is not in S(u, w).
        #         So |S(u, w)| decreases by 1 if (v, w) is in E_c_old.
        #         
        #      2. (v, w) for all w != u, v.
        #         Similarly, |S(v, w)| decreases by 1 if (u, w) is in E_c_old.
        #         
        #      3. (u, v).
        #         S(u, v) = N(u) & N(v).
        #         N(u) loses v. N(v) loses u.
        #         Did v belong to S(u, v)? i.e., is v in N(v)? No (no self loops).
        #         Did u belong to S(u, v)? i.e., is u in N(u)? No.
        #         So S(u, v) is UNCHANGED by the removal of edge (u, v).
        #         Wait, is this correct?
        #         Common neighbors of u and v are vertices w such that (u,w) and (v,w) are edges.
        #         The edge (u,v) itself is not a common neighbor.
        #         So yes, S(u,v) is unchanged.
        #      
        #      So, for c_old, we need to update cost for pairs (u, w) and (v, w) where w is a common neighbor?
        #      No, we need to update for ALL w, but the count only changes if w was a neighbor of the other endpoint.
        #      
        #      Let's formalize the delta for c_old:
        #      For each w in 0..n-1, w != u, v:
        #         if (v, w) in E_c_old:
        #            count(u, w) -= 1
        #            if count(u, w) becomes 1 (was 2), cost -= 1
        #            if count(u, w) becomes 0 (was 1), cost unchanged (was 0, now 0)
        #            Wait, cost is 1 if count >= 2.
        #            If count was 2, new count 1 -> cost decreases by 1.
        #            If count was 3, new count 2 -> cost unchanged.
        #            If count was 1, new count 0 -> cost unchanged.
        #         Similarly for (v, w) if (u, w) in E_c_old.
        #      
        #      Then we add the edge back in c_new.
        #      For each w in 0..n-1, w != u, v:
        #         if (v, w) in E_c_new:
        #            count(u, w) += 1
        #            if count(u, w) becomes 2 (was 1), cost += 1
        #            if count(u, w) becomes 3 (was 2), cost unchanged.
        #         Similarly for (v, w) if (u, w) in E_c_new.
        
        # Implementation Plan:
        # 1. Maintain neighbors[c][u] as bitmasks.
        # 2. Maintain count[c][u][v] as a 2D array (or flattened) of common neighbor counts.
        #    To save memory/time, we can just store the counts. n=28 -> 28*28*5 = 3920 ints. Negligible.
        # 3. Initialize counts by iterating all pairs.
        # 4. In the loop:
        #    - Pick random edge e=(u,v).
        #    - Pick new color c_new != c_old.
        #    - Calculate delta_cost.
        #    - If delta_cost < 0 (or SA accept), apply.
        
        # Let's implement this.
        
        # Initialize
        neighbors = [[0] * n for _ in range(5)]
        # count[c][i][j] is common neighbors of i and j in color c.
        # We only need i < j to save space, but full matrix is easier for random access.
        # Let's use a list of lists of lists? Or a flat array?
        # For n=28, a 3D list is fine.
        counts = [[[0] * n for _ in range(n)] for _ in range(5)]
        
        colors = [random.randint(0, 4) for _ in range(total_edges)]
        
        # Build initial neighbors and counts
        for e in range(total_edges):
            u = edge_u[e]
            v = edge_v[e]
            c = colors[e]
            neighbors[c][u] |= (1 << v)
            neighbors[c][v] |= (1 << u)
            
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = neighbors[c][i] & neighbors[c][j]
                    # popcount
                    cnt = bin(common).count('1')
                    counts[c][i][j] = cnt
                    counts[c][j][i] = cnt
        
        # Calculate initial cost
        cost = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    if counts[c][i][j] >= 2:
                        cost += 1
        
        if cost == 0:
            return colors
        
        best_cost = cost
        best_colors = colors[:]
        
        # Local Search Loop
        # We will use a simple hill climbing with random restarts if stuck.
        # Or simulated annealing.
        # Given the speed, simple hill climbing with random restarts is likely best.
        
        while time.time() < time_limit:
            # Pick a random edge
            e = random.randint(0, total_edges - 1)
            u = edge_u[e]
            v = edge_v[e]
            c_old = colors[e]
            
            # Pick a new color
            # We can try all 4 other colors and pick the best one.
            # This is "steepest descent" or "best improvement".
            # With O(1) update, trying all 4 is fast.
            
            best_delta = 0
            best_c_new = c_old
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                # Calculate delta for removing from c_old and adding to c_new
                delta = 0
                
                # Remove from c_old
                # Update counts for pairs (u, w) and (v, w)
                # We need to iterate w such that (v, w) is in E_c_old for (u, w) update?
                # No, we need to check if (v, w) is in E_c_old.
                # This is equivalent to checking if w is in neighbors[c_old][v].
                
                # Iterate over all w? n=28, so 28 iterations.
                # For each w, check if w is in neighbors[c_old][v].
                
                # Optimization: Only w in neighbors[c_old][v] affect count(u, w).
                # And only w in neighbors[c_old][u] affect count(v, w).
                
                # Let's iterate over set bits in neighbors[c_old][v]
                mask_v = neighbors[c_old][v]
                # Exclude u? u is not in neighbors[c_old][v] because (u,v) is the edge being removed?
                # Wait, neighbors[c_old][v] currently includes u because the edge is still there.
                # We are simulating the removal.
                # So we should exclude u from the mask?
                # Yes, because (u,v) is the edge.
                # Actually, if w=u, count(u, u) is not defined/used.
                # So we can just iterate over all w in mask_v, w != u.
                
                # Bit iteration in Python is slow.
                # n=28 is small. Just loop w from 0 to n-1.
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # Check if (v, w) is in E_c_old
                    if (neighbors[c_old][v] >> w) & 1:
                        # count(u, w) will decrease by 1
                        old_cnt = counts[c_old][u][w]
                        if old_cnt == 2:
                            delta -= 1 # cost decreases
                        # if old_cnt > 2, cost unchanged
                        # if old_cnt < 2, cost unchanged (was 0, stays 0)
                    
                    # Check if (u, w) is in E_c_old
                    if (neighbors[c_old][u] >> w) & 1:
                        # count(v, w) will decrease by 1
                        old_cnt = counts[c_old][v][w]
                        if old_cnt == 2:
                            delta -= 1
                    
                # Add to c_new
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # Check if (v, w) is in E_c_new
                    if (neighbors[c_new][v] >> w) & 1:
                        # count(u, w) will increase by 1
                        old_cnt = counts[c_new][u][w]
                        if old_cnt == 1:
                            delta += 1 # cost increases
                        # if old_cnt >= 2, cost unchanged
                        # if old_cnt == 0, cost unchanged
                    
                    # Check if (u, w) is in E_c_new
                    if (neighbors[c_new][u] >> w) & 1:
                        # count(v, w) will increase by 1
                        old_cnt = counts[c_new][v][w]
                        if old_cnt == 1:
                            delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
            
            if best_delta < 0:
                # Apply the best move
                c_new = best_c_new
                
                # Update neighbors
                neighbors[c_old][u] &= ~(1 << v)
                neighbors[c_old][v] &= ~(1 << u)
                neighbors[c_new][u] |= (1 << v)
                neighbors[c_new][v] |= (1 << u)
                
                # Update counts
                # We need to update counts for all affected pairs.
                # Pairs (u, w) where (v, w) in E_c_old or (v, w) in E_c_new?
                # Actually, we can just recompute the count for the specific pairs affected.
                # Affected pairs in c_old: (u, w) for w in N_c_old(v) \ {u}, and (v, w) for w in N_c_old(u) \ {v}.
                # Affected pairs in c_new: (u, w) for w in N_c_new(v) \ {u}, and (v, w) for w in N_c_new(u) \ {v}.
                
                # To be safe and simple, we can just update the specific counts we touched in the delta calculation.
                # But we need to update the `counts` array to reflect the new state.
                
                # Let's iterate w again to update the counts array.
                
                # Remove from c_old
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (neighbors[c_old][v] >> w) & 1:
                        # Note: neighbors[c_old][v] has already been updated?
                        # No, we updated neighbors BEFORE this loop?
                        # In the code above, I updated neighbors first.
                        # So neighbors[c_old][v] no longer has u.
                        # But does it have w?
                        # If (v, w) was in E_c_old, it is still in E_c_old (unless w=u, which we skipped).
                        # So the check `(neighbors[c_old][v] >> w) & 1` is correct for the NEW state?
                        # Wait. The edge (v, w) is NOT being removed. Only (u, v) is removed.
                        # So (v, w) is still in E_c_old.
                        # So the condition holds.
                        
                        # We need to decrement counts[c_old][u][w]
                        counts[c_old][u][w] -= 1
                        counts[c_old][w][u] -= 1
                        
                    if (neighbors[c_old][u] >> w) & 1:
                        counts[c_old][v][w] -= 1
                        counts[c_old][w][v] -= 1
                        
                # Add to c_new
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if (neighbors[c_new][v] >> w) & 1:
                        counts[c_new][u][w] += 1
                        counts[c_new][w][u] += 1
                        
                    if (neighbors[c_new][u] >> w) & 1:
                        counts[c_new][v][w] += 1
                        counts[c_new][w][v] += 1
                        
                colors[e] = c_new
                cost += best_delta
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    
                    if best_cost == 0:
                        return best_colors
            else:
                # No improving move found.
                # Random restart?
                # If we get stuck, restart with a new random initialization.
                # To avoid infinite loops, we can count steps since last improvement.
                # For now, let's just keep trying random edges.
                # If stuck for too long, restart.
                pass # Simplified: just keep trying. The random edge selection will eventually find a move or we restart.
                
                # Add a restart mechanism
                # We need a counter.
                # Let's add a simple restart logic outside this inner loop?
                # Or just break and restart the whole `solve_fast`?
                # Since `solve_fast` is called in a loop in `construct`, we can just return None if stuck?
                # No, better to restart internally.
                
                # Let's add a restart counter.
                # This requires modifying the loop structure.
                # For now, let's assume random restarts in the outer loop of `construct` is sufficient.
                # But if we get stuck in a local minimum, we waste time.
                # Let's add a simple restart: if no improvement for 1000 steps, restart.
                
                # I'll implement a restart by breaking out and letting the outer loop handle it?
                # No, `solve_fast` is a single call.
                # Let's modify `solve_fast` to handle restarts internally.
                
                # Actually, the current structure is a single while loop.
                # Let's add a restart counter.
                
                # I will refactor the loop to include restarts.
                
                # Break out of the current search and restart
                # To do this cleanly, I should wrap the search in a function or use a flag.
                # Given the constraints, I'll just let it run. If it's stuck, it's stuck.
                # But random restarts are crucial.
                
                # Let's restart the initialization.
                break # Break out of the while loop to restart?
                # No, the while loop is the main search.
                # If I break, I return best_colors.
                # If best_colors is not 0, I return it.
                # Then `construct` will try the next n or return it.
                # This is not ideal. I want to try multiple restarts for the same n.
                
                # Let's change the structure:
                # Outer loop: restarts
                # Inner loop: hill climbing
                # If stuck, break inner, go to next restart.
                
                # I will rewrite the loop structure in the final code.
                
                pass

        return best_colors if best_cost == 0 else None

    # Revised solve_fast with restarts
    
    def solve_fast_restarts(n, time_limit):
        total_edges = n * (n - 1) // 2
        if total_edges == 0:
            return []
        
        edge_u = [0] * total_edges
        edge_v = [0] * total_edges
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_u[idx] = i
                edge_v[idx] = j
                idx += 1
        
        best_cost = float('inf')
        best_colors = None
        
        while time.time() < time_limit:
            # Initialize
            neighbors = [[0] * n for _ in range(5)]
            counts = [[[0] * n for _ in range(n)] for _ in range(5)]
            colors = [random.randint(0, 4) for _ in range(total_edges)]
            
            for e in range(total_edges):
                u = edge_u[e]
                v = edge_v[e]
                c = colors[e]
                neighbors[c][u] |= (1 << v)
                neighbors[c][v] |= (1 << u)
                
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        common = neighbors[c][i] & neighbors[c][j]
                        cnt = bin(common).count('1')
                        counts[c][i][j] = cnt
                        counts[c][j][i] = cnt
                        
            cost = 0
            for c in range(5):
                for i in range(n):
                    for j in range(i + 1, n):
                        if counts[c][i][j] >= 2:
                            cost += 1
                            
            if cost == 0:
                return colors
            
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
                
            # Hill Climbing
            steps_since_improvement = 0
            max_steps = 5000
            
            while steps_since_improvement < max_steps and time.time() < time_limit:
                e = random.randint(0, total_edges - 1)
                u = edge_u[e]
                v = edge_v[e]
                c_old = colors[e]
                
                best_delta = 0
                best_c_new = c_old
                
                for c_new in range(5):
                    if c_new == c_old:
                        continue
                    delta = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (neighbors[c_old][v] >> w) & 1:
                            if counts[c_old][u][w] == 2:
                                delta -= 1
                        if (neighbors[c_old][u] >> w) & 1:
                            if counts[c_old][v][w] == 2:
                                delta -= 1
                                
                        if (neighbors[c_new][v] >> w) & 1:
                            if counts[c_new][u][w] == 1:
                                delta += 1
                        if (neighbors[c_new][u] >> w) & 1:
                            if counts[c_new][v][w] == 1:
                                delta += 1
                                
                    if delta < best_delta:
                        best_delta = delta
                        best_c_new = c_new
                        
                if best_delta < 0:
                    c_new = best_c_new
                    neighbors[c_old][u] &= ~(1 << v)
                    neighbors[c_old][v] &= ~(1 << u)
                    neighbors[c_new][u] |= (1 << v)
                    neighbors[c_new][v] |= (1 << u)
                    
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if (neighbors[c_old][v] >> w) & 1:
                            counts[c_old][u][w] -= 1
                            counts[c_old][w][u] -= 1
                        if (neighbors[c_old][u] >> w) & 1:
                            counts[c_old][v][w] -= 1
                            counts[c_old][w][v] -= 1
                            
                        if (neighbors[c_new][v] >> w) & 1:
                            counts[c_new][u][w] += 1
                            counts[c_new][w][u] += 1
                        if (neighbors[c_new][u] >> w) & 1:
                            counts[c_new][v][w] += 1
                            counts[c_new][w][v] += 1
                            
                    colors[e] = c_new
                    cost += best_delta
                    steps_since_improvement = 0
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
                        if best_cost == 0:
                            return best_colors
                else:
                    steps_since_improvement += 1
                    
        return best_colors if best_cost == 0 else None

    # Main Logic
    best_n = 11
    best_colors = []
    
    # Seed solution for n=11
    for i in range(11):
        for j in range(i + 1, 11):
            d = min(j - i, 11 - (j - i))
            best_colors.append(d - 1)
            
    # Try higher n
    for n in [28, 27, 26]:
        if time.time() > deadline - 10:
            break
            
        t_alloc = min(40.0, deadline - time.time() - 5.0)
        if t_alloc <= 0:
            break
            
        res = solve_fast_restarts(n, t_alloc)
        
        if res:
            # Verify
            if len(res) == n * (n - 1) // 2:
                # Quick validity check
                neighbors = [[0] * n for _ in range(5)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = res[idx]
                        neighbors[c][i] |= (1 << j)
                        neighbors[c][j] |= (1 << i)
                        idx += 1
                        
                valid = True
                for c in range(5):
                    for u in range(n):
                        nu = neighbors[c][u]
                        for v in range(u + 1, n):
                            nv = neighbors[c][v]
                            common = nu & nv
                            if common & (common - 1):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                        
                if valid:
                    best_n = n
                    best_colors = res
                    # Since we go high to low, we can return immediately?
                    # Yes, we want to maximize n.
                    return {'n': best_n, 'colors': best_colors}
                    
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
