import numpy as np
import random
import time

# Precompute validity of all 64 tournaments on 4 vertices
# Vertices 0, 1, 2, 3
# Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
# Bit i in mask corresponds to edge i.
# 0: i -> j, 1: j -> i
valid_tournament_mask = [False] * 64

for mask in range(64):
    # Reconstruct out-degrees
    out_deg = [0, 0, 0, 0]
    # Edge 0: (0,1)
    if mask & (1 << 0):
        out_deg[1] += 1
    else:
        out_deg[0] += 1
    # Edge 1: (0,2)
    if mask & (1 << 1):
        out_deg[2] += 1
    else:
        out_deg[0] += 1
    # Edge 2: (0,3)
    if mask & (1 << 2):
        out_deg[3] += 1
    else:
        out_deg[0] += 1
    # Edge 3: (1,2)
    if mask & (1 << 3):
        out_deg[2] += 1
    else:
        out_deg[1] += 1
    # Edge 4: (1,3)
    if mask & (1 << 4):
        out_deg[3] += 1
    else:
        out_deg[1] += 1
    # Edge 5: (2,3)
    if mask & (1 << 5):
        out_deg[3] += 1
    else:
        out_deg[2] += 1
    
    if max(out_deg) == 3:
        valid_tournament_mask[mask] = True

def get_mask(t, a, b, c, d):
    # t is list of lists, t[i][j] is 1 if j->i, 0 if i->j
    # Wait, standard: t[i][j] = 0 means i->j. 1 means j->i.
    # Let's stick to: t[i][j] = 1 if i->j, 0 if j->i? 
    # The prompt says: "0 means i -> j, 1 means j -> i"
    # So t[i][j] = 1 implies j beats i.
    # Let's use t[i][j] as 1 if j->i.
    
    # Edge (a,b): index 0. 0 means a->b. 1 means b->a.
    # In our t, t[a][b] is 1 if b->a. So if t[a][b]==1, bit is 1.
    # So bit 0 = t[a][b]
    
    m = 0
    m |= t[a][b] << 0
    m |= t[a][c] << 1
    m |= t[a][d] << 2
    m |= t[b][c] << 3
    m |= t[b][d] << 4
    m |= t[c][d] << 5
    return m

def count_bad_sets(t, n):
    bad = 0
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    m = get_mask(t, a, b, c, d)
                    if not valid_tournament_mask[m]:
                        bad += 1
    return bad

def simulate_annealing(n, time_limit):
    # Initialize random tournament
    t = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                t[i][j] = 1 # j->i
                t[j][i] = 0
            else:
                t[i][j] = 0 # i->j
                t[j][i] = 1
                
    current_cost = count_bad_sets(t, n)
    if current_cost == 0:
        return t
        
    T = 1.0
    cooling = 0.999
    start_time = time.time()
    
    # Precompute pairs for speed? 
    # We need to iterate over all edges (u,v) to pick one to flip.
    # And for each flip, iterate over all (x,y) from remaining.
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        # Ensure u < v for consistency? Not strictly needed if we handle matrix.
        # But for flipping, we just toggle t[u][v] and t[v][u].
        
        # Calculate delta cost
        # Old cost contribution of sets containing (u,v)
        # New cost contribution
        
        # To speed up, we can just compute the cost of the affected sets.
        # Affected sets are {u, v, x, y} where x,y in [0..n-1] \ {u,v}
        
        delta = 0
        # We need to be careful: if we flip, the mask changes.
        # We can compute old_mask and new_mask for each affected set.
        
        # Optimization: 
        # The mask only changes in the bit corresponding to (u,v).
        # Let bit_idx be the index of (u,v) in the 4-set's edge ordering.
        # This depends on the relative order of u,v,x,y.
        
        # Simpler: Just recompute mask for each affected set.
        # There are ~n^2/2 affected sets.
        # For n=60, ~900 sets.
        
        # To avoid modifying t before calculating delta, we calculate delta based on current t.
        
        # List of other vertices
        others = [i for i in range(n) if i != u and i != v]
        
        # We need to sum up (new_cost - old_cost) for all pairs in others
        # old_cost_contrib = sum(1 if not valid(old_mask) else 0)
        # new_cost_contrib = sum(1 if not valid(new_mask) else 0)
        
        # We can iterate pairs in others.
        # Note: u and v are fixed.
        
        # To make it faster, we can precompute the masks for all 4-sets? No, too much memory.
        
        # Let's just loop.
        # We need to know the bit position of (u,v) in the mask for a given set {u,v,x,y}.
        
        # Let's sort the 4 vertices a<b<c<d.
        # The bit position of (u,v) depends on their positions in the sorted list.
        
        # This is getting complex. 
        # Alternative: Just flip, calculate new cost of affected sets, add to delta, unflip?
        # No, we need delta to decide.
        
        # Let's compute old_bad and new_bad for affected sets.
        
        old_bad = 0
        new_bad = 0
        
        # We will iterate over all x, y in others with x < y
        for i in range(len(others)):
            x = others[i]
            for j in range(i+1, len(others)):
                y = others[j]
                
                # Set is {u, v, x, y}
                # Sort them to get canonical order a,b,c,d
                verts = sorted([u, v, x, y])
                a, b, c, d = verts
                
                # Get old mask
                old_m = get_mask(t, a, b, c, d)
                if not valid_tournament_mask[old_m]:
                    old_bad += 1
                    
                # Determine new mask
                # The edge (u,v) is flipped.
                # Find which bit in the mask corresponds to (u,v).
                # The mask bits are for (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
                # We need to find the index k such that {verts[k], verts[k+1]} == {u,v} is false?
                # No, the edges are specific pairs.
                # Edge 0: (a,b)
                # Edge 1: (a,c)
                # Edge 2: (a,d)
                # Edge 3: (b,c)
                # Edge 4: (b,d)
                # Edge 5: (c,d)
                
                # Find which edge is (u,v)
                # u and v are in {a,b,c,d}.
                if (a == u and b == v) or (a == v and b == u):
                    bit = 0
                elif (a == u and c == v) or (a == v and c == u):
                    bit = 1
                elif (a == u and d == v) or (a == v and d == u):
                    bit = 2
                elif (b == u and c == v) or (b == v and c == u):
                    bit = 3
                elif (b == u and d == v) or (b == v and d == u):
                    bit = 4
                elif (c == u and d == v) or (c == v and d == u):
                    bit = 5
                else:
                    bit = -1 # Should not happen
                
                new_m = old_m
                if bit != -1:
                    # Flip the bit
                    new_m = old_m ^ (1 << bit)
                    
                if not valid_tournament_mask[new_m]:
                    new_bad += 1
                    
        delta = new_bad - old_bad
        
        # Acceptance
        if delta < 0:
            # Flip
            t[u][v] = 1 - t[u][v]
            t[v][u] = 1 - t[v][u]
            current_cost += delta
        else:
            if random.random() < np.exp(-delta / T):
                t[u][v] = 1 - t[u][v]
                t[v][u] = 1 - t[v][u]
                current_cost += delta
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
        if current_cost == 0:
            return t

    return None

def construct():
    # Try to find a small n
    # Lower bound 31
    # We will try n from 31 to 83
    # Time budget per n: 2 seconds
    # Total time: 120s -> 60 * 2 = 120s. Tight.
    # Let's try larger n first? No, we want minimal n.
    # But small n is harder.
    # Maybe try 40, 50, 60 first?
    # If we find 60, that's good.
    # If we can find 40, that's better.
    
    # Let's try a range.
    # We will use a total time budget of 110s.
    # We will try n in increasing order.
    # For each n, we give it a time slice.
    
    start_total = time.time()
    
    # Heuristic: Smaller n is harder. 
    # Let's try n=50, 55, 60 first as they are more likely to be solvable by SA quickly.
    # If we find one, we return it.
    # If we want to minimize, we should try smaller n.
    # But if we spend all time on n=31 and fail, we score 0 (or fallback).
    # Fallback is 83.
    # So we should try to find *any* n < 83.
    # Then maybe try to reduce?
    
    # Strategy:
    # 1. Try n=60 for 10s.
    # 2. Try n=55 for 10s.
    # 3. Try n=50 for 10s.
    # 4. Try n=45 for 10s.
    # 5. Try n=40 for 10s.
    # 6. Try n=35 for 10s.
    # 7. Try n=31 for remaining time.
    
    # This maximizes chance of finding a solution.
    
    candidates = [60, 55, 50, 45, 40, 35, 31]
    
    for n in candidates:
        if time.time() - start_total > 100:
            break
            
        # Time slice
        slice_time = 10.0
        if n == 31:
             slice_time = 110.0 - (time.time() - start_total)
             if slice_time < 1: continue
             
        t = simulate_annealing(n, slice_time)
        
        if t is not None:
            # Verify
            if count_bad_sets(t, n) == 0:
                # Format output
                orientation = []
                for i in range(n):
                    for j in range(i+1, n):
                        orientation.append(t[i][j])
                return {'n': n, 'orientation': orientation}
                
    # Fallback: Paley 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}

