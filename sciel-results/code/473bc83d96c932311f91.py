"""Constructs a 5-colouring of K_n with no monochromatic C4."""
import time
import random
import math

def construct():
    t_start = time.time()
    deadline = 130.0
    
    def get_deadline():
        return t_start + deadline
    
    def count_c4s(colors, n, color):
        """Count number of C4s in a specific color class."""
        # Build adjacency sets for this color
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                idx = i * (n - 1) - i*(i+1)//2 + (j - i - 1)
                if colors[idx] == color:
                    adj[i].add(j)
                    adj[j].add(i)
        
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                # Count common neighbors of i and j
                common = len(adj[i] & adj[j])
                c4_count += common
        return c4_count // 2  # Each C4 counted twice? No, each pair (i,j) with common neighbors k,l forms a C4.
                               # Wait, a C4 is i-k-j-l-i. The pair (i,j) with common neighbors {k,l} is one C4.
                               # So sum of common neighbors over all pairs is exactly the number of C4s.
                               # Let's re-verify. A C4 has 4 edges. It has 2 pairs of opposite vertices.
                               # Each pair of opposite vertices shares 2 common neighbors.
                               # So for a specific C4, it contributes 2 to the sum?
                               # Let vertices be 0,1,2,3 in cycle.
                               # Pairs: (0,2) common neighbors {1,3} -> count 2.
                               # Pairs: (1,3) common neighbors {0,2} -> count 2.
                               # Pairs: (0,1) common neighbors {} -> 0.
                               # So sum is 4. But there is only 1 C4.
                               # So dividing by 2 is correct? No, 4/2 = 2. Still wrong.
                               # Actually, each C4 is counted by its two pairs of opposite vertices.
                               # For each such pair, the number of common neighbors is at least 2.
                               # If common neighbors are exactly {k,l}, then it contributes 2 to the sum for that pair.
                               # So total contribution of one C4 is 2 (from one pair) + 2 (from other pair)? No.
                               # For pair (i,j), if common neighbors are {k,l}, then the C4s are i-k-j-l-i and i-l-j-k-i?
                               # No, i-k-j-l is a path. The cycle is i-k-j-l-i.
                               # So for pair (i,j) with common neighbors k,l, we have one C4: i-k-j-l-i.
                               # Wait, i and j are not adjacent in this C4.
                               # The C4 is i-k-j-l-i.
                               # So the pair (i,j) "sees" the two neighbors k and l.
                               # Does this correspond to 1 C4 or 2?
                               # The cycle is i -> k -> j -> l -> i. This is one specific cycle.
                               # If we swap k and l, we get i -> l -> j -> k -> i, which is the same cycle reversed?
                               # No, i-k-j-l-i is the same as i-l-j-k-i? No.
                               # i-k-j-l-i visits i, k, j, l.
                               # i-l-j-k-i visits i, l, j, k.
                               # These are different cycles if k != l.
                               # However, in an undirected graph, a cycle is a set of edges.
                               # Edges for first: (i,k), (k,j), (j,l), (l,i).
                               # Edges for second: (i,l), (l,j), (j,k), (k,i).
                               # Since edges are undirected, (i,k) is (k,i), etc.
                               # The set of edges is the same! {(i,k), (j,k), (j,l), (i,l)}.
                               # So for a pair (i,j) with common neighbors {k,l}, there is exactly ONE C4 involving i,j as opposite vertices.
                               # And each C4 has exactly two pairs of opposite vertices.
                               # So the sum of common neighbors over all pairs counts each C4 exactly twice.
                               # Therefore, dividing by 2 gives the number of C4s.
                               # My previous code `return c4_count // 2` was correct.
    
    def count_c4s_fast(colors, n, color):
        """Faster count using bitmasks or sets."""
        adj = [set() for _ in range(n)]
        # Precompute index mapping
        idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                idx_map[(i,j)] = idx
                idx += 1
        
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx_map[(i,j)]] == color:
                    adj[i].add(j)
                    adj[j].add(i)
        
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = len(adj[i] & adj[j])
                c4_count += common
        return c4_count // 2

    # We will use a simpler, robust approach.
    # Start with n=11. Try to increment n.
    # For each n, use Simulated Annealing to minimize total C4s.
    
    best_result = {'n': 11, 'colors': [min(j - i, n - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]}
    
    for target_n in range(12, 27):
        if time.time() > get_deadline() - 30:
            break
            
        # Initialize colors
        # Use a heuristic init: random or based on smaller n
        colors = [random.randint(0, 4) for _ in range(target_n * (target_n - 1) // 2)]
        
        # Precompute indices
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * (target_n - 1) - i*(i+1)//2 + (j - i - 1)
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        max_iter = 500000 # Adjust based on time
        
        # Fast C4 counter using bitsets?
        # For n up to 26, N^2 sets are fine.
        
        current_energy = sum(count_c4s_fast(colors, target_n, c) for c in range(5))
        best_energy = current_energy
        best_colors = colors[:]
        
        for it in range(max_iter):
            if time.time() > get_deadline() - 15:
                break
                
            # Pick a random edge
            idx = random.randint(0, len(colors) - 1)
            # Reverse mapping from idx to (i,j)
            # This is slow. Let's precompute list of (i,j)
            
            # Optimization: Precompute list of pairs
            pairs = [(i, j) for i in range(target_n) for j in range(i+1, target_n)]
            # This is O(n^2) memory, fine for n=26.
            
            # Actually, we need to update energy efficiently.
            # Changing one edge (u,v) from c_old to c_new.
            # This affects C4s involving edge (u,v).
            # A C4 containing (u,v) must have u and v as adjacent vertices.
            # C4: u-v-w-x-u.
            # Common neighbors of u and v? No, u and v are adjacent.
            # The C4 is formed by u, v, and two other vertices w, x such that:
            # (v,w), (w,x), (x,u) are in the same color c.
            # So for color c, let N_c(u) be neighbors of u in color c.
            # Let N_c(v) be neighbors of v in color c.
            # A C4 involving edge (u,v) in color c is determined by a pair (w,x) where:
            # w in N_c(v), x in N_c(u), and (w,x) in color c.
            # Also w != x.
            
            # This delta calculation is complex.
            # Given the time constraints and the need for robustness,
            # let's stick to the full recount but optimize it.
            # For n=26, edges = 325.
            # Recounting all C4s:
            # For each color (5):
            #   Build adj (325 checks)
            #   For each pair (i,j) (325 pairs):
            #     Intersection of sets (size <= 24)
            # Total ops ~ 5 * 325 * 24 ~ 40,000.
            # 500,000 iterations * 40,000 ops = 20 billion ops. Too slow.
            
            # We MUST use incremental updates or a faster method.
            # Let's use the bitmask approach for adjacency.
            # adj[c][v] is a bitmask of neighbors of v in color c.
            
            # Re-initialize for SA with bitmasks
            adj = [[0] * target_n for _ in range(5)]
            for i in range(len(colors)):
                c = colors[i]
                # Need (i,j) for idx i
                # Let's precompute pairs list once
                pass
            
            # Let's rewrite the SA loop properly with bitmasks
            adj = [[0] * target_n for _ in range(5)]
            pairs = []
            idx = 0
            for i in range(target_n):
                for j in range(i+1, target_n):
                    pairs.append((i, j))
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            def get_energy():
                e = 0
                for c in range(5):
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            common = (adj[c][i] & adj[c][j])
                            # Popcount
                            e += bin(common).count('1')
                return e // 2
            
            current_energy = get_energy()
            best_energy = current_energy
            best_colors = colors[:]
            best_adj = [row[:] for row in adj]
            
            for it in range(max_iter):
                if time.time() > get_deadline() - 15:
                    break
                    
                # Pick random edge
                idx = random.randint(0, len(pairs) - 1)
                u, v = pairs[idx]
                c_old = colors[idx]
                c_new = random.randint(0, 4)
                if c_old == c_new:
                    continue
                
                # Calculate delta energy
                # Remove (u,v) from c_old, add to c_new.
                
                # For c_old:
                # C4s lost: pairs (w,x) such that w in N_{c_old}(v), x in N_{c_old}(u), (w,x) in c_old.
                # Actually, simpler:
                # Energy contribution of color c is sum over pairs (i,j) of |N_c(i) \cap N_c(j)|.
                # Changing edge (u,v) affects:
                # 1. Pairs (u, x) for x != v.
                # 2. Pairs (v, x) for x != u.
                # 3. Pair (u, v) itself.
                
                # Let's compute delta for c_old (removing) and c_new (adding).
                
                def delta_energy(c, u, v, adding):
                    # sign = +1 if adding, -1 if removing
                    # We need to update adj[c] and see change in energy.
                    # Instead of full update, compute delta.
                    # Let S_u = N_c(u) without v (if removing) or with v (if adding)?
                    # When we add (u,v) to color c:
                    # - N_c(u) gains v. N_c(v) gains u.
                    # - For any pair (i, j):
                    #   - If {i,j} = {u,v}: intersection changes.
                    #   - If i = u, j != v: N_c(u) changes, so |N_c(u) \cap N_c(j)| changes by 1 if v in N_c(j), else 0.
                    #   - If i = v, j != u: N_c(v) changes, so |N_c(v) \cap N_c(j)| changes by 1 if u in N_c(j), else 0.
                    #   - Else: no change.
                    
                    # So delta =
                    # 1. Change in intersection for pair (u,v):
                    #    Old: |N(u) \cap N(v)| (before change)
                    #    New: |N(u)' \cap N(v)'| (after change)
                    #    If adding: N(u)' = N(u) U {v}, N(v)' = N(v) U {u}.
                    #    Intersection: (N(u) U {v}) \cap (N(v) U {u})
                    #    = (N(u) \cap N(v)) U (N(u) \cap {u}) U ({v} \cap N(v)) U ({v} \cap {u})
                    #    Note: u not in N(u), v not in N(v).
                    #    So = (N(u) \cap N(v)) U ({v} if v in N(v) else empty) ... wait.
                    #    v in N(v) is false. u in N(u) is false.
                    #    So = (N(u) \cap N(v)) U ({v} if v in N(u)? No, v is in N(u)' but is v in N(v)'? Yes. But v is in N(u) already? No, we are adding.
                    #    Let's be precise.
                    #    A = N(u), B = N(v).
                    #    A' = A U {v}, B' = B U {u}.
                    #    A' \cap B' = (A \cap B) U (A \cap {u}) U ({v} \cap B) U ({v} \cap {u})
                    #    A \cap {u} is empty (u not in A).
                    #    {v} \cap {u} is empty.
                    #    {v} \cap B: is v in B? Yes, if (u,v) was already in color c? No, we are adding. So v not in B.
                    #    Wait, if we are adding (u,v) to color c, it was NOT in color c before.
                    #    So v not in N(u) and u not in N(v).
                    #    So {v} \cap B is empty.
                    #    So A' \cap B' = A \cap B.
                    #    So intersection for pair (u,v) does NOT change?
                    #    Wait, if u and v are connected, they are in each other's neighborhoods.
                    #    N(u) contains v if edge (u,v) is in color c.
                    #    So if adding, v is added to N(u).
                    #    So A = N(u) (old), v not in A.
                    #    A' = A U {v}.
                    #    B = N(v) (old), u not in B.
                    #    B' = B U {u}.
                    #    A' \cap B' = (A U {v}) \cap (B U {u})
                    #    = (A \cap B) U (A \cap {u}) U ({v} \cap B) U ({v} \cap {u})
                    #    A \cap {u}: u in A? No.
                    #    {v} \cap B: v in B? No.
                    #    So A' \cap B' = A \cap B.
                    #    So the term for pair (u,v) is unchanged.
                    
                    # 2. Change for pairs (u, w) where w != v.
                    #    N(u) becomes A'.
                    #    |A' \cap N(w)| - |A \cap N(w)|
                    #    = | (A U {v}) \cap N(w) | - |A \cap N(w)|
                    #    = 1 if v in N(w), else 0.
                    #    Sum over w != v.
                    #    This is the number of w != v such that v in N_c(w).
                    #    i.e., number of common neighbors of u and v in color c, EXCLUDING v?
                    #    No, w is a neighbor of u. We check if v is a neighbor of w.
                    #    So we are counting edges (w,v) in color c where w is in N(u).
                    #    i.e., |N(u) \cap N(v)|?
                    #    If w in N(u) and v in N(w), then w in N(v) (symmetry).
                    #    So w is a common neighbor of u and v.
                    #    So this sum is |N(u) \cap N(v)|.
                    
                    # 3. Change for pairs (v, w) where w != u.
                    #    Similarly, sum is |N(v) \cap N(u)| = |N(u) \cap N(v)|.
                    
                    # Total delta for adding = 2 * |N(u) \cap N(v)|.
                    # For removing, delta = -2 * |N(u) \cap N(v)|.
                    # BUT, we must use the OLD N(u) and N(v) before the change.
                    
                    if adding:
                        # N(u) and N(v) do not contain each other.
                        common = (adj[c][u] & adj[c][v])
                        count = bin(common).count('1')
                        return 2 * count
                    else:
                        # N(u) and N(v) DO contain each other.
                        # We need to remove (u,v) from consideration?
                        # The formula derived was for adding.
                        # For removing, we reverse.
                        # Old state: connected. New state: not connected.
                        # Delta = New - Old.
                        # Let's re-derive for removing.
                        # A = N(u) \ {v}, B = N(v) \ {u}. (These are the neighbors after removal).
                        # Old A_old = A U {v}, B_old = B U {u}.
                        # Pair (u,v):
                        #   Old inter: (A U {v}) \cap (B U {u}) = (A \cap B) U (A \cap {u}) U ({v} \cap B) U ({v} \cap {u})
                        #   A \cap {u} empty. {v} \cap {u} empty.
                        #   {v} \cap B: v in B? No (v not in N(v)).
                        #   So Old inter = A \cap B.
                        #   New inter: A \cap B.
                        #   Change: 0.
                        
                        # Pairs (u, w) w != v:
                        #   New N(u) = A. Old N(u) = A U {v}.
                        #   |A \cap N(w)| - |(A U {v}) \cap N(w)|
                        #   = -1 if v in N(w), else 0.
                        #   Sum over w != v.
                        #   Count of w != v such that v in N(w).
                        #   i.e., w in N(v) and w != v.
                        #   Since w in N(v) implies w != v (no self loops).
                        #   So this is |N(v)|? No.
                        #   w must be in N(u) (since we are summing over neighbors w of u).
                        #   So w in N(u) and v in N(w) => w in N(v).
                        #   So w is common neighbor of u and v.
                        #   In OLD state, u and v are connected.
                        #   Common neighbors are (A_old \cap B_old) \ {u,v}?
                        #   A_old = N(u), B_old = N(v).
                        #   Inter = N(u) \cap N(v).
                        #   This set contains w such that w-u and w-v are edges.
                        #   Does it contain u or v? No.
                        #   So count is |N(u) \cap N(v)|.
                        
                        # So delta for removing = -2 * |N(u) \cap N(v)|.
                        common = (adj[c][u] & adj[c][v])
                        count = bin(common).count('1')
                        return -2 * count
                
                delta_old = delta_energy(c_old, u, v, adding=False)
                delta_new = delta_energy(c_new, u, v, adding=True)
                delta_total = delta_old + delta_new
                
                if delta_total < 0 or random.random() < math.exp(-delta_total / T):
                    # Accept
                    colors[idx] = c_new
                    # Update adj
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    
                    current_energy += delta_total
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors[:]
                        best_adj = [row[:] for row in adj]
                    
                    T *= cooling
                    if T < 1e-6:
                        T = 1e-6
                        
        # After SA, check if valid
        if best_energy == 0:
            best_result = {'n': target_n, 'colors': best_colors}
            return best_result
            
        # If not valid, keep best result so far and try next n?
        # Or try to repair?
        # For now, just continue to next n, hoping a larger n starts from a better random seed?
        # No, larger n is harder.
        # We should probably keep the best valid n found.
        # If we didn't find valid for 12, we can't jump to 13 easily.
        # We should restart with a better init?
        # Let's just return the best valid n found.
        
    return best_result

# Run once to test? No, construct() is called by evaluator.
if __name__ == "__main__":
    res = construct()
    print(res['n'])
