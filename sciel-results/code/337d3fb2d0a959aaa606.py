import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) points
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                vec = [a, b, c]
                # Normalize: make the first non-zero coordinate 1
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)
                        vec = [(v * inv) % 5 for v in vec]
                        break
                tvec = tuple(vec)
                if tvec not in seen:
                    seen.add(tvec)
                    points.append(tvec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # 2. Construct polarity graph
    adj = [0] * n
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
                
    # Helper to count C4s using bitmasks
    def count_c4s(adj_list):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_list[u] & adj_list[v]
                k = bin(common).count('1')
                if k >= 2:
                    total += k * (k - 1) // 2
        return total // 2

    # Helper to calculate delta C4 when adding edge (x, y) to current adj
    # Assumes x, y are not connected
    def delta_add(x, y, adj_list):
        delta = 0
        # The new C4s are formed by paths of length 3 between x and y.
        # A path x-u-v-y becomes a C4 x-u-v-y-x.
        # This happens if u is a neighbor of x, v is a neighbor of y, and u-v is an edge.
        # Alternatively, count common neighbors of pairs?
        # The standard formula for delta when adding (x,y):
        # Increase in C4s is the number of length-3 paths between x and y.
        # Number of length-3 paths from x to y = sum_{v in N(x)} |N(v) \cap N(y)|
        # Wait, this counts paths x-v-w-y.
        # Let's verify.
        # If we add (x,y), each path x-v-w-y becomes a C4.
        # So we need to count the number of such paths.
        
        # Efficient way:
        # For each neighbor v of x:
        #   Count how many neighbors of v are also neighbors of y.
        #   This is popcount(adj[v] & adj[y]).
        # Sum these up.
        
        count = 0
        for v in range(n):
            if adj_list[x] & (1 << v):
                count += bin(adj_list[v] & adj_list[y]).count('1')
        return count

    # Helper to calculate delta C4 when removing edge (x, y) from current adj
    # Assumes x, y are connected
    def delta_remove(x, y, adj_list):
        # Removing (x,y) destroys C4s that use this edge.
        # A C4 using (x,y) must be of the form x-y-v-w-x.
        # This is determined by pairs of common neighbors of x and y.
        # If x and y have k common neighbors, there are k*(k-1)/2 C4s using edge (x,y).
        # Wait, is that right?
        # A C4 is a cycle of 4. If (x,y) is in the cycle, the other two vertices u, v must be common neighbors of x and y, and u-v must be an edge?
        # No. The cycle is x-y-v-u-x.
        # y is connected to v. x is connected to u. u is connected to v.
        # Also x is connected to y.
        # So u and v are common neighbors of x and y?
        # x-u, x-y, y-v.
        # For x-y-v-u-x to be a cycle:
        # 1. x-y is edge (yes)
        # 2. y-v is edge (yes, v in N(y))
        # 3. v-u is edge (yes, u in N(v))
        # 4. u-x is edge (yes, u in N(x))
        # So we need u in N(x) \cap N(v) and v in N(y) \cap N(u).
        # This is getting complicated.
        
        # Let's use the symmetric property.
        # Delta_add(x,y) on graph G' (without xy) gives the number of C4s added.
        # If we have G (with xy), removing xy reduces C4s by the number of C4s that contain xy.
        # The number of C4s containing edge (x,y) is equal to the number of paths of length 3 between x and y in G - (x,y)?
        # No.
        # Let's just calculate the change in the total C4 count.
        # Total C4s = sum_{u<v} binom(|N(u) \cap N(v)|, 2) / 2? No, that counts each C4 twice?
        # Actually, the formula `sum_{u<v} binom(common, 2)` counts each C4 exactly twice?
        # A C4 (a,b,c,d) has common neighbors:
        # (a,c) share b,d. (b,d) share a,c.
        # So summing binom(common, 2) over all pairs counts each C4 twice (once for the diagonal a-c, once for b-d).
        # So total C4s = 0.5 * sum_{u<v} binom(|N(u) \cap N(v)|, 2).
        
        # If we remove (x,y):
        # N(x) loses y. N(y) loses x.
        # For any pair (u, v):
        # If u=x or v=x or u=y or v=y, the common neighbor count changes.
        # This is what the previous code did. It seemed correct.
        
        delta = 0
        for v in range(n):
            if v == x or v == y:
                continue
            
            # Pair (x, v)
            # Before removal: common = N(x) \cap N(v). Since y is in N(x), if y in N(v), y was in common.
            k_old = bin(adj_list[x] & adj_list[v]).count('1')
            # After removal: y is not in N(x).
            if adj_list[v] & (1 << y):
                k_new = k_old - 1
            else:
                k_new = k_old
            
            delta += (k_new * (k_new - 1) // 2) - (k_old * (k_old - 1) // 2)
            
            # Pair (y, v)
            k_old = bin(adj_list[y] & adj_list[v]).count('1')
            # After removal: x is not in N(y).
            if adj_list[v] & (1 << x):
                k_new = k_old - 1
            else:
                k_new = k_old
            
            delta += (k_new * (k_new - 1) // 2) - (k_old * (k_old - 1) // 2)
            
        # Pair (x, y)
        # Before: common = N(x) \cap N(y).
        # After: N'(x) = N(x) \ {y}, N'(y) = N(y) \ {x}.
        # Common' = N(x) \ N(y) ... no.
        # N'(x) \cap N'(y) = (N(x) \ {y}) \cap (N(y) \ {x}) = N(x) \cap N(y) since x notin N(x) and y notin N(y).
        # So k_old for pair (x,y) is unchanged?
        # Wait. k_xy = |N(x) \cap N(y)|.
        # The formula for C4 count involves binom(k, 2).
        # Does k_xy change?
        # N(x) changes by removing y. Is y in N(y)? No. So y was not in N(x) \cap N(y).
        # N(y) changes by removing x. Is x in N(x)? No. So x was not in N(x) \cap N(y).
        # So N(x) \cap N(y) is unchanged.
        # Thus, the term for pair (x,y) does not change.
        
        return delta

    # 3. Add the 91st edge
    if len(edges) < 91:
        best_delta = float('inf')
        best_edge = None
        
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    # Calculate C4 increment
                    d = delta_add(i, j, adj)
                    if d < best_delta:
                        best_delta = d
                        best_edge = (i, j)
                        
        if best_edge:
            i, j = best_edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            edges.append((i, j))
            
    current_c4s = count_c4s(adj)
    
    # 4. Local Search with Restarts
    improved = True
    stall_count = 0
    MAX_STALL = 20
    
    while time.time() < deadline - 2:
        improved = False
        best_swap = None
        best_delta_total = 0 # We want to minimize C4s, so we want negative delta
        
        edges_list = list(edges)
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
                    
        # Randomize order to escape local minima?
        # random.shuffle(edges_list)
        # random.shuffle(non_edges)
        
        for e1 in edges_list:
            u1, v1 = e1
            # Remove e1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            local_best_delta = 0
            local_best_edge = None
            
            for e2 in non_edges:
                u2, v2 = e2
                # Calculate delta for adding e2 to the graph where e1 is removed
                d = delta_add(u2, v2, adj)
                
                if d < local_best_delta:
                    local_best_delta = d
                    local_best_edge = e2
                    
            # Revert remove
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            
            if local_best_edge and local_best_delta < best_delta_total:
                best_delta_total = local_best_delta
                best_swap = (e1, local_best_edge)
                
        if best_swap and best_delta_total < 0:
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            
            # Perform swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            # Update edges list
            edges.remove(e1)
            edges.append(e2)
            
            current_c4s += best_delta_total
            improved = True
            stall_count = 0
        else:
            # No improvement found in this pass
            stall_count += 1
            if stall_count > MAX_STALL:
                # Random restart?
                # Perturb the graph: remove a random edge, add a random non-edge
                if non_edges and edges:
                    e1 = random.choice(edges)
                    e2 = random.choice(non_edges)
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    # Ensure e1 and e2 are distinct and valid
                    # e1 is in edges, e2 is in non_edges, so they are distinct.
                    
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    
                    edges.remove(e1)
                    edges.append(e2)
                    
                    current_c4s = count_c4s(adj)
                    stall_count = 0
                else:
                    break # Cannot improve further
            
    # Return edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                result.append([i, j])
                
    return result
