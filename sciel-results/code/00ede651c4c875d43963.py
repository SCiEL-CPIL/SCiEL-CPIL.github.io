import time
import random
import numpy as np

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing 4-cycles.
    Strategy:
    1. Construct a C4-free core with 90 edges using the polarity graph of PG(2,5).
    2. Select the best single edge to add to minimize new C4s.
    3. Use fast local search (edge swaps) with delta-C4 calculation to minimize the total number of C4s.
    """
    n = 31
    q = 5
    m = 91
    
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin under 120s limit

    # --- Step 1: Build C4-free core with 90 edges using PG(2,5) polarity ---
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: B(p,q) = x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    adj = [0] * n  # Bitsets for neighbors
    edges_core = []
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 - z1*z2) % 5 == 0:
                edges_core.append((i, j))
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    if len(edges_core) != 90:
        # Fallback: Greedy C4-free construction
        adj = [0] * n
        edges_core = []
        random.seed(42)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges_core) >= 90:
                break
            # Check if adding (u,v) creates a C4
            # C4 u-v-a-b-u requires a in N(v), b in N(u), a!=b, (a,b) in E
            creates_c4 = False
            Nv = adj[v]
            Nu = adj[u]
            # Iterate bits in Nv
            a = Nv
            while a:
                lsb = a & -a
                a_idx = lsb.bit_length() - 1
                a -= lsb
                # Check if any neighbor of a in Nu is adjacent to a (and != a)
                # Neighbors of a in Nu: adj[a] & Nu
                # If a is in Nu, we must be careful.
                # Actually, if we just check if there is ANY edge (x,y) with x in Nv, y in Nu, x!=y, (x,y) in E
                # This is equivalent to: for each x in Nv, is there y in Nu such that y in adj[x] and y != x?
                common_with_a = adj[a_idx] & Nu
                # If a_idx is in Nu, then a_idx is a candidate for y=x, which is invalid.
                # So if a_idx in Nu, we need at least 1 other neighbor in Nu? No, we just need 1 valid pair.
                # If common_with_a has any bit set, and it's not ONLY the bit for a_idx (if a_idx in Nu), then creates_c4.
                if common_with_a:
                    if a_idx in Nu:
                        # Check if there is another bit
                        if common_with_a != (1 << a_idx):
                            creates_c4 = True
                            break
                    else:
                        creates_c4 = True
                        break
            if not creates_c4:
                edges_core.append((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if len(edges_core) < 90:
            for u, v in all_pairs:
                if len(edges_core) >= 90:
                    break
                if not (adj[u] >> v) & 1:
                    edges_core.append((u, v))
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

    # --- Step 2: Find best single edge to add to minimize C4s ---
    
    def popcount(x):
        return bin(x).count('1')

    def count_added_c4s(u, v, adj):
        """
        Count C4s created by adding edge (u,v) to graph defined by adj.
        Assumes (u,v) is NOT in adj.
        """
        Nu = adj[u]
        Nv = adj[v]
        count = 0
        a = Nv
        while a:
            lsb = a & -a
            a_idx = lsb.bit_length() - 1
            a -= lsb
            # Count neighbors of a_idx in Nu
            count += popcount(adj[a_idx] & Nu)
        # Subtract cases where a=b
        common = Nu & Nv
        count -= popcount(common)
        return count

    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                continue
            add = count_added_c4s(u, v, adj)
            if add < best_add:
                best_add = add
                best_edge = (u, v)
    
    if best_edge is None:
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    best_edge = (u, v)
                    break
            if best_edge:
                break

    # Apply the best edge
    final_edges = set(edges_core)
    final_edges.add(best_edge)
    u, v = best_edge
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    
    # --- Step 3: Local search with fast delta calculation ---
    
    def count_c4_total(adj):
        """
        Count total C4s in graph.
        Sum over non-edges (u,v) of C(common(u,v), 2)
        """
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if not ((adj[u] >> v) & 1):
                    cn = popcount(adj[u] & adj[v])
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
        return total

    current_c4s = count_c4_total(adj)
    
    # Helper to get list of edges and non-edges
    def get_edges_non_edges(adj):
        edges = []
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    edges.append((u, v))
                else:
                    non_edges.append((u, v))
        return edges, non_edges

    edges_list, non_edges_list = get_edges_non_edges(adj)
    
    # Simulated Annealing / Iterated Local Search
    random.seed(42)
    
    # Precompute neighbors lists for faster iteration in delta?
    # Bitsets are fast enough.
    
    improved = True
    max_iter = 0
    while improved and time.time() < deadline - 10:
        improved = False
        max_iter += 1
        
        # Try all single swaps? Or random sample?
        # With fast delta, we can try more.
        # Let's do a random restart of local search if stuck.
        
        # 1. Greedy local search: try all swaps, pick best improvement
        best_delta = 0
        best_swap = None
        
        # Sample swaps to save time?
        # 91 edges * 470 non-edges = 42k swaps.
        # Each swap eval:
        #   Remove e1: update adj
        #   Calc delta_add e2: O(deg^2) ~ 36 ops
        #   Add e2: update adj
        #   Restore adj
        #   Total ~ 100 ops.
        # 42k * 100 = 4.2M ops. Very fast in Python? Maybe 1-2 seconds.
        # So we can do full greedy passes.
        
        for e1 in edges_list:
            if time.time() > deadline - 10:
                break
            u1, v1 = e1
            for e2 in non_edges_list:
                u2, v2 = e2
                
                # Simulate swap
                # Remove e1
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Calc delta for adding e2 to new graph
                delta_add = count_added_c4s(u2, v2, adj)
                
                # Restore e1
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                
                # We also need delta_remove for e1 in original graph?
                # No, the change is:
                # C_new = C_old - C4s(e1) + C4s_new(e2)
                # C4s(e1) is the number of C4s containing e1 in the ORIGINAL graph.
                # Wait.
                # If we remove e1, we lose C4s containing e1.
                # If we add e2, we gain C4s containing e2 in the graph WITHOUT e1.
                # So we need C4s(e1 | G) and C4s_new(e2 | G - e1).
                
                # Let's calculate C4s(e1 | G) once outside inner loop?
                # No, it depends on G.
                # But G is fixed for the outer loop iteration.
                
                # So precompute delta_remove for e1?
                # delta_remove = count_added_c4s(u1, v1, adj) # Since adding it back creates the same C4s
                # Wait, count_added_c4s(u1, v1, adj) calculates C4s created by adding (u1,v1) to adj.
                # This is exactly the number of C4s containing (u1,v1) in adj.
                
                # So:
                # delta_remove = count_added_c4s(u1, v1, adj)
                # This is constant for all e2 in the inner loop!
                
                # Let's optimize the loop structure.
                pass # Placeholder, will restructure below

        # Restructured Loop
        best_delta = 0
        best_swap = None
        
        for e1 in edges_list:
            if time.time() > deadline - 10:
                break
            u1, v1 = e1
            delta_remove = count_added_c4s(u1, v1, adj)
            
            # Remove e1 to prepare for delta_add calculations
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            
            for e2 in non_edges_list:
                u2, v2 = e2
                delta_add = count_added_c4s(u2, v2, adj)
                delta = delta_add - delta_remove
                if delta < best_delta:
                    best_delta = delta
                    best_swap = (e1, e2)
            
            # Restore e1
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            
        if best_swap:
            e1, e2 = best_swap
            # Apply swap
            u1, v1 = e1
            u2, v2 = e2
            
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            final_edges.discard(e1)
            final_edges.add(e2)
            
            current_c4s += best_delta
            
            # Update lists
            edges_list.remove(e1)
            edges_list.append(e2)
            non_edges_list.remove(e2)
            non_edges_list.append(e1)
            
            improved = True
            
    # Final polish: try all swaps one more time
    if time.time() < deadline - 5:
        edges_list, non_edges_list = get_edges_non_edges(adj)
        for _ in range(5): # A few passes
            best_delta = 0
            best_swap = None
            for e1 in edges_list:
                if time.time() > deadline - 5:
                    break
                u1, v1 = e1
                delta_remove = count_added_c4s(u1, v1, adj)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                for e2 in non_edges_list:
                    u2, v2 = e2
                    delta_add = count_added_c4s(u2, v2, adj)
                    delta = delta_add - delta_remove
                    if delta < best_delta:
                        best_delta = delta
                        best_swap = (e1, e2)
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
            
            if not best_swap:
                break
                
            e1, e2 = best_swap
            u1, v1 = e1
            u2, v2 = e2
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            final_edges.discard(e1)
            final_edges.add(e2)
            current_c4s += best_delta
            edges_list, non_edges_list = get_edges_non_edges(adj)

    final_edges_list = [list(e) for e in final_edges]
    
    # Ensure exactly m edges
    if len(final_edges_list) < m:
        existing = set(tuple(sorted(e)) for e in final_edges_list)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        for i, j in all_pairs:
            if (i, j) not in existing:
                final_edges_list.append([i, j])
                existing.add((i, j))
                if len(final_edges_list) == m:
                    break
    
    return final_edges_list[:m]

if __name__ == "__main__":
    start = time.time()
    e = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(e)}")
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in e:
        adj[u, v] = 1
        adj[v, u] = 1
    c4_count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 0:
                c = np.sum(adj[u, :] * adj[v, :])
                c4_count += c * (c - 1) // 2
    print(f"C4 Count: {c4_count}")
