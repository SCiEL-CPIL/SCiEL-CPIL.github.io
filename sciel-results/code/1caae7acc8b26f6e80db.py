import time
import random
import numpy as np

def construct():
    """
    Construct a graph with 31 vertices and 91 edges minimizing 4-cycles.
    Strategy:
    1. Construct a C4-free core with 90 edges using the polarity graph of PG(2,5).
    2. Select the best single edge to add to minimize new C4s.
    3. Use local search (edge swaps) with fast C4 counting to minimize total C4s.
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
    
    adj_sets = [set() for _ in range(n)]
    edges_core = []
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 - z1*z2) % 5 == 0:
                edges_core.append([i, j])
                adj_sets[i].add(j)
                adj_sets[j].add(i)
                
    if len(edges_core) != 90:
        # Fallback: Greedy C4-free construction
        adj_sets = [set() for _ in range(n)]
        edges_core = []
        random.seed(42)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges_core) >= 90:
                break
            # Check if adding (u,v) creates a C4
            # A C4 containing new edge (u,v) is u-v-a-b-u where a~v, b~u, a~b
            creates_c4 = False
            for a in adj_sets[v]:
                for b in adj_sets[u]:
                    if a != b and b in adj_sets[a]:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            if not creates_c4:
                edges_core.append([u, v])
                adj_sets[u].add(v)
                adj_sets[v].add(u)
        
        if len(edges_core) < 90:
            for u, v in all_pairs:
                if len(edges_core) >= 90:
                    break
                if v not in adj_sets[u]:
                    edges_core.append([u, v])
                    adj_sets[u].add(v)
                    adj_sets[v].add(u)

    # --- Step 2: Find best single edge to add to minimize C4s ---
    def count_added_c4s(u, v, adj_sets):
        count = 0
        for a in adj_sets[u]:
            for b in adj_sets[v]:
                if a != b and b in adj_sets[a]:
                    count += 1
        return count

    best_edge = None
    best_add = float('inf')
    
    for u in range(n):
        if time.time() > deadline - 5:
            break
        for v in range(u+1, n):
            if v in adj_sets[u]:
                continue
            add = count_added_c4s(u, v, adj_sets)
            if add < best_add:
                best_add = add
                best_edge = [u, v]
    
    if best_edge is None:
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj_sets[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break

    # Apply the best edge
    final_edges = edges_core + [best_edge]
    adj_sets[best_edge[0]].add(best_edge[1])
    adj_sets[best_edge[1]].add(best_edge[0])
    
    # --- Step 3: Fast C4 counting using numpy ---
    def build_adj_matrix(adj_sets):
        A = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in adj_sets[i]:
                A[i, j] = 1
        return A
    
    def count_c4_from_matrix(A):
        """Count C4s: sum over non-adjacent pairs (u,v) of C(common_neighbors(u,v), 2)"""
        # Common neighbors of u,v = (A @ A)[u,v] - A[u,v] (subtract if adjacent)
        # Actually, (A @ A)[u,v] counts paths of length 2 from u to v.
        # If u,v are adjacent, this includes the direct edge path? No, A@A counts 2-step paths.
        # For non-adjacent u,v: common = (A@A)[u,v]
        # For adjacent u,v: common = (A@A)[u,v] - 1 (subtract the triangle path u-w-v where the edge u-v exists? No)
        # Actually, (A@A)[u,v] = number of w such that A[u,w]=1 and A[w,v]=1.
        # This is the number of common neighbors, regardless of whether u,v are adjacent.
        # So common(u,v) = (A@A)[u,v] for all pairs.
        
        A2 = A @ A
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if A[u, v] == 0:  # non-adjacent
                    cn = A2[u, v]
                    total += cn * (cn - 1) // 2
        return total
    
    A = build_adj_matrix(adj_sets)
    current_c4s = count_c4_from_matrix(A)
    
    # --- Step 4: Local search with edge swaps ---
    edges_set = set(tuple(sorted(e)) for e in final_edges)
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def get_non_edges(edges_set):
        return [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
    
    non_edges = get_non_edges(edges_set)
    
    random.seed(123)
    
    # Precompute A2 for fast updates
    A2 = A @ A
    
    def update_A2_add(A, A2, u, v):
        """Add edge (u,v) and update A and A2."""
        A[u, v] = 1
        A[v, u] = 1
        # A2[i,j] changes for all i,j where the new edge affects 2-step paths
        # New 2-step paths through u: i-u-v-j and i-v-u-j
        # A2[i,j] += A[i,u]*A[v,j] + A[i,v]*A[u,j] for all i,j
        # But we need to be careful: A2[i,j] = sum_k A[i,k]*A[k,j]
        # Adding A[u,v]=1 and A[v,u]=1:
        # A2[i,j] += A[i,u]*A[v,j] + A[i,v]*A[u,j]
        # But A[i,u] and A[v,j] include the new edge if i=v or j=u etc.
        # This gets complicated. Let's just recompute A2.
        pass
    
    # For n=31, recomputing A2 is fast: 31x31 matrix multiply
    def recompute_A2(A):
        return A @ A
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 10:
        improved = False
        iteration += 1
        
        # Try random swaps
        num_attempts = min(200, max(50, 200 - iteration * 5))
        for _ in range(num_attempts):
            if time.time() > deadline - 10:
                break
            
            # Pick a random edge to remove
            e1 = random.choice(list(edges_set))
            u1, v1 = e1
            
            # Pick a random non-edge to add
            if not non_edges:
                break
            e2 = random.choice(non_edges)
            u2, v2 = e2
            
            # Simulate swap
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            A2_new = A @ A
            new_c4s = 0
            for u in range(n):
                for v in range(u+1, n):
                    if A[u, v] == 0:
                        cn = A2_new[u, v]
                        new_c4s += cn * (cn - 1) // 2
            
            # Revert
            A[u2, v2] = 0
            A[v2, u2] = 0
            A[u1, v1] = 1
            A[v1, u1] = 1
            
            delta = new_c4s - current_c4s
            if delta < 0:
                # Apply swap
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                
                A[u1, v1] = 0
                A[v1, u1] = 0
                A[u2, v2] = 1
                A[v2, u2] = 1
                
                non_edges = get_non_edges(edges_set)
                current_c4s = new_c4s
                improved = True
                break  # Restart with new state
    
    # --- Step 5: Final greedy improvement - try all single swaps ---
    if time.time() < deadline - 5:
        edges_list = list(edges_set)
        non_edges_list = get_non_edges(edges_set)
        
        best_improvement = 0
        best_swap = None
        
        for e1 in edges_list:
            if time.time() > deadline - 5:
                break
            u1, v1 = e1
            for e2 in non_edges_list:
                u2, v2 = e2
                
                A[u1, v1] = 0
                A[v1, u1] = 0
                A[u2, v2] = 1
                A[v2, u2] = 1
                
                A2_new = A @ A
                new_c4s = 0
                for u in range(n):
                    for v in range(u+1, n):
                        if A[u, v] == 0:
                            cn = A2_new[u, v]
                            new_c4s += cn * (cn - 1) // 2
                
                A[u2, v2] = 0
                A[v2, u2] = 0
                A[u1, v1] = 1
                A[v1, u1] = 1
                
                delta = current_c4s - new_c4s
                if delta > best_improvement:
                    best_improvement = delta
                    best_swap = (e1, e2)
        
        if best_swap:
            e1, e2 = best_swap
            edges_set.discard(e1)
            edges_set.add(e2)
            
            u1, v1 = e1
            u2, v2 = e2
            A[u1, v1] = 0
            A[v1, u1] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            current_c4s -= best_improvement

    # Convert back to list of lists
    final_edges = [list(e) for e in edges_set]
    
    # Ensure exactly m edges
    if len(final_edges) < m:
        existing = set(tuple(sorted(e)) for e in final_edges)
        for i, j in all_pairs:
            if (i, j) not in existing:
                final_edges.append([i, j])
                existing.add((i, j))
                if len(final_edges) == m:
                    break
    
    return final_edges[:m]

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
