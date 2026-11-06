import time
import numpy as np
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    n = 31
    q = 5
    target_edges = 91
    
    # Generate points of PG(2, q) using homogeneous coordinates
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, q)
                    pt = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, -1, q)
                    pt = (0, 1, (z * inv_y) % q)
                else:
                    pt = (0, 0, 1)
                
                if pt not in points:
                    points.append(pt)
    
    if len(points) != n:
        points = []
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        
        if len(points) != n:
            random.seed(42)
            edges = []
            edge_set = set()
            while len(edges) < target_edges:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    edge_set.add((u, v))
                    edges.append([u, v])
            return edges
    
    # Build adjacency using polarity
    neighbors = [set() for _ in range(n)]
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]) % q
            if dot == 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
                edges.append([i, j])
    
    current_edges = len(edges)
    
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in neighbors[i]:
                non_edges.append((i, j))
    
    edges_to_add = target_edges - current_edges
    
    if edges_to_add <= 0:
        return edges[:target_edges]
    
    def count_new_c4s(u, v):
        count = 0
        for a in neighbors[u]:
            if a == v:
                continue
            for b in neighbors[v]:
                if b == u or b == a:
                    continue
                if b in neighbors[a]:
                    count += 1
        return count
    
    best_edge = None
    min_cost = float('inf')
    
    for u, v in non_edges:
        if time.time() - start_time > DEADLINE - 10:
            break
        cost = count_new_c4s(u, v)
        if cost < min_cost:
            min_cost = cost
            best_edge = (u, v)
    
    if best_edge is None:
        if non_edges:
            best_edge = non_edges[0]
        else:
            return edges[:target_edges]
    
    u, v = best_edge
    edges.append([u, v])
    neighbors[u].add(v)
    neighbors[v].add(u)
    
    # Correct C4 count: sum over all unordered pairs {u,v} of C(common(u,v), 2) / 2
    # Each C4 is counted exactly twice (once for each pair of opposite vertices)
    def count_total_c4s(nbrs):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = len(nbrs[i] & nbrs[j])
                total += common * (common - 1) // 2
        return total // 2
    
    initial_c4 = count_total_c4s(neighbors)
    
    random.seed(42)
    
    best_c4 = initial_c4
    best_edges = [e[:] for e in edges]
    best_neighbors = [set(nb) for nb in neighbors]
    
    current_c4 = initial_c4
    cur_neighbors = [set(nb) for nb in neighbors]
    cur_edges = [e[:] for e in edges]
    
    # Efficient delta computation for edge swap (remove (a,b), add (c,d))
    def compute_delta_c4(nbrs, a, b, c, d):
        """Compute change in C4 count when removing edge (a,b) and adding edge (c,d)."""
        # The C4 count = (1/2) * sum_{u<v} C(|N(u) ∩ N(v)|, 2)
        # When we change adjacency, only pairs involving a, b, c, d are affected
        # Actually, common neighbors change for any pair where one endpoint's neighbor set changes
        
        # Pairs whose common neighbor count changes:
        # Removing (a,b): a's neighbors lose b, b's neighbors lose a
        #   -> common(a, w) changes for all w (decreases by 1 if b was in N(w) ∩ N(a) after removal... 
        #   Actually: common(a,w) = |N(a) ∩ N(w)|. After removing a-b, N(a) loses b.
        #   So common(a,w) decreases by 1 iff b ∈ N(w) (i.e., w ∈ N(b)) AND b was in N(a)∩N(w)
        #   Since b was in N(a) (edge existed), b ∈ N(a)∩N(w) iff b ∈ N(w) iff w ∈ N(b)
        #   Similarly common(b,w) decreases by 1 iff w ∈ N(a)
        #   Also common(a,b) itself: |N(a)∩N(b)|. N(a) loses b, N(b) loses a. 
        #   b was not in N(a)∩N(b) (no self-loops), a was not in N(a)∩N(b).
        #   So common(a,b) is unchanged!
        #   Wait: N(a) contains b before removal. N(b) contains a before removal.
        #   N(a)∩N(b): does it contain b? b∈N(a) yes, b∈N(b)? No (no self-loop). So b∉N(a)∩N(b).
        #   Does it contain a? a∈N(a)? No. So a∉N(a)∩N(b).
        #   So removing a-b doesn't change N(a)∩N(b). Correct.
        
        # Adding (c,d): c's neighbors gain d, d's neighbors gain c
        #   common(c,w) increases by 1 iff d ∈ N(w) iff w ∈ N(d)
        #   common(d,w) increases by 1 iff c ∈ N(w) iff w ∈ N(c)
        #   common(c,d) unchanged (similar argument)
        
        # But we need to be careful: the changes interact if {a,b} and {c,d} share vertices
        # For simplicity, compute the full effect by tracking which pairs change
        
        # Affected pairs: all pairs involving a, b, c, or d
        affected = set()
        for x in [a, b, c, d]:
            for y in range(n):
                if x != y:
                    pair = (min(x, y), max(x, y))
                    affected.add(pair)
        
        delta = 0
        for (u, v) in affected:
            # Compute old common count
            old_common = len(nbrs[u] & nbrs[v])
            old_val = old_common * (old_common - 1) // 2
            
            # Compute new common count after the swap
            new_Nu = set(nbrs[u])
            new_Nv = set(nbrs[v])
            
            # Apply removal of (a,b)
            if a in new_Nu and b in new_Nu:
                # a is u or b is u... let me think differently
                pass
            
            # This is getting complicated. Let me just do it directly.
            # Build modified neighbor sets for u and v
            new_Nu = set(nbrs[u])
            new_Nv = set(nbrs[v])
            
            # Remove edge (a,b) effect on N(u) and N(v)
            if u == a:
                new_Nu.discard(b)
            if u == b:
                new_Nu.discard(a)
            if v == a:
                new_Nv.discard(b)
            if v == b:
                new_Nv.discard(a)
            
            # Add edge (c,d) effect on N(u) and N(v)
            if u == c:
                new_Nu.add(d)
            if u == d:
                new_Nu.add(c)
            if v == c:
                new_Nv.add(d)
            if v == d:
                new_Nv.add(c)
            
            new_common = len(new_Nu & new_Nv)
            new_val = new_common * (new_common - 1) // 2
            
            delta += (new_val - old_val)
        
        return delta // 2  # Because we sum over unordered pairs, each C4 counted twice
    
    # Simulated annealing with efficient delta
    T = 100.0
    cooling = 0.999
    T_min = 0.1
    
    iter_count = 0
    while time.time() - start_time < DEADLINE and T > T_min:
        iter_count += 1
        
        # Pick random edge to remove
        edge_idx = random.randint(0, len(cur_edges) - 1)
        a, b = cur_edges[edge_idx]
        
        # Pick random non-edge to add
        found_non_edge = False
        attempts = 0
        while attempts < 50:
            c, d = random.sample(range(n), 2)
            if c > d:
                c, d = d, c
            if d not in cur_neighbors[c]:
                found_non_edge = True
                break
            attempts += 1
        
        if not found_non_edge:
            T *= cooling
            continue
        
        if (c == a and d == b) or (c == b and d == a):
            T *= cooling
            continue
        
        delta = compute_delta_c4(cur_neighbors, a, b, c, d)
        
        if delta < 0 or (delta == 0 and random.random() < 0.1) or (delta > 0 and random.random() < np.exp(-delta / T)):
            # Accept the move
            cur_neighbors[a].discard(b)
            cur_neighbors[b].discard(a)
            cur_neighbors[c].add(d)
            cur_neighbors[d].add(c)
            cur_edges[edge_idx] = [c, d]
            current_c4 += delta
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = [e[:] for e in cur_edges]
                best_neighbors = [set(nb) for nb in cur_neighbors]
        
        T *= cooling
    
    # Final polish: try all possible single-edge swaps greedily
    if time.time() - start_time < DEADLINE - 5:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for i in range(len(best_edges)):
                if time.time() - start_time > DEADLINE - 2:
                    break
                a, b = best_edges[i]
                for j in range(len(best_edges)):
                    if i == j:
                        continue
                    c, d = best_edges[j]
                    # Try removing (a,b) and adding a non-edge
                    # Find non-edges
                    for k in range(n):
                        if time.time() - start_time > DEADLINE - 2:
                            break
                        for l in range(k + 1, n):
                            if l in best_neighbors[k]:
                                continue
                            if (k == a and l == b) or (k == b and l == a):
                                continue
                            # Try swap
                            delta = compute_delta_c4(best_neighbors, a, b, k, l)
                            if delta < 0:
                                best_neighbors[a].discard(b)
                                best_neighbors[b].discard(a)
                                best_neighbors[k].add(l)
                                best_neighbors[l].add(k)
                                best_edges[i] = [k, l]
                                best_c4 += delta
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    
    if best_c4 < initial_c4:
        return best_edges
    else:
        return edges
