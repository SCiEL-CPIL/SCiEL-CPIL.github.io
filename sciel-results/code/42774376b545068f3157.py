import random
import time

def construct():
    # We aim for n=40, then 39.
    
    def get_independent_set(adj, n, k):
        """
        Returns a list of vertices forming an independent set of size k, or None.
        Uses B&B.
        """
        mask = (1 << n) - 1
        # Complement graph H
        h_adj = [0] * n
        for i in range(n):
            h_adj[i] = (~adj[i]) & mask & ~(1 << i)
            
        found_set = []
        
        def dfs(candidates, size, current_set):
            if size == k:
                found_set.extend(current_set)
                return True
            if not candidates:
                return False
            if size + candidates.bit_count() < k:
                return False
                
            # Heuristic: pick vertex with max degree in H within candidates
            best_v = -1
            max_deg = -1
            c = candidates
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                deg = (h_adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                c ^= lsb
                
            if best_v == -1:
                return False
                
            # Include best_v
            current_set.append(best_v)
            new_candidates = candidates & h_adj[best_v]
            if dfs(new_candidates, size + 1, current_set):
                return True
            current_set.pop()
            
            # Exclude best_v
            candidates &= ~(1 << best_v)
            if dfs(candidates, size, current_set):
                return True
                
            return False
            
        if dfs((1 << n) - 1, 0, []):
            return found_set
        return None

    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & -neighbors
                j = lsb.bit_length() - 1
                if j > i:
                    common = adj[i] & adj[j]
                    common &= ~((1 << (j + 1)) - 1)
                    if common:
                        return False
                neighbors ^= lsb
        return True

    def remove_random_triangle(adj, n):
        # Find one triangle and remove a random edge from it
        # To be efficient, we can just find the first one
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & -neighbors
                j = lsb.bit_length() - 1
                if j > i:
                    common = adj[i] & adj[j]
                    common &= ~((1 << (j + 1)) - 1)
                    if common:
                        # Triangle (i, j, k)
                        k_bit = common & -common
                        k = k_bit.bit_length() - 1
                        # Remove random edge: (i,j), (j,k), (i,k)
                        r = random.randint(0, 2)
                        if r == 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                        elif r == 1:
                            adj[j] &= ~(1 << k)
                            adj[k] &= ~(1 << j)
                        else:
                            adj[i] &= ~(1 << k)
                            adj[k] &= ~(1 << i)
                        return True
                neighbors ^= lsb
        return False

    def add_edge_in_set(adj, n, vertices):
        # Add a random edge between two vertices in the set
        # Check if it creates a triangle
        if len(vertices) < 2:
            return False
            
        # Shuffle to pick random pair
        vs = vertices[:]
        random.shuffle(vs)
        u, v = vs[0], vs[1]
        
        if adj[u] & (1 << v):
            return False
            
        # Check triangle
        common = adj[u] & adj[v]
        if common:
            return False
            
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def random_graph(n):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    for target_n in [40, 39]:
        start_time = time.time()
        limit = 40 if target_n == 40 else 80
        
        # Try multiple random restarts
        for restart in range(20):
            if time.time() - start_time > limit:
                break
                
            adj = random_graph(target_n)
            
            # Phase 1: Remove triangles
            # We do this in a loop until no triangles or timeout
            for _ in range(5000):
                if time.time() - start_time > limit:
                    break
                if is_triangle_free(adj, target_n):
                    break
                if not remove_random_triangle(adj, target_n):
                    break
            
            if not is_triangle_free(adj, target_n):
                continue
                
            # Phase 2: Reduce independence number
            for _ in range(10000):
                if time.time() - start_time > limit:
                    break
                    
                # Check for independent set
                indep = get_independent_set(adj, target_n, 10)
                if indep is None:
                    # Valid!
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': target_n, 'edges': edges}
                    
                # Try to break it
                success = False
                for _ in range(10):
                    if add_edge_in_set(adj, target_n, indep):
                        success = True
                        break
                if not success:
                    # Stuck, restart
                    break

    # Fallback
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 3 == 1]}
