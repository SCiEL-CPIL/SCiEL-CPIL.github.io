import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def is_triangle_free(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for u in range(n):
            for v in range(n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if common:
                        return False
        return True

    def greedy_alpha_lower_bound(n, adj):
        """Return size of a greedy independent set (lower bound on alpha)."""
        remaining = (1 << n) - 1
        size = 0
        while remaining:
            best_v = -1
            best_deg = n + 1
            temp = remaining
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(remaining & adj[v]).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            size += 1
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return size

    def find_independent_set_10(n, adj):
        """
        Finds an independent set of size 10.
        Returns the set as a bitmask if found, else None.
        """
        # Quick greedy check
        greedy_size = greedy_alpha_lower_bound(n, adj)
        if greedy_size < 10:
            # It's possible greedy missed it, but for random graphs it's a good indicator.
            # We still need exact check if greedy < 10? No, greedy is a lower bound.
            # If greedy < 10, alpha might still be >= 10.
            # But if greedy >= 10, we definitely have one.
            pass 

        # Exact search
        def backtrack(candidates, size, current_set):
            if size == 10:
                return current_set
            if size + bin(candidates).count('1') < 10:
                return None

            # Heuristic: Pick vertex with max degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= ~(1 << v)
                deg = bin(candidates & adj[v]).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v == -1:
                return None

            candidates &= ~(1 << best_v)

            # Option 1: Include best_v
            new_candidates = candidates & ~adj[best_v]
            result = backtrack(new_candidates, size + 1, current_set | (1 << best_v))
            if result is not None:
                return result

            # Option 2: Exclude best_v
            result = backtrack(candidates, size, current_set)
            if result is not None:
                return result

            return None

        return backtrack((1 << n) - 1, 0, 0)

    def validate(n, edges):
        if not is_triangle_free(n, edges):
            return False
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        if find_independent_set_10(n, adj) is not None:
            return False
        return True

    # Fallback: known valid graph on 26 vertices
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]

    # Try to find larger n, starting from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5:
            break

        found = False
        # Adaptive number of attempts based on remaining time
        remaining_time = deadline - time.time()
        max_attempts = min(1000, max(100, int(remaining_time * 20)))

        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break

            # Generate random maximal triangle-free graph
            adj = [0] * n
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            # Quick density check
            edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
            if edge_count < n * n // 10:
                continue

            # Fast greedy pre-filter: if greedy finds alpha >= 10, skip? 
            # No, we WANT alpha < 10. If greedy >= 10, we have an independent set.
            # But maybe we can repair it?
            
            # Try local search repair
            current_adj = adj[:]
            improved = False
            
            # Limit repair steps to avoid getting stuck
            max_repair_steps = 50
            for _ in range(max_repair_steps):
                if time.time() > deadline - 2:
                    break
                
                # Find independent set of size 10
                iso_set = find_independent_set_10(n, current_adj)
                if iso_set is None:
                    # Success!
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if current_adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
                
                # Try to add an edge within the independent set
                # The independent set is a bitmask
                verts = []
                temp = iso_set
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= ~(1 << v)
                    verts.append(v)
                
                edge_added = False
                random.shuffle(verts) # Randomize order to escape local minima
                
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u = verts[i]
                        v = verts[j]
                        # Check if u, v are not already connected
                        if (current_adj[u] & (1 << v)) == 0:
                            # Check if adding edge creates triangle
                            # Triangle if common neighbor exists
                            if (current_adj[u] & current_adj[v]) == 0:
                                # Add edge
                                current_adj[u] |= (1 << v)
                                current_adj[v] |= (1 << u)
                                edge_added = True
                                break
                    if edge_added:
                        break
                
                if not edge_added:
                    # Stuck: no edge can be added within this independent set
                    # Break out of repair loop for this attempt
                    break
            
            # If we exited repair loop without success, we discard this graph
            # (Or we could keep the best one found, but returning immediately is faster)

    # Local search: try to grow smaller valid graphs by adding edges
    # Try to find a valid graph on n vertices by starting from n-1 and adding a vertex
    for target_n in range(39, 26, -1):
        if time.time() > deadline - 3:
            break

        # Try random construction for target_n
        for attempt in range(200):
            if time.time() > deadline - 2:
                break

            adj = [0] * target_n
            all_edges = [(i, j) for i in range(target_n) for j in range(i + 1, target_n)]
            random.shuffle(all_edges)

            for u, v in all_edges:
                if (adj[u] & adj[v]) == 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)

            edge_count = sum(bin(adj[i]).count('1') for i in range(target_n)) // 2
            if edge_count < target_n * target_n // 10:
                continue

            # Try repair
            current_adj = adj[:]
            max_repair_steps = 50
            for _ in range(max_repair_steps):
                if time.time() > deadline - 2:
                    break
                
                iso_set = find_independent_set_10(target_n, current_adj)
                if iso_set is None:
                    edges = []
                    for u in range(target_n):
                        for v in range(u + 1, target_n):
                            if current_adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': target_n, 'edges': edges}
                
                verts = []
                temp = iso_set
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= ~(1 << v)
                    verts.append(v)
                
                edge_added = False
                random.shuffle(verts)
                
                for i in range(len(verts)):
                    for j in range(i+1, len(verts)):
                        u = verts[i]
                        v = verts[j]
                        if (current_adj[u] & (1 << v)) == 0:
                            if (current_adj[u] & current_adj[v]) == 0:
                                current_adj[u] |= (1 << v)
                                current_adj[v] |= (1 << u)
                                edge_added = True
                                break
                    if edge_added:
                        break
                
                if not edge_added:
                    break

    return {'n': best_n, 'edges': best_edges}
