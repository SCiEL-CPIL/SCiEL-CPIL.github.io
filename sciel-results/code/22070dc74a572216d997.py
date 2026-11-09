import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_edges = 0
    best_adj = None

    for restart in range(200):
        if time.time() > deadline:
            break

        adj = [0] * n
        current_edges = 0

        # Greedy phase
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)

        def can_add(u, v, adj):
            # Check if path of length 5 exists from u to v
            # Use bitsets
            mask_uv = ~((1 << u) | (1 << v)) & ((1 << n) - 1)
            
            s = 1 << u
            # Step 1
            s = adj[u] & mask_uv
            if s == 0: return True
            
            # Step 2
            next_s = 0
            temp = s
            while temp:
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1)
                next_s |= adj[idx]
                temp -= lsb
            s = next_s & mask_uv
            if s == 0: return True

            # Step 3
            next_s = 0
            temp = s
            while temp:
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1)
                next_s |= adj[idx]
                temp -= lsb
            s = next_s & mask_uv
            if s == 0: return True

            # Step 4
            next_s = 0
            temp = s
            while temp:
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1)
                next_s |= adj[idx]
                temp -= lsb
            s = next_s & mask_uv
            if s == 0: return True

            # Step 5
            next_s = 0
            temp = s
            while temp:
                lsb = temp & -temp
                idx = (lsb.bit_length() - 1)
                next_s |= adj[idx]
                temp -= lsb
            s = next_s & mask_uv

            if (s >> v) & 1:
                return False
            return True

        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not (adj[u] >> v) & 1:
                if can_add(u, v, adj):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges += 1

        # Local Search: Add single edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if can_add(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            current_edges += 1
                            improved = True
                            if time.time() > deadline:
                                break
                if time.time() > deadline:
                    break

        # Kick: Remove 2 random edges and refill
        # This helps escape local optima
        if time.time() < deadline - 5:
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
            if len(edges_list) >= 2:
                random.shuffle(edges_list)
                e1 = edges_list[0]
                e2 = edges_list[1]
                
                adj[e1[0]] &= ~(1 << e1[1])
                adj[e1[1]] &= ~(1 << e1[0])
                adj[e2[0]] &= ~(1 << e2[1])
                adj[e2[1]] &= ~(1 << e2[0])
                current_edges -= 2

                # Refill
                for u, v in edges_to_try:
                    if time.time() > deadline:
                        break
                    if not (adj[u] >> v) & 1:
                        if can_add(u, v, adj):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            current_edges += 1
                
                # Try single additions again
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    for u in range(n):
                        for v in range(u+1, n):
                            if not (adj[u] >> v) & 1:
                                if can_add(u, v, adj):
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
                                    current_edges += 1
                                    improved = True
                                    if time.time() > deadline:
                                        break
                        if time.time() > deadline:
                            break

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    edges = []
    if best_adj:
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
    else:
        # Fallback to seed
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]

    return edges
