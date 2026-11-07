"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def ac3(domains, adj):
        """Enforce arc consistency using bitmasks."""
        q = deque()
        for u in range(len(domains)):
            for v in adj[u]:
                q.append((u, v))
        
        while q:
            u, v = q.popleft()
            du = domains[u]
            dv = domains[v]
            
            # Remove colors from du that are not supported in dv
            # c is unsupported in dv if dv == {c}
            new_du = du
            temp = du
            while temp:
                lsb = temp & -temp
                if dv == lsb:
                    new_du &= ~lsb
                temp ^= lsb
            
            if new_du != du:
                if new_du == 0:
                    return False
                domains[u] = new_du
                for w in adj[u]:
                    if w != v:
                        q.append((w, u))
        
        return True

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains as bitmasks
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask

        # Preprocess with Arc Consistency
        if not ac3(domains, adj):
            # Should not happen per Alon-Tarsi, but fallback
            return [int(l[0]) for l in lists]

        colour = [-1] * n
        current_domains = domains[:]
        unassigned = set(range(n))
        degrees = [len(adj[i]) for i in range(n)]

        sys.setrecursionlimit(10000)

        def backtrack():
            if time.time() > deadline:
                return False

            if not unassigned:
                return True

            # MRV + degree tie-breaking
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            
            for v in unassigned:
                cnt = current_domains[v].bit_count()
                if cnt == 0:
                    return False
                if cnt < best_count or (cnt == best_count and degrees[v] > best_deg):
                    best_count = cnt
                    best_v = v
                    best_deg = degrees[v]
                    if cnt == 1:
                        break

            if best_v == -1:
                return False

            v = best_v
            d = current_domains[v]
            
            # Extract candidates and sort by LCV
            candidates = []
            temp_d = d
            while temp_d:
                lsb = temp_d & -temp_d
                c = lsb.bit_length() - 1
                candidates.append(c)
                temp_d ^= lsb

            if len(candidates) > 1:
                scores = []
                for c in candidates:
                    mask = 1 << c
                    removals = 0
                    for u in adj[v]:
                        if u in unassigned and (current_domains[u] & mask):
                            removals += 1
                    scores.append(removals)
                candidates = [c for c, s in sorted(zip(candidates, scores), key=lambda x: x[1])]

            for c in candidates:
                colour[v] = c
                unassigned.remove(v)
                
                conflict = False
                saved_neighbors = []
                c_mask = 1 << c
                
                for u in adj[v]:
                    if u in unassigned:
                        if current_domains[u] & c_mask:
                            saved_neighbors.append(u)
                            current_domains[u] &= ~c_mask
                            if current_domains[u] == 0:
                                conflict = True
                                break

                if not conflict:
                    if backtrack():
                        return True

                # Undo
                colour[v] = -1
                unassigned.add(v)
                for u in saved_neighbors:
                    current_domains[u] |= c_mask

            return False

        success = backtrack()
        
        if success:
            return [int(c) for c in colour]
        else:
            # Fallback: greedy coloring
            col = [-1] * n
            order = sorted(range(n), key=lambda u: -degrees[u])
            for v in order:
                taken = {col[u] for u in adj[v] if col[u] != -1}
                free = [c for c in lists[v] if c not in taken]
                col[v] = int(free[0]) if free else int(lists[v][0])
            return col

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 2.0:
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        out.append(colouring)
        
    # Ensure we have exactly 4 results with correct lengths
    while len(out) < 4:
        # Generate the missing instance to get correct n
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        # Quick greedy fallback
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        col = [-1] * n
        for v in range(n):
            taken = {col[u] for u in adj[v] if col[u] != -1}
            free = [c for c in lists[v] if c not in taken]
            col[v] = int(free[0]) if free else int(lists[v][0])
        out.append(col)

    return out
# EVOLVE-BLOCK-END
