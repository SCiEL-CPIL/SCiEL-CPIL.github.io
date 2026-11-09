"""Optimized CSP solver for List Coloring on Planar Bipartite Graphs."""
# EVOLVE-BLOCK-START
import time
import collections

def construct():
    START_TIME = time.time()
    # Leave a safety margin for the evaluator's 120s limit
    DEADLINE = START_TIME + 110.0

    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve_instance(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmask domains
        # Bit i is set if color i is allowed
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        # Check for immediate failures
        for i in range(n):
            if domains[i] == 0:
                return [0] * n

        # Arc Consistency Propagation
        # For binary constraint X != Y, a value c in D_X is removed if D_Y == {c}.
        # This means we only need to propagate when a domain becomes a singleton.
        
        def propagate(domains, adj, n, initial_arcs=None):
            queue = collections.deque()
            if initial_arcs:
                queue.extend(initial_arcs)
            else:
                # Initial AC: check all arcs
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                dv = domains[v]
                if dv == 0:
                    return False
                
                # If v is singleton, remove that value from u
                if dv & (dv - 1) == 0:
                    du = domains[u]
                    if du & dv:
                        new_du = du & ~dv
                        if new_du == 0:
                            return False
                        if new_du != du:
                            domains[u] = new_du
                            # u changed, so u's neighbors (except v) might be affected
                            for x in adj[u]:
                                if x != v:
                                    queue.append((x, u))
            return True

        if not propagate(domains, adj, n):
            return [0] * n

        # Greedy pre-pass for singletons to reduce search space
        color = [-1] * n
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if color[i] == -1 and domains[i] & (domains[i] - 1) == 0:
                    c = (domains[i] & -domains[i]).bit_length() - 1
                    color[i] = c
                    changed = True
                    bit = 1 << c
                    for nb in adj[i]:
                        if color[nb] == -1:
                            if domains[nb] & bit:
                                domains[nb] ^= bit
                                if domains[nb] == 0:
                                    return [0] * n
            if changed:
                if not propagate(domains, adj, n):
                    return [0] * n

        # Backtracking Search
        def backtrack():
            if time.time() > DEADLINE:
                return False
            
            # MRV + Degree Heuristic
            min_size = n + 1
            min_v = -1
            min_deg = -1
            for v in range(n):
                if color[v] == -1:
                    size = domains[v].bit_count()
                    if size == 0:
                        return False
                    if size < min_size or (size == min_size and len(adj[v]) > min_deg):
                        min_size = size
                        min_v = v
                        min_deg = len(adj[v])
                        if size == 1:
                            # Can't get better than size 1, but higher degree might be better?
                            # Usually MRV is primary. If size is 1, it's forced.
                            # We can break early if we find a size 1? 
                            # Not strictly, but size 1 is the minimum possible.
                            # Let's continue to find the one with highest degree if multiple size 1s?
                            # Actually, any size 1 is fine. Breaking early is a safe optimization.
                            break
            
            if min_v == -1:
                return True
            
            dv = domains[min_v]
            
            # Determine color order using Least Constraining Value (LCV) heuristic
            # We prefer colors that leave the most options for neighbors.
            # Score for color c: sum of (domains[nb].bit_count() - 1) for all nb in adj[min_v] where c is in domains[nb]
            # We want to maximize this score.
            
            # Get list of colors in domain
            colors_in_domain = []
            temp_dv = dv
            while temp_dv:
                c = (temp_dv & -temp_dv).bit_length() - 1
                colors_in_domain.append(c)
                temp_dv ^= (1 << c)
            
            if len(colors_in_domain) > 1:
                # Calculate LCV scores
                scores = []
                for c in colors_in_domain:
                    bit = 1 << c
                    score = 0
                    for nb in adj[min_v]:
                        if color[nb] == -1 and (domains[nb] & bit):
                            score += domains[nb].bit_count() - 1
                    scores.append((score, c))
                # Sort by score descending (highest score = least constraining)
                scores.sort(reverse=True)
                # Reorder colors_in_domain
                colors_in_domain = [c for _, c in scores]
            
            # Iterate through ordered colors
            for c in colors_in_domain:
                bit = 1 << c
                
                color[min_v] = c
                
                backup = []
                valid = True
                for nb in adj[min_v]:
                    if color[nb] == -1:
                        if domains[nb] & bit:
                            backup.append((nb, domains[nb]))
                            domains[nb] ^= bit
                            if domains[nb] == 0:
                                valid = False
                                break
                            
                if valid:
                    # Run propagation to enforce AC
                    # Only neighbors of min_v are directly affected, but propagation can cascade.
                    # We pass the arcs (x, nb) where nb is a neighbor of min_v that changed.
                    # Actually, simpler: just run full propagate or targeted.
                    # Targeted: arcs (x, y) where y is a neighbor of min_v.
                    # But if a neighbor becomes singleton, it affects ITS neighbors.
                    # The propagate function handles the queue expansion.
                    # We just need to seed the queue with the arcs that might now be inconsistent.
                    # The arcs (nb, x) for x in adj[nb] are the ones to check.
                    # Specifically, if domains[nb] changed, we need to check (x, nb) for all x in adj[nb].
                    
                    affected_arcs = []
                    for nb, _ in backup:
                        for x in adj[nb]:
                            if x != min_v and color[x] == -1:
                                affected_arcs.append((x, nb))
                    
                    if propagate(domains, adj, n, initial_arcs=affected_arcs):
                        if backtrack():
                            return True
                    # Else: restore and continue loop

                # Restore
                for nb, old_dom in backup:
                    domains[nb] = old_dom
                color[min_v] = -1
            
            return False

        if backtrack():
            # Ensure all are assigned and valid
            result = []
            for i in range(n):
                if color[i] == -1:
                    return [0] * n
                # Verify color is in original list
                # Note: domains[i] might have been reduced, but color[i] must be in original lists[i]
                # And color[i] must not equal any neighbor's color.
                if not (lists[i].count(color[i]) > 0):
                    return [0] * n
                for nb in adj[i]:
                    if color[nb] == color[i]:
                        return [0] * n
                result.append(int(color[i]))
            return result
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        
        # Final safety check: ensure types are int and length is correct
        if len(sol) == n:
            out.append([int(c) for c in sol])
        else:
            out.append([0] * n)
            
    return out
# EVOLVE-BLOCK-END
