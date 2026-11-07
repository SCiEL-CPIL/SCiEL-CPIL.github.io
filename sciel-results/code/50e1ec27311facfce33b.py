"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def is_valid(colour, n, edges, lists):
        if not isinstance(colour, (list, tuple)):
            return False
        if len(colour) != n:
            return False
        for i, c in enumerate(colour):
            if not isinstance(c, int) or isinstance(c, bool):
                return False
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    def backtrack_mrv(n, adj, lists, deadline):
        """Backtracking with MRV, forward checking, and dynamic variable ordering using bitmasks."""
        colour = [-1] * n
        # domain_mask[v] is a bitmask of available colors for vertex v
        # Color c corresponds to bit (1 << c)
        domain_mask = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domain_mask[v] = mask
        
        # Precompute bit counts for 0..31 (since max 5 colors, mask < 32)
        # Actually colors are 0..4, so masks are 0..31.
        # But wait, palette is 5, so colors 0,1,2,3,4.
        # 1<<4 is 16. Max mask is 31.
        
        sys.setrecursionlimit(200000)
        
        # Helper to get list of colors from mask
        def mask_to_colors(m):
            colors = []
            for c in range(5):
                if m & (1 << c):
                    colors.append(c)
            return colors

        # Helper to get popcount
        # Python 3.8+ has int.bit_count()
        def popcount(x):
            return x.bit_count()

        def solve():
            if time.time() > deadline:
                return None

            # Find unassigned vertex with minimum remaining domain (MRV)
            best_v = -1
            best_size = float('inf')
            best_tiebreak = -1
            for v in range(n):
                if colour[v] == -1:
                    sz = popcount(domain_mask[v])
                    if sz < best_size:
                        best_size = sz
                        best_v = v
                        # Tiebreak: higher degree is better? Or just first?
                        # Let's use degree as tiebreak for consistency
                        best_tiebreak = len(adj[v])
                        if sz == 0:
                            return False
                    elif sz == best_size and len(adj[v]) > best_tiebreak:
                        best_v = v
                        best_tiebreak = len(adj[v])

            if best_v == -1:
                return True  # All assigned

            v = best_v
            # Try colours in order of least constraining value
            # We need to iterate through available colors
            # To optimize, we can just iterate through bits set in domain_mask[v]
            
            # Get list of available colors
            avail_colors = mask_to_colors(domain_mask[v])
            
            # Sort by impact (least constraining first)
            # Impact: number of unassigned neighbors that have this color in their domain
            def get_impact(c):
                bit = 1 << c
                impact = 0
                for u in adj[v]:
                    if colour[u] == -1 and (domain_mask[u] & bit):
                        impact += 1
                return impact

            avail_colors.sort(key=get_impact)

            for c in avail_colors:
                colour[v] = c
                bit = 1 << c
                
                # Save old domains for neighbors to undo
                saved_domains = []
                ok = True
                for u in adj[v]:
                    if colour[u] == -1:
                        if domain_mask[u] & bit:
                            saved_domains.append((u, domain_mask[u]))
                            domain_mask[u] &= ~bit
                            if domain_mask[u] == 0:
                                ok = False
                                break
                
                if ok:
                    result = solve()
                    if result is not None:
                        return result

                # Undo
                colour[v] = -1
                for u, old_mask in saved_domains:
                    domain_mask[u] = old_mask

            return False

        result = solve()
        if result:
            return [int(c) for c in colour]
        return None

    def greedy_with_repair(n, adj, edges, lists, deadline, rng_seed):
        """Greedy assignment with random restarts and repair."""
        rng = Random(rng_seed)
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            colour = [-1] * n
            domain = [set(lists[v]) for v in range(n)]
            success = True
            
            for v in order:
                # Filter domain by assigned neighbours
                available = set()
                for c in domain[v]:
                    if any(colour[u] == c for u in adj[v]):
                        continue
                    available.add(c)
                
                if not available:
                    success = False
                    break
                
                # Pick a colour that minimizes impact on unassigned neighbours
                best_c = None
                best_impact = float('inf')
                for c in available:
                    impact = sum(1 for u in adj[v] if colour[u] == -1 and c in domain[u])
                    if impact < best_impact:
                        best_impact = impact
                        best_c = c
                
                colour[v] = best_c
                for u in adj[v]:
                    if colour[u] == -1:
                        domain[u].discard(best_c)
            
            if success:
                # Verify
                valid = True
                for i, c in enumerate(colour):
                    if c not in lists[i]:
                        valid = False
                        break
                if valid:
                    for u, v in edges:
                        if colour[u] == colour[v]:
                            valid = False
                            break
                if valid:
                    return [int(c) for c in colour]
        
        return None

    def local_search(n, adj, edges, lists, deadline, rng_seed):
        """Simulated annealing / hill climbing with random restarts."""
        rng = Random(rng_seed)
        # Start with random valid-list assignment
        colour = [rng.choice(lists[v]) for v in range(n)]
        
        def count_conflicts():
            return sum(1 for u, v in edges if colour[u] == colour[v])
        
        def try_flip(v, c):
            """Try flipping vertex v to colour c, return delta in conflicts."""
            old_c = colour[v]
            if old_c == c:
                return 0
            delta = 0
            for u in adj[v]:
                if colour[u] == old_c:
                    delta += 1  # was conflicting, now not
                if colour[u] == c:
                    delta -= 1  # was not conflicting, now is
            return delta
        
        best_colour = list(colour)
        best_conflicts = count_conflicts()
        
        if best_conflicts == 0:
            return best_colour
        
        current_conflicts = best_conflicts
        T = 1.0
        cooling = 0.9995
        max_iter = 200000
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random vertex in conflict, or random vertex
            in_conflict = [v for v in range(n) if any(colour[v] == colour[u] for u in adj[v])]
            if in_conflict:
                v = rng.choice(in_conflict)
            else:
                v = rng.randrange(n)
            
            old_c = colour[v]
            # Try a different colour from the list
            candidates = [c for c in lists[v] if c != old_c]
            if not candidates:
                continue
            c = rng.choice(candidates)
            
            delta = try_flip(v, c)
            
            if delta <= 0 or rng.random() < (10 ** (-current_conflicts / max(T, 0.01))):
                colour[v] = c
                current_conflicts += delta
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_colour = list(colour)
                    if best_conflicts == 0:
                        return best_colour
            # else: reject, keep old colour
            
            T *= cooling
            if T < 0.001:
                break
        
        return best_colour if best_conflicts == 0 else None

    def solve_instance(n, adj, edges, lists, deadline):
        """Try multiple strategies in sequence."""
        # Strategy 1: Backtracking with MRV
        result = backtrack_mrv(n, adj, lists, deadline)
        if result is not None and is_valid(result, n, edges, lists):
            return result
        
        # Strategy 2: Greedy with multiple restarts
        for seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            result = greedy_with_repair(n, adj, edges, lists, deadline, seed)
            if result is not None and is_valid(result, n, edges, lists):
                return result
        
        # Strategy 3: Local search with multiple restarts
        for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline:
                break
            result = local_search(n, adj, edges, lists, deadline, seed)
            if result is not None and is_valid(result, n, edges, lists):
                return result
        
        # Last resort: return any list-valid assignment (may not be proper)
        return [lists[v][0] for v in range(n)]

    out = []
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        result = solve_instance(n, adj, edges, lists, deadline)
        
        # Ensure types are correct
        result = [int(c) for c in result]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
