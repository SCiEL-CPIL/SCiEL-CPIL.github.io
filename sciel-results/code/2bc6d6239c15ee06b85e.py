"""Combined program: Structural backtracking with diversified greedy fallback."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque

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

    def greedy_diversified(n, adj, lists, deadline, rng):
        """
        Inspired by 'inspiration 1': Try multiple greedy orderings and random repairs.
        This is fast and might solve easy instances instantly.
        """
        # Different ordering strategies
        orders = []
        # 1. Degree descending
        orders.append(sorted(range(n), key=lambda v: -len(adj[v])))
        # 2. BFS order
        seen = [False] * n
        q = deque([0])
        seen[0] = True
        bfs_order = []
        while q:
            u = q.popleft()
            bfs_order.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
        for v in range(n):
            if not seen[v]:
                bfs_order.append(v)
        orders.append(bfs_order)
        
        # 3. Random shuffle
        rand_order = list(range(n))
        rng.shuffle(rand_order)
        orders.append(rand_order)

        for order in orders:
            colour = [-1] * n
            valid = True
            for v in order:
                # Find available colors
                taken = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        taken.add(colour[u])
                
                avail = [c for c in lists[v] if c not in taken]
                if avail:
                    colour[v] = avail[0] 
                else:
                    valid = False
                    break
            
            if valid:
                # Quick verification
                ok = True
                for v in range(n):
                    if colour[v] not in lists[v]:
                        ok = False
                        break
                    for u in adj[v]:
                        if u > v and colour[u] == colour[v]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return colour
        
        return None

    def bipartite_solve(n, adj, edges, lists, deadline):
        """
        Current program's strong solver: Exploit bipartite structure.
        Assigns one side, then checks the other.
        """
        # Get bipartition
        color = [-1] * n
        parts = [[], []]
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            parts[0].append(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        parts[color[v]].append(v)
                        q.append(v)
        
        # Choose smaller part to enumerate/backtrack on
        if len(parts[0]) <= len(parts[1]):
            small, large = parts[0], parts[1]
        else:
            small, large = parts[1], parts[0]
        
        small_set = set(small)
        large_set = set(large)
        
        # Precompute neighbors for speed
        large_neighbors = {v: [u for u in adj[v] if u in small_set] for v in large}
        small_neighbors = {v: [u for u in adj[v] if u in large_set] for v in small}
        
        colour = [-1] * n
        sys.setrecursionlimit(200000)
        
        def solve_small(idx):
            if time.time() > deadline:
                return False
            if idx == len(small):
                # Assign large vertices greedily
                for v in large:
                    used_colors = set()
                    for u in large_neighbors[v]:
                        used_colors.add(colour[u])
                    available = [c for c in lists[v] if c not in used_colors]
                    if not available:
                        return False
                    colour[v] = available[0]
                return True
            
            v = small[idx]
            # Heuristic: try colors that leave more options for neighbors
            # Simple version: iterate lists[v]
            for c in lists[v]:
                colour[v] = c
                # Check feasibility for large neighbors
                feasible = True
                for u in small_neighbors[v]:
                    used = set()
                    for w in large_neighbors[u]:
                        if colour[w] != -1:
                            used.add(colour[w])
                    if all(c2 in used for c2 in lists[u]):
                        feasible = False
                        break
                if feasible:
                    if solve_small(idx + 1):
                        return True
                colour[v] = -1
            return False
        
        if solve_small(0):
            return [int(c) for c in colour]
        return None

    def backtrack_mrv(n, adj, lists, deadline):
        """
        Fallback: General backtracking with MRV.
        """
        colour = [-1] * n
        domain = [set(lists[v]) for v in range(n)]
        constraint_count = [0] * n
        sys.setrecursionlimit(200000)

        def solve():
            if time.time() > deadline:
                return False

            best_v = -1
            best_size = float('inf')
            best_tiebreak = -1
            for v in range(n):
                if colour[v] == -1:
                    sz = len(domain[v])
                    if sz < best_size:
                        best_size = sz
                        best_v = v
                        best_tiebreak = constraint_count[v]
                        if sz == 0:
                            return False
                    elif sz == best_size and constraint_count[v] > best_tiebreak:
                        best_v = v
                        best_tiebreak = constraint_count[v]

            if best_v == -1:
                return True

            v = best_v
            # Order colors by least constraining value
            colour_order = sorted(
                domain[v],
                key=lambda c: sum(1 for u in adj[v] if colour[u] == -1 and c in domain[u]),
                reverse=False
            )

            for c in colour_order:
                colour[v] = c
                constraint_count[v] = len(adj[v])
                saved = []
                ok = True
                for u in adj[v]:
                    if colour[u] == -1 and c in domain[u]:
                        domain[u].discard(c)
                        saved.append((u, c))
                        constraint_count[u] += 1
                        if len(domain[u]) == 0:
                            ok = False
                            break

                if ok:
                    if solve():
                        return True

                colour[v] = -1
                constraint_count[v] = 0
                for u, c2 in saved:
                    domain[u].add(c2)
                    constraint_count[u] -= 1

            return False

        if solve():
            return [int(c) for c in colour]
        return None

    def local_search(n, adj, edges, lists, deadline, rng_seed):
        """
        Last resort: Simulated Annealing with proper conflict tracking.
        """
        rng = Random(rng_seed)
        colour = [rng.choice(lists[v]) for v in range(n)]
        
        # Build edge list for fast conflict counting
        edge_list = [(u, v) for u, v in edges]
        
        conflict_count = [0] * n
        total_conflicts = 0
        for u, v in edge_list:
            if colour[u] == colour[v]:
                conflict_count[u] += 1
                conflict_count[v] += 1
                total_conflicts += 1
        
        if total_conflicts == 0:
            return list(colour)
        
        best_colour = list(colour)
        best_conflicts = total_conflicts
        current_conflicts = total_conflicts
        T = 1.0
        cooling = 0.999
        start_t = time.time()
        
        while True:
            if time.time() > deadline:
                break
            
            conflicted = [v for v in range(n) if conflict_count[v] > 0]
            if conflicted:
                v = rng.choice(conflicted)
            else:
                v = rng.randrange(n)
            
            old_c = colour[v]
            candidates = [c for c in lists[v] if c != old_c]
            if not candidates:
                continue
            c = rng.choice(candidates)
            
            # Compute delta: how many conflicts are removed/added
            delta = 0
            for u in adj[v]:
                if colour[u] == old_c:
                    delta -= 1  # removing a conflict
                if colour[u] == c:
                    delta += 1  # adding a conflict
            
            if delta <= 0 or rng.random() < (10 ** (-current_conflicts / max(T, 0.01))):
                colour[v] = c
                for u in adj[v]:
                    if colour[u] == old_c:
                        conflict_count[u] -= 1
                        conflict_count[v] -= 1
                    if colour[u] == c:
                        conflict_count[u] += 1
                        conflict_count[v] += 1
                current_conflicts += delta
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_colour = list(colour)
                    if best_conflicts == 0:
                        return best_colour
            
            T *= cooling
            if T < 0.001:
                break
        
        return best_colour if best_conflicts == 0 else None

    def solve_instance(n, adj, edges, lists, deadline):
        """
        Strategy:
        1. Quick diversified greedy (Inspiration)
        2. Bipartite structural backtracking (Current)
        3. General MRV backtracking (Current)
        4. Local Search (Current)
        """
        rng = Random(123) # Fixed seed for greedy attempts to be deterministic
        
        # 1. Fast Greedy
        result = greedy_diversified(n, adj, lists, deadline, rng)
        if result is not None:
            return result
            
        # 2. Bipartite Solver
        result = bipartite_solve(n, adj, edges, lists, deadline)
        if result is not None:
            return result
        
        # 3. General Backtracking
        result = backtrack_mrv(n, adj, lists, deadline)
        if result is not None:
            return result
        
        # 4. Local Search
        for seed in [42, 123, 456, 789, 101112]:
            if time.time() > deadline:
                break
            result = local_search(n, adj, edges, lists, deadline, seed)
            if result is not None:
                return result
        
        # Fallback: Return something valid type-wise (will score 0 for this instance if invalid)
        return [lists[v][0] for v in range(n)]

    out = []
    start_time = time.time()
    # Allocate time per instance: harder instances (more splits) get more time
    # Total budget: 145s (5s safety margin before 150s kill)
    total_budget = 145.0
    # Roughly proportional to complexity: n = 8 + 4*splits
    complexities = [8 + 4*s for s, _ in ((20, 701), (40, 702), (60, 703), (100, 704))]
    total_complexity = sum(complexities)
    
    for i, (splits, seed) in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        # Give each instance time proportional to its complexity, but ensure minimum time
        instance_budget = total_budget * complexities[i] / total_complexity
        # But also ensure we don't exceed remaining time
        remaining = total_budget - (time.time() - start_time)
        instance_deadline = min(start_time + remaining, time.time() + max(instance_budget, 5.0))
        
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        result = solve_instance(n, adj, edges, lists, instance_deadline)
        
        # Ensure types are strictly Python int
        result = [int(c) for c in result]
        out.append(result)
        
    return out
# EVOLVE-BLOCK-END
