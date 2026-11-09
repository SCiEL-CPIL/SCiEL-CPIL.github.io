"""Rewritten program for strong edge colouring with local search."""
# EVOLVE-BLOCK-START
import time
import random
import copy

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def build_conflict(edges, adj):
        """Build conflict graph: edges conflict if they share a vertex or are joined by an edge."""
        n = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        conflict = [set() for _ in range(n)]
        for i in range(n):
            u, v = edges[i]
            for j in range(i + 1, n):
                w, x = edges[j]
                if u == w or u == x or v == w or v == x:
                    conflict[i].add(j)
                    conflict[j].add(i)
                elif (u, w) in adj or (u, x) in adj or (v, w) in adj or (v, x) in adj:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict
    
    def count_colours(assign, n_edges):
        """Count distinct colours used."""
        return len(set(assign))
    
    def is_valid_class(assign, idx, n_edges, conflict, edges, adj):
        """Check if all edges with same colour as idx form an induced matching."""
        c = assign[idx]
        for j in conflict[idx]:
            if assign[j] == c:
                return False
        return True
    
    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Build edge-to-vertex and adjacency sets for quick lookup
        edge_adj = set()
        for u, v in edges:
            edge_adj.add((u, v))
            edge_adj.add((v, u))
        
        # Build conflict graph
        conflict = build_conflict(edges, adj)
        
        # Initial greedy colouring
        assign = [0] * m
        for i in range(m):
            used = set()
            for j in conflict[i]:
                if assign[j] != -1:
                    used.add(assign[j])
            c = 0
            while c in used:
                c += 1
            assign[i] = c
        
        best_assign = assign[:]
        best_count = count_colours(assign, m)
        
        # Local search with simulated annealing
        rng = random.Random(seed + 42)
        start_time = time.time()
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        while time.time() - start_time < time_limit and T > min_T:
            # Pick a random edge
            i = rng.randrange(m)
            old_c = assign[i]
            
            # Find available colours
            used = set()
            for j in conflict[i]:
                used.add(assign[j])
            
            # Try to reassign to a colour that reduces count or is random
            if best_count <= 1:
                break
            
            # Strategy: try to move to an existing colour or a new one
            # First try to consolidate: pick a colour used by a non-conflicting edge
            candidates = []
            for j in range(m):
                if j != i and j not in conflict[i]:
                    if assign[j] not in used:
                        candidates.append(assign[j])
            
            if candidates:
                new_c = rng.choice(candidates)
            else:
                # Use smallest available colour
                c = 0
                while c in used:
                    c += 1
                new_c = c
            
            if new_c == old_c:
                T *= cooling
                continue
            
            # Check if this move is valid (no conflicts within new colour class)
            valid = True
            for j in conflict[i]:
                if assign[j] == new_c:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            # Apply move
            assign[i] = new_c
            new_count = count_colours(assign, m)
            
            if new_count < best_count:
                best_count = new_count
                best_assign = assign[:]
            elif new_count == best_count:
                # Accept with some probability to escape plateaus
                if rng.random() < 0.1:
                    best_assign = assign[:]
            else:
                # Worse: accept with probability exp(-delta/T)
                delta = new_count - best_count
                if rng.random() < (2.718281828 ** (-delta / T)):
                    best_assign = assign[:]
            
            T *= cooling
        
        return best_assign
    
    out = []
    deadline = time.time() + 140  # Safety margin before 150s limit
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 5:
            # Fallback to greedy
            n, d, seed = spec
            edges = instance(n, d, seed)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        else:
            out.append(solve_instance(*spec, time_limit=min(remaining - 5, 40)))
    return out
# EVOLVE-BLOCK-END
