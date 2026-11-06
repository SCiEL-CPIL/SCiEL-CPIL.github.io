"""Sparse pancyclic graph on 40 vertices - simulated annealing search."""
import random
import time
import math

N = 40
DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    random.seed(42)
    start_time = time.time()
    
    # Build adjacency as bitmasks for speed
    # adj[i] is a 40-bit integer with bit j set iff edge (i,j) exists
    adj = [0] * N
    edges = set()
    
    def add_edge(u, v):
        if u == v: return
        if u > v: u, v = v, u
        if (u, v) not in edges:
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) in edges:
            edges.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    def has_edge(u, v):
        return (adj[u] >> v) & 1
    
    # --- Build initial solution: a "spoke + path" structure ---
    # Use vertex 0 as hub, connect to all others (39 edges)
    # Plus a path through vertices 1..39 (38 edges) = 77 edges total
    # This gives cycles of all lengths 3..40
    
    for i in range(1, N):
        add_edge(0, i)
    for i in range(1, N - 1):
        add_edge(i, i + 1)
    
    # Cycles: for length k, use 0, 1, 2, ..., k-1 (with edge (k-1, 0) via hub)
    # Actually: cycle of length k = [0, 1, 2, ..., k-1] 
    # Edges needed: (0,1),(1,2),...,(k-2,k-1),(k-1,0)
    # All exist! (0,i) for all i, and (i,i+1) for i=1..38
    # For k=40: need edge (39,0) which exists.
    
    def get_cycles():
        cycles = []
        for k in range(3, N + 1):
            cycles.append(list(range(k)))  # [0, 1, 2, ..., k-1]
        return cycles
    
    def validate_cycles(cycles):
        for cyc in cycles:
            k = len(cyc)
            for i in range(k):
                u = cyc[i]
                v = cyc[(i + 1) % k]
                if not has_edge(u, v):
                    return False
        return True
    
    def cycle_cost(cycles):
        """Total edges needed by all cycles."""
        needed = set()
        for cyc in cycles:
            k = len(cyc)
            for i in range(k):
                u = cyc[i]
                v = cyc[(i + 1) % k]
                if u > v: u, v = v, u
                needed.add((u, v))
        return len(needed)
    
    # Objective: minimize |edges| - 40
    # Current: 77 edges, score = 77 - 40 = 37
    
    best_edges = set(edges)
    best_cycles = get_cycles()
    best_score = len(best_edges) - 40
    
    # Simulated annealing
    # State: adjacency matrix
    # Moves: add/remove edges, or restructure cycles
    # Key insight: we can try to find cycles that share more edges
    
    # Strategy: Try to find a set of cycles (one per length 3..40) that
    # collectively use as few distinct edges as possible, then add any missing edges.
    
    # Alternative: directly minimize edges while keeping all cycles valid.
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    step = 0
    while time.time() - start_time < DEADLINE:
        step += 1
        
        # Random move
        r = random.random()
        
        if r < 0.4:
            # Try to remove a random edge
            if len(edges) > 40:  # keep at least 40
                e = random.choice(list(edges))
                remove_edge(*e)
                # Check if all cycles still valid
                if validate_cycles(best_cycles):
                    new_score = len(edges) - 40
                    if new_score < best_score:
                        best_edges = set(edges)
                        best_score = new_score
                else:
                    add_edge(*e)
        elif r < 0.6:
            # Try to add a random edge (might help enable better cycles later)
            u = random.randint(0, N - 1)
            v = random.randint(0, N - 1)
            if u != v:
                add_edge(u, v)
        elif r < 0.8:
            # Try to restructure: find a shorter path between two vertices
            # and use it in a cycle
            # Pick a random cycle length and try to find an alternative cycle
            k = random.randint(3, N)
            # Try random permutations of k vertices
            for attempt in range(5):
                if time.time() - start_time > DEADLINE:
                    break
                verts = random.sample(range(N), k)
                # Check if this forms a valid cycle with current edges
                valid = True
                for i in range(k):
                    u = verts[i]
                    v = verts[(i + 1) % k]
                    if not has_edge(u, v):
                        valid = False
                        break
                if valid:
                    # This cycle is valid; update best_cycles for this length
                    # (0-indexed: length k corresponds to index k-3 in cycles list)
                    idx = k - 3
                    old_cycle = best_cycles[idx]
                    best_cycles[idx] = verts
                    # Check if we can now remove some edges
                    # Compute which edges are still needed
                    needed = set()
                    for cyc in best_cycles:
                        kl = len(cyc)
                        for i in range(kl):
                            u = cyc[i]
                            v = cyc[(i + 1) % kl]
                            if u > v: u, v = v, u
                            needed.add((u, v))
                    # Remove edges not in needed
                    for e in list(edges):
                        if e not in needed:
                            remove_edge(*e)
                    if validate_cycles(best_cycles):
                        new_score = len(edges) - 40
                        if new_score < best_score:
                            best_edges = set(edges)
                            best_score = new_score
                        else:
                            # Revert
                            for e in best_edges:
                                if e not in edges:
                                    add_edge(*e)
                            best_cycles[idx] = old_cycle
                            break
                    else:
                        best_cycles[idx] = old_cycle
                        break
        else:
            # Try to find a cycle of a specific length using DFS
            k = random.randint(3, N)
            # Simple random walk to find a cycle of length k
            for attempt in range(3):
                if time.time() - start_time > DEADLINE:
                    break
                # Random start
                start = random.randint(0, N - 1)
                path = [start]
                visited = {start}
                found = False
                for step2 in range(k):
                    if time.time() - start_time > DEADLINE:
                        break
                    curr = path[-1]
                    neighbors = [j for j in range(N) if has_edge(curr, j)]
                    if not neighbors:
                        break
                    # Prefer unvisited
                    unvisited = [n for n in neighbors if n not in visited]
                    if unvisited:
                        nxt = random.choice(unvisited)
                    else:
                        # Try to close the cycle
                        if step2 == k - 1 and has_edge(curr, start):
                            found = True
                            break
                        else:
                            break
                    if nxt == start and step2 == k - 1:
                        found = True
                        break
                    if nxt in visited and nxt != start:
                        break
                    path.append(nxt)
                    visited.add(nxt)
                
                if found and len(path) == k:
                    # Check it's a valid cycle
                    valid = True
                    for i in range(k):
                        u = path[i]
                        v = path[(i + 1) % k]
                        if not has_edge(u, v):
                            valid = False
                            break
                    if valid and len(set(path)) == k:
                        idx = k - 3
                        old_cycle = best_cycles[idx]
                        best_cycles[idx] = path
                        # Try to prune edges
                        needed = set()
                        for cyc in best_cycles:
                            kl = len(cyc)
                            for i in range(kl):
                                u = cyc[i]
                                v = cyc[(i + 1) % kl]
                                if u > v: u, v = v, u
                                needed.add((u, v))
                        for e in list(edges):
                            if e not in needed:
                                remove_edge(*e)
                        if validate_cycles(best_cycles):
                            new_score = len(edges) - 40
                            if new_score < best_score:
                                best_edges = set(edges)
                                best_score = new_score
                            else:
                                for e in best_edges:
                                    if e not in edges:
                                        add_edge(*e)
                                best_cycles[idx] = old_cycle
                        else:
                            best_cycles[idx] = old_cycle
    
        # Cool
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final cleanup: remove any edges not needed by best_cycles
    needed = set()
    for cyc in best_cycles:
        k = len(cyc)
        for i in range(k):
            u = cyc[i]
            v = cyc[(i + 1) % k]
            if u > v: u, v = v, u
            needed.add((u, v))
    
    for e in list(edges):
        if e not in needed:
            remove_edge(*e)
    
    # Validate final result
    if not validate_cycles(best_cycles):
        # Fallback to known-good solution
        best_edges = set()
        adj = [0] * N
        for i in range(1, N):
            best_edges.add((0, i))
            adj[0] |= (1 << i)
            adj[i] |= (1 << 0)
        for i in range(1, N - 1):
            best_edges.add((i, i + 1))
            adj[i] |= (1 << (i + 1))
            adj[i + 1] |= (1 << i)
        best_cycles = [list(range(k)) for k in range(3, N + 1)]
    
    # Ensure all cycle edges are in edge set
    for cyc in best_cycles:
        k = len(cyc)
        for i in range(k):
            u = cyc[i]
            v = cyc[(i + 1) % k]
            if u > v: u, v = v, u
            if (u, v) not in best_edges:
                best_edges.add((u, v))
    
    return {'edges': list(best_edges), 'cycles': best_cycles}
