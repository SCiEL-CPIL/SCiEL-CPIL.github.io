"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Strategy: Randomized Constructive Search
    # 1. Start with a random sparse graph (e.g., 45 edges).
    # 2. Identify missing cycle lengths.
    # 3. Add edges that cover the most missing lengths.
    # 4. Iterate with occasional random restarts or perturbations.
    # 5. Once pancyclic, try to remove edges (local search).
    
    # Helper: Check if a cycle of length L exists in a graph
    # Using backtracking with pruning. Optimized for N=40.
    def find_cycle(graph_adj, L, start_vertex=None):
        if L < 3 or L > N:
            return None
        
        # Special case: Hamiltonian cycle
        if L == N:
            # We can use a simple heuristic: try to find a Hamiltonian cycle
            # using backtracking. This is expensive but N=40 is small enough
            # if the graph is dense enough.
            pass

        adj = graph_adj
        best_cycle = None
        
        # For efficiency, we can restrict the search space
        # Symmetry breaking: fix the start vertex to 0 (if connected)
        # But since the graph is undirected and we want ANY cycle, 
        # we can just pick a random start or iterate.
        
        # To speed up, we'll use a randomized DFS with a time limit per attempt
        # or just a standard DFS if the graph is small enough.
        
        # Let's use a more efficient approach for small L
        if L == 3:
            for u in range(N):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v and w in adj[u]:
                                return [u, v, w]
            return None
        
        if L == 4:
            for u in range(N):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v:
                                for x in adj[w]:
                                    if x > w and x in adj[u]:
                                        return [u, v, w, x]
            return None

        # General case: Backtracking
        # We need to find a simple cycle of length L.
        # We can fix the start vertex to 0 to reduce symmetry, 
        # but only if 0 is part of some potential cycle.
        # Actually, for robustness, let's try a few start vertices.
        
        # Optimization: Only try start vertices that have degree >= 2
        candidates = [v for v in range(N) if len(adj[v]) >= 2]
        if not candidates:
            return None
            
        random.shuffle(candidates)
        
        # Limit the number of start vertices to try to avoid timeout
        max_starts = 10
        
        for start in candidates[:max_starts]:
            # DFS
            path = [start]
            visited = {start}
            
            def dfs(current, depth):
                nonlocal best_cycle
                if best_cycle:
                    return True
                
                if depth == L:
                    if start in adj[current]:
                        best_cycle = list(path)
                        return True
                    return False
                
                # Pruning: 
                # 1. Not enough vertices left
                if N - len(visited) < L - depth:
                    return False
                # 2. If we are at depth 2, we cannot go to start yet (unless L=3)
                #    Actually, we can only close the cycle at depth L.
                
                # Heuristic: Try neighbors in order of degree
                neighbors = list(adj[current])
                random.shuffle(neighbors)
                
                for nxt in neighbors:
                    if nxt == start and depth < L - 1:
                        continue
                    if nxt == start and depth == L - 1:
                        # This is the closing edge, handled in depth == L check
                        # Wait, if depth == L-1, we are at the last vertex before closing.
                        # So if nxt == start, we return True in the depth==L check.
                        # So we should allow nxt == start ONLY if depth == L-1?
                        # No, the check is: if depth == L, check if start in adj[current].
                        # So at depth L-1, we move to a neighbor. If that neighbor is start, 
                        # then at depth L, we check if start in adj[start] (False) 
                        # Wait. The cycle is v0-v1-...-v_{L-1}-v0.
                        # Path is [v0, ..., v_{L-1}]. Current is v_{L-1}.
                        # We check if v0 is in adj[v_{L-1}].
                        # So we don't need to "move" to start. We just check adjacency.
                        # So in the loop, we should NOT add start to path unless it's the very end?
                        # Actually, the standard DFS for cycle:
                        # If len(path) == L: check if path[-1] connected to path[0].
                        # So we never add start to path again.
                        if nxt in visited:
                            continue
                            
                    path.append(nxt)
                    visited.add(nxt)
                    
                    if dfs(nxt, depth + 1):
                        return True
                    
                    path.pop()
                    visited.remove(nxt)
                
                return False
            
            if dfs(start, 1):
                return best_cycle
        
        return None

    def get_missing_lengths(adj):
        missing = []
        for L in range(3, N + 1):
            if find_cycle(adj, L) is None:
                missing.append(L)
        return missing

    def build_adj(edges):
        adj = {i: set() for i in range(N)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_pancyclic(adj):
        for L in range(3, N + 1):
            if find_cycle(adj, L) is None:
                return False
        return True

    def get_all_cycles(adj):
        cycles = {}
        for L in range(3, N + 1):
            c = find_cycle(adj, L)
            if c:
                cycles[L] = c
        return cycles

    # Main Search Loop
    best_edges = None
    best_cycles = None
    best_score = float('inf')

    # We will run several random restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break
            
        # Initialize with a random sparse graph
        # Start with a random spanning tree (N-1 edges) + some random extra edges
        # Or just random edges until we have ~45 edges
        edges = set()
        
        # Add a random Hamiltonian path to ensure connectivity
        perm = list(range(N))
        random.shuffle(perm)
        for i in range(N - 1):
            u, v = perm[i], perm[i+1]
            edges.add((min(u, v), max(u, v)))
            
        # Add random extra edges to reach a target density
        target_edges = 45 + restart # Increase density slightly with restarts
        all_possible_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges:
            if len(edges) >= target_edges:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                
        adj = build_adj(edges)
        
        # Iterative improvement
        for step in range(100):
            if time.time() - start_time > deadline - 2:
                break
                
            missing = get_missing_lengths(adj)
            
            if not missing:
                # Pancyclic! Try to remove edges
                # Local search: try to remove one edge at a time
                removed_any = True
                while removed_any:
                    removed_any = False
                    edge_list = list(edges)
                    random.shuffle(edge_list)
                    for e in edge_list:
                        if time.time() - start_time > deadline - 2:
                            break
                        edges.discard(e)
                        new_adj = build_adj(edges)
                        if is_pancyclic(new_adj):
                            adj = new_adj
                            removed_any = True
                            # Update best
                            if len(edges) < best_score:
                                best_score = len(edges)
                                best_edges = set(edges)
                                best_cycles = get_all_cycles(adj)
                        else:
                            edges.add(e)
                    if time.time() - start_time > deadline - 2:
                        break
                break
            else:
                # Add an edge that covers the most missing lengths
                # Heuristic: Pick a missing length, find a path of length L-1, add the closing edge
                # Or just pick a random missing length and try to fix it
                missing_L = random.choice(missing)
                
                # Find a path of length missing_L - 1
                # This is hard. Instead, let's just add a random edge and see if it helps
                # Or better: pick two vertices that are "far apart" in terms of missing cycle constraints
                # Simple heuristic: Add a random edge
                all_possible_edges = [(i, j) for i in range(N) for j in range(i+1, N) if (i, j) not in edges]
                if not all_possible_edges:
                    break
                    
                # Try a few random edges
                random.shuffle(all_possible_edges)
                improved = False
                for u, v in all_possible_edges[:10]:
                    edges.add((u, v))
                    new_adj = build_adj(edges)
                    new_missing = get_missing_lengths(new_adj)
                    if len(new_missing) < len(missing):
                        adj = new_adj
                        missing = new_missing
                        improved = True
                        break
                    else:
                        edges.discard((u, v))
                        
                if not improved:
                    # Stuck, add a random edge anyway to perturb
                    u, v = random.choice(all_possible_edges)
                    edges.add((u, v))
                    adj = build_adj(edges)
                    
        # Update best if we found a pancyclic graph
        if is_pancyclic(adj):
            if len(edges) < best_score:
                best_score = len(edges)
                best_edges = set(edges)
                best_cycles = get_all_cycles(adj)

    # If we didn't find a good one, fallback to the 59-edge construction
    if best_edges is None:
        edges = set()
        for i in range(N):
            edges.add((min(i, (i+1)%N), max(i, (i+1)%N)))
        for k in range(2, 21):
            edges.add((0, k))
        best_edges = edges
        best_cycles = {}
        adj = build_adj(edges)
        for L in range(3, N+1):
            best_cycles[L] = find_cycle(adj, L)
            if not best_cycles[L]:
                # Should not happen for this construction
                best_cycles[L] = list(range(L)) # Dummy, will fail validation

    edge_list = sorted(list(best_edges))
    cycles_out = [best_cycles[L] for L in range(3, N+1)]
    
    return {'edges': edge_list, 'cycles': cycles_out}
# EVOLVE-BLOCK-END
