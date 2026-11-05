import time
import random
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # 1. Initial Construction: The Fan Graph
    edges = set()
    
    # Base Hamiltonian Cycle
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v: u, v = v, u
        edges.add((u, v))
        
    # Chords
    for m in range(2, 21):
        u, v = 0, m
        if u > v: u, v = v, u
        edges.add((u, v))
        
    current_edges = list(edges)
    
    # Helper to check if graph is pancyclic
    def is_pancyclic(edge_set):
        # Quick check: Connected?
        # Build adjacency list
        adj = {i: set() for i in range(N)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check connectivity (BFS)
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        if len(visited) != N:
            return False
            
        # Check for Hamiltonian Cycle (Length 40)
        # This is the most expensive check.
        # We can use a simple backtracking with pruning.
        # Since we start from a valid HC and only remove edges, 
        # we can check if the specific HC 0-1-...-39-0 is still intact.
        # If not, we need to find another one.
        # For speed, first check the original HC.
        hc_path = list(range(N))
        valid_hc = True
        for i in range(N):
            u, v = hc_path[i], hc_path[(i+1)%N]
            if (min(u,v), max(u,v)) not in edge_set:
                valid_hc = False
                break
        if not valid_hc:
            # Try to find any HC. 
            # This is NP-hard, but N=40 is small.
            # However, doing this for every candidate is too slow.
            # Heuristic: If the original HC is broken, assume it's invalid 
            # unless we have time to search.
            # Given the tight budget, we will assume that if the main HC is broken,
            # we might lose length 40. 
            # But other HCs might exist.
            # To be safe and fast, we will only accept removals if the 
            # "canonical" HC is preserved OR if we can quickly verify another one.
            # For now, let's just check if the canonical HC is present.
            # This might miss valid graphs, but it's a safe lower bound.
            return False
            
        # Check for cycles of length 3 to 39
        # For each k, we need a cycle.
        # We can use the known cycles from the fan construction as "witnesses".
        # If we removed an edge that was part of the witness cycle for k,
        # we need to find a new one.
        
        # To keep it fast, we will just check if the *specific* witness cycles
        # from the initial construction are still present.
        # If they are, the graph is definitely pancyclic.
        # This is a sufficient condition, not necessary.
        # It means we might not find improvements that rely on new cycles,
        # but it's a good start.
        
        # Witness cycles:
        # k=40: 0-1-...-39-0
        # k=3..21: 0-1-...-(k-1)-0
        # k=22..39: 0-39-...-(41-k)-0
        
        for k in range(3, N + 1):
            if k == N:
                cycle_verts = list(range(N))
            elif k <= 21:
                cycle_verts = list(range(k))
            else:
                m = 41 - k
                cycle_verts = [0] + list(range(39, m - 1, -1))
                
            # Check if all edges in this cycle exist in edge_set
            valid_cycle = True
            for i in range(len(cycle_verts)):
                u = cycle_verts[i]
                v = cycle_verts[(i+1) % len(cycle_verts)]
                if (min(u,v), max(u,v)) not in edge_set:
                    valid_cycle = False
                    break
            if not valid_cycle:
                return False
                
        return True

    # Local Search
    best_edges = set(current_edges)
    
    # Try to remove edges
    # We iterate through edges and try to remove them.
    # We do this in random order to avoid bias.
    edge_list = list(edges)
    random.shuffle(edge_list)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle again for each pass
        random.shuffle(edge_list)
        for e in edge_list:
            if time.time() > deadline:
                break
            if e not in best_edges:
                continue
            # Try removing e
            candidate = set(best_edges)
            candidate.remove(e)
            if is_pancyclic(candidate):
                best_edges = candidate
                improved = True
                # Update edge_list if needed, but it's just for iteration
                pass
                
    # If we found a better graph, use it. Otherwise, use the original.
    # The original is always valid.
    # We return the best found.
    
    edge_list_final = list(best_edges)
    
    # Construct cycles for the returned graph
    # Note: The cycles returned must be valid for the *returned* edge set.
    # If we removed edges, the "witness" cycles used in is_pancyclic are still valid
    # because is_pancyclic only returned True if those specific witness cycles were intact.
    # So we can just return those witness cycles.
    
    cycles = []
    for k in range(3, N + 1):
        if k == N:
            cycles.append(list(range(N)))
        elif k <= 21:
            cycles.append(list(range(k)))
        else:
            m = 41 - k
            cycles.append([0] + list(range(39, m - 1, -1)))
            
    return {'edges': edge_list_final, 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
