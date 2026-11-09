import time
import random
import sys

def construct():
    n = 16
    target_edges = 65
    
    # Initialize with seed graph: K_{8,8} + edge (0,1)
    edges = set()
    for u in range(8):
        for v in range(8, 16):
            edges.add((u, v))
    edges.add((0, 1))
    
    # Helper to compute score
    def get_score(edge_set):
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Calculate degrees
        deg = [len(adj[i]) for i in range(n)]
        
        max_sum = 0
        # Find triangles
        # For each edge (u, v), check common neighbors
        # To avoid duplicates, only consider w > v? Or just check all and divide?
        # Simpler: iterate u < v < w
        # 16 vertices, 16*15*14/6 = 560 triples. Very fast.
        for u in range(n):
            for v in range(u + 1, n):
                if v in adj[u]:
                    common = adj[u] & adj[v]
                    for w in common:
                        if w > v:
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    # Initial score
    current_score = get_score(edges)
    best_score = current_score
    best_edges = set(edges)
    
    # All possible edges
    all_edges = set((i, j) for i in range(n) for j in range(i + 1, n))
    
    start_time = time.time()
    timeout = 140.0  # Safety margin
    
    # Simple local search
    # We try to improve by swapping edges.
    # We can also try to reduce the degree of high-degree vertices.
    
    # To speed up, we can pre-calculate something?
    # For n=16, get_score is very fast.
    
    steps = 0
    while time.time() - start_time < timeout:
        # Pick a random edge to remove
        if not edges:
            break
        e_remove = random.choice(list(edges))
        edges.discard(e_remove)
        
        # Pick a random non-edge to add
        # We need to ensure we don't add an edge already in the set
        # and that the new edge is not the same as removed (handled by discard)
        # and that we maintain simple graph (no multi-edges, no loops)
        
        # Find a valid non-edge
        # Non-edges are all_edges - edges
        # But we removed one, so it's a non-edge now.
        # We should not add it back immediately if we want to move?
        # Actually, swapping usually means remove one, add another different one.
        
        # Let's pick a random pair from all_edges that is not in edges
        # To avoid picking the one we just removed, we can just pick from all_edges - edges
        # But if we pick the one we removed, it's a no-op.
        
        non_edges = all_edges - edges
        if not non_edges:
            edges.add(e_remove)
            continue
            
        e_add = random.choice(list(non_edges))
        
        # Ensure e_add is not e_remove (it can't be, as e_remove is in edges? 
        # Wait. We discarded e_remove. So e_remove is in non_edges.
        # If we pick e_add == e_remove, we revert.
        # Let's avoid that.
        if e_add == e_remove:
             edges.add(e_remove)
             continue
             
        edges.add(e_add)
        
        new_score = get_score(edges)
        
        if new_score < best_score:
            best_score = new_score
            best_edges = set(edges)
            # If we hit 24 or lower, we can stop or continue to polish
            if best_score <= 23:
                break
        else:
            # Revert if worse?
            # Hill climbing: revert if worse.
            # Simulated annealing: accept with probability.
            # Given the small space, hill climbing might get stuck.
            # But 26 is close to 24.
            # Let's try a bit of randomness.
            # If new_score <= current_score + 1, accept?
            # For now, let's just revert if worse to ensure we don't degrade best.
            # But we need to escape local minima.
            # Let's do: if new_score < current_score, accept.
            # Else, with prob 0.1, accept (noise).
            if random.random() < 0.1 and new_score <= best_score:
                 pass # Keep the change
            else:
                 # Revert
                 edges.discard(e_add)
                 edges.add(e_remove)
        
        # Update current score for acceptance logic if we accepted
        if e_add in edges:
             current_score = new_score
        else:
             current_score = get_score(edges) # Should be old score

        steps += 1
        if steps % 10000 == 0:
            # Print progress to stderr if needed, but don't slow down
            pass

    return {'n': n, 'edges': [[u, v] for u, v in best_edges]}
