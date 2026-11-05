import time
import random

_START_TIME = time.time()
DEADLINE = 120.0

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _can_add_edge(adj, neighbors, u, v, c, n):
    """Check if edge (u,v) with color c can be added without violating rainbow 4-cycle constraint."""
    for a in neighbors[u]:
        if a == v:
            continue
        ca = adj[u][a]
        for b in neighbors[v]:
            if b == u or b == a:
                continue
            if adj[a][b] != -1:
                cb = adj[a][b]
                cv = adj[b][v]
                # Check all four colors are distinct
                if ca == c or cb == c or cv == c:
                    return False
                if ca == cb or ca == cv or cb == cv:
                    return False
    return True

def _local_search(adj, neighbors, edges, skipped_edges, rng, max_iterations=100):
    """Local search: try to add skipped edges by re-coloring existing edges."""
    n = 20
    num_colors = 20
    current_edges = edges[:]
    
    for _ in range(max_iterations):
        if _time_left() < 10.0:
            break
        
        # Pick a random skipped edge
        if not skipped_edges:
            break
        
        idx = rng.randrange(len(skipped_edges))
        u, v = skipped_edges[idx]
        
        # Try to find a color for this edge
        found_color = None
        colors = list(range(num_colors))
        rng.shuffle(colors)
        
        for c in colors:
            if _can_add_edge(adj, neighbors, u, v, c, n):
                found_color = c
                break
        
        if found_color is not None:
            # Add the edge
            adj[u][v] = found_color
            adj[v][u] = found_color
            neighbors[u].add(v)
            neighbors[v].add(u)
            current_edges.append([u, v, found_color])
            skipped_edges.pop(idx)
        else:
            # Try re-coloring an existing edge to free up a color
            # Pick a random existing edge and try to re-color it
            if current_edges:
                edge_idx = rng.randrange(len(current_edges))
                eu, ev, ec = current_edges[edge_idx]
                
                # Try to re-color this edge
                new_colors = list(range(num_colors))
                rng.shuffle(new_colors)
                
                for new_c in new_colors:
                    if new_c == ec:
                        continue
                    
                    # Temporarily remove the edge
                    adj[eu][ev] = -1
                    adj[ev][eu] = -1
                    neighbors[eu].discard(ev)
                    neighbors[ev].discard(eu)
                    
                    # Check if we can add the skipped edge
                    if _can_add_edge(adj, neighbors, u, v, found_color if found_color is not None else 0, n):
                        # Try each color for the skipped edge
                        for c2 in range(num_colors):
                            if _can_add_edge(adj, neighbors, u, v, c2, n):
                                # Add skipped edge
                                adj[u][v] = c2
                                adj[v][u] = c2
                                neighbors[u].add(v)
                                neighbors[v].add(u)
                                current_edges.append([u, v, c2])
                                skipped_edges.pop(idx)
                                
                                # Re-add the re-colored edge
                                adj[eu][ev] = new_c
                                adj[ev][eu] = new_c
                                neighbors[eu].add(ev)
                                neighbors[ev].add(eu)
                                current_edges[edge_idx] = [eu, ev, new_c]
                                break
                        break
                    
                    # Restore the edge
                    adj[eu][ev] = ec
                    adj[ev][eu] = ec
                    neighbors[eu].add(ev)
                    neighbors[ev].add(eu)
                    break
    
    return current_edges

def construct():
    n = 20
    num_colors = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_edges = []
    best_count = 0
    
    # We will run multiple restarts with different random seeds
    # to explore the solution space.
    attempts = 0
    
    while _time_left() > 15.0:
        rng = random.Random(attempts)
        
        # Shuffle edges to change the greedy order
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        # adj[u][v] = color (0..19) or -1 if no edge
        adj = [[-1] * n for _ in range(n)]
        # neighbors[u] = list of v where adj[u][v] != -1
        neighbors = [set() for _ in range(n)]
        edges = []
        skipped_edges = []
        
        for (u, v) in shuffled_edges:
            # Try to find a color for edge (u, v)
            placed = False
            # Randomize color order to avoid bias towards low colors
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            for c in colors:
                if _can_add_edge(adj, neighbors, u, v, c, n):
                    # Place edge
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    placed = True
                    break
            
            # If no color works, we skip this edge.
            if not placed:
                skipped_edges.append((u, v))
        
        # Local search phase: try to add back skipped edges
        edges = _local_search(adj, neighbors, edges, skipped_edges, rng, max_iterations=50)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
        attempts += 1
        
    # Fallback if no edges were found (should not happen)
    if not best_edges:
        best_edges = [[0, 1, 0]]
        
    return best_edges
