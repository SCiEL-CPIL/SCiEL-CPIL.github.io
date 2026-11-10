import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n = 18
    all_vertices = list(range(n))
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if adj_mask[u] & (1 << v):
            return True  # already present
        common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
        temp = common
        while temp:
            lsb = temp & (-temp)
            a = lsb.bit_length() - 1
            if adj_mask[a] & (common ^ lsb):
                return False
            temp ^= lsb
        return True

    def has_triangle_in_set(adj_mask, vertex_set):
        """Check if the induced subgraph on vertex_set contains a triangle."""
        temp = vertex_set
        while temp:
            lsb = temp & (-temp)
            a = lsb.bit_length() - 1
            rest = temp ^ lsb
            neighbors_in_set = adj_mask[a] & rest
            if neighbors_in_set:
                # Check if any two neighbors of a in the set are connected
                ntemp = neighbors_in_set
                while ntemp:
                    nlsb = ntemp & (-ntemp)
                    b = nlsb.bit_length() - 1
                    if adj_mask[b] & (neighbors_in_set ^ nlsb):
                        return True
                    ntemp ^= nlsb
            temp ^= lsb
        return False

    def max_triangle_free_size(adj_mask, time_limit=None):
        """Find the largest triangle-free induced subgraph using branch and bound.
        Returns (size, vertices_mask) or just size if time_limit is None."""
        if time_limit is None:
            time_limit = deadline - 2

        # Precompute: for each vertex, which other vertices form edges
        # Use a faster approach: iterate subsets via branch and bound
        
        best_size = [0]
        best_mask = [0]
        
        # Order vertices by degree (higher degree first to prune faster)
        degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degrees[x])
        
        def search(idx, current_mask, current_size):
            if time.time() > time_limit:
                return
            if idx == n:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                    best_mask[0] = current_mask
                return

            v = order[idx]
            
            # Upper bound
            if current_size + (n - idx) <= best_size[0]:
                return

            # Option 1: don't include vertex v
            search(idx + 1, current_mask, current_size)

            # Option 2: include vertex v if it doesn't create a triangle
            # Adding v creates a triangle iff there exist u, w in current_mask
            # such that adj[u][w] and adj[u][v] and adj[w][v]
            new_neighbors = adj_mask[v] & current_mask
            # Check if new_neighbors has any edge
            has_edge = False
            temp = new_neighbors
            while temp:
                lsb = temp & (-temp)
                w = lsb.bit_length() - 1
                if adj_mask[w] & (new_neighbors ^ lsb):
                    has_edge = True
                    break
                temp ^= lsb

            if not has_edge:
                search(idx + 1, current_mask | (1 << v), current_size + 1)

        search(0, 0, 0)
        return best_size[0], best_mask[0]

    def generate_maximal_k4_free():
        """Generate a random maximal K4-free graph by greedily adding edges."""
        adj_mask = [0] * n
        edges = []
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)

        for u, v in shuffled_edges:
            if can_add_edge(adj_mask, u, v):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append((u, v))

        return edges, adj_mask

    def try_improve_by_adding_edges(adj_mask, edges, current_score, time_limit):
        """Try to add edges to reduce the max triangle-free set size."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all non-edges in random order
            non_edges = [(u, v) for u, v in all_edges if not (adj_mask[u] & (1 << v))]
            random.shuffle(non_edges)
            
            for u, v in non_edges:
                if time.time() > time_limit:
                    break
                if not can_add_edge(adj_mask, u, v):
                    continue
                
                # Try adding this edge
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                edges.append((u, v))
                
                new_score, _ = max_triangle_free_size(adj_mask, time_limit - 1)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    adj_mask[u] ^= (1 << v)
                    adj_mask[v] ^= (1 << u)
                    edges.pop()
        
        return current_score

    def verify_score_11(adj_mask):
        """Check if every 12-vertex subset contains a triangle (i.e., max triangle-free <= 11)."""
        # Check all C(18,12) = 18564 subsets of size 12
        # For each, check if it contains a triangle
        for subset in combinations(range(n), 12):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if not has_triangle_in_set(adj_mask, mask):
                return False
        return True

    # Baseline: complete tripartite K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    tripartite_edges = [(a, b) for i in range(3) for j in range(i+1, 3) for a in parts[i] for b in parts[j]]
    tripartite_adj = [0] * n
    for u, v in tripartite_edges:
        tripartite_adj[u] |= (1 << v)
        tripartite_adj[v] |= (1 << u)
    
    best_score = 12
    best_edges = list(tripartite_edges)
    best_adj = list(tripartite_adj)

    # Phase 1: Generate many random maximal K4-free graphs and check for score 11
    num_random = 50
    for i in range(num_random):
        if time.time() > deadline - 20:
            break
        
        edges, adj_mask = generate_maximal_k4_free()
        
        # Quick check: count triangles
        tri_count = 0
        for a, b, c in combinations(range(n), 3):
            if (adj_mask[a] & (1 << b)) and (adj_mask[a] & (1 << c)) and (adj_mask[b] & (1 << c)):
                tri_count += 1
        
        # Need at least C(18,12) = 18564 triangles to cover all 12-sets
        # But that's a necessary not sufficient condition
        # Just compute the actual max triangle-free size
        score, mask = max_triangle_free_size(adj_mask, deadline - 15)
        
        if score < best_score:
            best_score = score
            best_edges = list(edges)
            best_adj = list(adj_mask)
            
            if score <= 11:
                # Verify it's really 11 or less
                if verify_score_11(adj_mask):
                    break
    
    # Phase 2: Local search from the best candidate
    if time.time() < deadline - 15 and best_score > 11:
        current_adj = list(best_adj)
        current_edges = list(best_edges)
        current_score = best_score
        current_score = try_improve_by_adding_edges(current_adj, current_edges, current_score, deadline - 10)
        if current_score < best_score:
            best_score = current_score
            best_edges = list(current_edges)
            best_adj = list(current_adj)
    
    # Phase 3: More random restarts if time remains
    if time.time() < deadline - 10 and best_score > 11:
        for i in range(30):
            if time.time() > deadline - 5:
                break
            edges, adj_mask = generate_maximal_k4_free()
            score, _ = max_triangle_free_size(adj_mask, deadline - 5)
            if score < best_score:
                best_score = score
                best_edges = list(edges)
                best_adj = list(adj_mask)
                if best_score <= 11:
                    break

    if best_edges is None:
        best_edges = tripartite_edges

    return [[u, v] for u, v in best_edges]
