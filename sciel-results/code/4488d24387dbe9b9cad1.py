import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0  # safety margin before 150s kill

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_clique(vertices, adj):
        """Check if vertices form a clique."""
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                if not (adj[u] >> v) & 1:
                    return False
        return True

    def greedy_clique_partition(n, edges, adj, max_time):
        """
        Greedy edge-coverage heuristic:
        1. Start with all edges as separate cliques
        2. Repeatedly try to merge cliques or find larger cliques that cover more edges
        3. Use a scoring function: for each potential clique, score = number of unique edges it covers
        """
        rem_adj = [0] * n
        for u, v in edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        cliques = []
        
        while True:
            if time.time() > max_time:
                break
            
            # Check if any edges remain
            has_edges = False
            for i in range(n):
                if rem_adj[i] != 0:
                    has_edges = True
                    break
            
            if not has_edges:
                break
            
            # Find the best clique to add: maximize edges covered
            best_clique = None
            best_score = 0
            
            # Strategy: try extending from each vertex
            # For each vertex, find its connected component in rem_adj
            # Then try to build a clique from that component
            
            for start_v in range(n):
                if rem_adj[start_v] == 0:
                    continue
                
                # Get neighbors of start_v in remaining graph
                neighbors_mask = rem_adj[start_v]
                
                # Try to build a clique starting with start_v
                # Greedy: add vertices that connect to all current clique members
                current_clique = [start_v]
                common = neighbors_mask
                
                while common:
                    # Pick vertex with highest degree in common
                    best_v = -1
                    best_deg = -1
                    temp = common
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        temp &= temp - 1
                        deg = bin(common & adj[v]).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_v = v
                    
                    if best_v == -1:
                        break
                    
                    # Check if adding best_v maintains clique property
                    # (all pairs in current_clique + [best_v] must be edges)
                    can_add = True
                    for cv in current_clique:
                        if not (adj[best_v] >> cv) & 1:
                            can_add = False
                            break
                    
                    if can_add:
                        current_clique.append(best_v)
                        common &= adj[best_v]
                    else:
                        # Skip this vertex, try next
                        common &= ~(1 << best_v)
                
                # Score this clique by number of edges it covers
                clique_set = set(current_clique)
                edge_count = 0
                for i in range(len(current_clique)):
                    for j in range(i + 1, len(current_clique)):
                        u, v = current_clique[i], current_clique[j]
                        if (rem_adj[u] >> v) & 1:
                            edge_count += 1
                
                if edge_count > best_score:
                    best_score = edge_count
                    best_clique = current_clique[:]
            
            if best_clique is None or len(best_clique) < 2:
                # Fallback: take any edge
                found = False
                for u in range(n):
                    if rem_adj[u]:
                        v = (rem_adj[u] & -rem_adj[u]).bit_length() - 1
                        if (rem_adj[u] >> v) & 1:
                            best_clique = [u, v]
                            found = True
                            break
                if not found:
                    break
            
            # Remove edges of best_clique from remaining graph
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    rem_adj[u] &= ~(1 << v)
                    rem_adj[v] &= ~(1 << u)
            
            cliques.append(best_clique)
        
        return cliques

    def local_improve(cliques, n, adj, max_time):
        """
        Try to improve the partition by:
        1. Merging two cliques if their union is also a clique
        2. Moving edges between cliques to reduce total count
        """
        if time.time() > max_time:
            return cliques
        
        improved = True
        while improved:
            if time.time() > max_time:
                break
            
            improved = False
            new_cliques = []
            
            # Try to merge pairs of cliques
            merged = False
            i = 0
            while i < len(cliques):
                if time.time() > max_time:
                    break
                
                merged_this = False
                for j in range(i + 1, len(cliques)):
                    # Check if union of cliques[i] and cliques[j] is a clique
                    union = list(set(cliques[i]) | set(cliques[j]))
                    if is_clique(union, adj):
                        # Merge them
                        new_cliques.append(union)
                        cliques[j] = None
                        merged_this = True
                        merged = True
                        break
                
                if not merged_this:
                    new_cliques.append(cliques[i])
                
                i += 1
            
            # Filter out None
            new_cliques = [c for c in new_cliques if c is not None]
            
            if merged:
                cliques = new_cliques
                improved = True
        
        return cliques

    def solve_instance(n, p, seed):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline: each edge as its own clique
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        # Allocate time based on instance size
        if n == 32:
            time_limit = DEADLINE - 40.0  # Reserve 40s for second instance
        else:
            time_limit = DEADLINE
        
        # Run greedy partition
        result = greedy_clique_partition(n, edges, adj, time_limit)
        
        if result is not None and len(result) > 0:
            # Apply local improvement
            improved = local_improve(result, n, adj, min(time_limit, DEADLINE - 5.0))
            
            if len(improved) < best_count:
                best_count = len(improved)
                best_cliques = improved
        
        return best_cliques

    # Solve both instances
    result1 = solve_instance(32, 0.6, 601)
    
    if time.time() > DEADLINE:
        # Fallback: return baseline for second
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]
