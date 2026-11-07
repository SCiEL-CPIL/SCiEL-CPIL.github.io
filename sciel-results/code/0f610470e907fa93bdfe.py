"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_clique(vertices, adj):
        if len(vertices) < 2:
            return False
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                if v not in adj[u]:
                    return False
        return True

    def greedy_clique_partition(n, k, edges, adj, time_limit):
        """
        Greedy algorithm: repeatedly find a large clique covering many uncovered edges,
        remove those edges, until all edges are covered.
        """
        edge_set = set(edges)
        cliques = []
        
        while edge_set:
            if time.time() > time_limit:
                break
            
            # Find a good clique: try to maximize number of uncovered edges covered
            # Strategy: pick an uncovered edge, then try to extend it to a large clique
            # that covers many uncovered edges
            
            best_clique = None
            best_count = 0
            
            # Sample some uncovered edges to find good clique seeds
            sample_edges = list(edge_set)
            if len(sample_edges) > 50:
                sample_edges = random.sample(sample_edges, 50)
            
            for u, v in sample_edges:
                # Try to build a clique containing u and v
                # Start with {u, v} and greedily add vertices that are adjacent to all current members
                # and cover many uncovered edges
                
                current = {u, v}
                # Candidate vertices: those adjacent to both u and v
                candidates = adj[u] & adj[v]
                
                # Greedily add candidates that maximize uncovered edge coverage
                while True:
                    best_candidate = None
                    best_gain = 0
                    
                    for cand in list(candidates):
                        # Count how many uncovered edges would be covered by adding cand
                        gain = 0
                        for member in current:
                            if (min(cand, member), max(cand, member)) in edge_set:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = cand
                    
                    if best_candidate is None or best_gain == 0:
                        break
                    
                    current.add(best_candidate)
                    candidates.discard(best_candidate)
                    # Update candidates: must be adjacent to new member
                    candidates &= adj[best_candidate]
                
                # Count total uncovered edges covered by this clique
                count = 0
                clique_list = list(current)
                for i in range(len(clique_list)):
                    for j in range(i+1, len(clique_list)):
                        a, b = clique_list[i], clique_list[j]
                        if (min(a, b), max(a, b)) in edge_set:
                            count += 1
                
                if count > best_count:
                    best_count = count
                    best_clique = sorted(current)
            
            if best_clique is None or best_count == 0:
                # Fallback: just take a single edge as a 2-vertex clique
                u, v = next(iter(edge_set))
                best_clique = [u, v]
                best_count = 1
            
            # Remove covered edges
            clique_set = set(best_clique)
            for u, v in combinations(best_clique, 2):
                edge_set.discard((min(u, v), max(u, v)))
            
            cliques.append(best_clique)
        
        # Check if we covered all edges
        if edge_set:
            # Cover remaining edges with 2-vertex cliques
            for u, v in edge_set:
                cliques.append([u, v])
        
        return cliques

    def local_search_improve(n, k, edges, adj, cliques, time_limit):
        """
        Try to improve the partition by merging or reorganizing cliques.
        """
        if time.time() > time_limit:
            return cliques
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            new_cliques = list(cliques)
            
            # Try to merge pairs of cliques if their union is a clique
            i = 0
            while i < len(new_cliques):
                if time.time() > time_limit:
                    break
                j = i + 1
                while j < len(new_cliques):
                    if time.time() > time_limit:
                        break
                    union = set(new_cliques[i]) | set(new_cliques[j])
                    if is_clique(list(union), adj):
                        # Check if the merged clique covers all edges from both original cliques
                        # and doesn't introduce issues
                        new_cliques.pop(j)
                        new_cliques[i] = sorted(union)
                        improved = True
                        j = i + 1
                    else:
                        j += 1
                i += 1
            
            # Try to split large cliques into smaller ones if it reduces total count
            # (unlikely to help, but try)
            # Actually, splitting increases count, so skip
            
            # Try to reassign edges: find if some edges in a clique could be moved
            # to form a better structure
            # This is complex; for now, just do the merge pass
            
            cliques = new_cliques
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        if time.time() > deadline - 5:
            # Not enough time left; return what we have
            break
        
        edges = instance(n, k, p, seed)
        adj = build_adjacency(n, edges)
        
        # Greedy partition
        cliques = greedy_clique_partition(n, k, edges, adj, deadline - 10)
        
        # Local search improvement
        cliques = local_search_improve(n, k, edges, adj, cliques, deadline - 5)
        
        # Validate: every clique must be a valid clique with at least 2 vertices
        # and every edge must be in exactly one clique
        valid_cliques = []
        covered_edges = set()
        for clique in cliques:
            if len(clique) < 2:
                continue
            if not is_clique(clique, adj):
                continue
            # Check no edge is covered twice
            clique_edges = set()
            for u, v in combinations(clique, 2):
                e = (min(u, v), max(u, v))
                if e in covered_edges:
                    # Edge already covered; this clique is invalid in our scheme
                    # We need to handle this differently
                    break
                clique_edges.add(e)
            else:
                covered_edges |= clique_edges
                valid_cliques.append(clique)
        
        # If some edges are not covered, add them as 2-vertex cliques
        uncovered = set(edges) - covered_edges
        for u, v in uncovered:
            valid_cliques.append([u, v])
        
        results.append(valid_cliques)
    
    # Ensure we have exactly 3 results
    while len(results) < 3:
        results.append([])
    
    # Convert to ensure proper types
    final_result = []
    for clique_list in results:
        converted = []
        for clique in clique_list:
            converted.append([int(v) for v in clique])
        final_result.append(converted)
    
    return final_result

# EVOLVE-BLOCK-END
