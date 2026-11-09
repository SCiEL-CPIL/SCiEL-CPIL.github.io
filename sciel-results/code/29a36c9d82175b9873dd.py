import time
import random
from itertools import combinations

_START = time.time()
_LIMIT = 145.0  # Safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency sets for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Precompute edge set for O(1) lookup
    edge_set = set(edges)
    
    def is_clique(vertices):
        """Check if a list of vertices forms a clique in the original graph."""
        if len(vertices) <= 1:
            return True
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                u, v = vertices[i], vertices[j]
                e = (u, v) if u < v else (v, u)
                if e not in edge_set:
                    return False
        return True

    def get_clique_edges(clique):
        """Get the set of edges covered by a clique."""
        return set((min(a,b), max(a,b)) for a,b in combinations(clique, 2))

    # Initial Greedy Partition
    # We use a simple greedy: pick edge, expand to maximal clique, remove edges.
    # We do this multiple times to get a good starting point for ILS.
    
    def initial_greedy():
        current_edges = set(edges)
        partition = []
        # Sort edges by a heuristic to get a decent start
        # Heuristic: edges with fewer common neighbors first (hard edges)
        edge_scores = []
        for e in edges:
            u, v = e
            common = len(adj[u] & adj[v])
            edge_scores.append((common, e))
        edge_scores.sort()
        
        # We need to pick edges from the remaining set, so we can't just iterate sorted list directly
        # without checking if edge is still present.
        # For speed, we'll just iterate the sorted list and pick the first valid one.
        # This is O(E^2) worst case but E is small (~1000).
        
        while current_edges:
            if not _deadline_ok():
                # Fallback
                for e in current_edges:
                    partition.append([e[0], e[1]])
                return partition
                
            # Find the first edge in sorted order that is still in current_edges
            picked_edge = None
            for score, e in edge_scores:
                if e in current_edges:
                    picked_edge = e
                    break
            
            if picked_edge is None:
                # Should not happen, but just in case
                picked_edge = next(iter(current_edges))
                
            u, v = picked_edge
            clique = [u, v]
            
            # Expand to maximal clique
            candidates = [w for w in adj[u] if w != v and (u,w) in current_edges and (v,w) in current_edges]
            
            while candidates:
                # Pick candidate with max degree in candidate subgraph
                best_w = -1
                best_deg = -1
                for w in candidates:
                    deg = 0
                    for x in candidates:
                        if w == x: continue
                        e = (w,x) if w<x else (x,w)
                        if e in current_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
                        
                if best_w == -1:
                    break
                    
                clique.append(best_w)
                candidates.remove(best_w)
                
                # Update candidates
                new_candidates = []
                for w in candidates:
                    e = (w, best_w) if w < best_w else (best_w, w)
                    if e in current_edges:
                        new_candidates.append(w)
                candidates = new_candidates
                
            partition.append(clique)
            
            # Remove edges
            for e in get_clique_edges(clique):
                if e in current_edges:
                    current_edges.remove(e)
                    
        return partition

    best_partition = initial_greedy()
    best_count = len(best_partition)
    
    # Local Search: Merge
    # Try to merge any two cliques in the partition if their union is a clique.
    # Note: If C1 and C2 are cliques, is C1 U C2 a clique?
    # Yes, if every vertex in C1 is connected to every vertex in C2.
    
    def try_merge(partition):
        """Attempt to merge cliques in the partition. Returns new partition and count of merges."""
        changed = True
        merges = 0
        while changed:
            changed = False
            new_partition = []
            used = [False] * len(partition)
            
            for i in range(len(partition)):
                if used[i]:
                    continue
                
                c1 = partition[i]
                merged_c1 = list(c1)
                
                for j in range(i + 1, len(partition)):
                    if used[j]:
                        continue
                    
                    c2 = partition[j]
                    
                    # Check if union is a clique
                    # This is O(|C1| * |C2|)
                    is_union_clique = True
                    for u in c1:
                        for v in c2:
                            e = (u, v) if u < v else (v, u)
                            if e not in edge_set:
                                is_union_clique = False
                                break
                        if not is_union_clique:
                            break
                    
                    if is_union_clique:
                        # Merge
                        merged_c1.extend(c2)
                        used[j] = True
                        merges += 1
                        changed = True
                        # Continue checking other cliques against the newly merged one
                        # This is a simple greedy merge. We don't restart the outer loop immediately
                        # but continue with the current merged_c1.
                    
                new_partition.append(merged_c1)
                
            partition = new_partition
            
        return partition, merges

    # Iterated Local Search
    rng = random.Random(seed)
    
    # Run ILS until time limit
    current_partition = best_partition
    current_count = best_count
    
    # We will run several rounds of ILS
    # Each round:
    # 1. Merge all possible merges
    # 2. If no merge possible, perturb
    # 3. Merge again
    
    max_iterations = 5000  # Cap iterations to avoid infinite loops if stuck
    iterations = 0
    
    while _deadline_ok() and iterations < max_iterations:
        iterations += 1
        
        # Step 1: Merge
        merged_partition, merges = try_merge(current_partition)
        
        if merges > 0:
            current_partition = merged_partition
            current_count = len(current_partition)
            if current_count < best_count:
                best_count = current_count
                best_partition = current_partition
            # Continue merging if possible
            continue
            
        # Step 2: No merge possible. Perturb.
        # Perturbation strategies:
        # A. Split a random clique into two parts (if size > 2)
        # B. Move a random vertex from one clique to another (if it maintains clique property for both? No, that breaks partition property easily)
        #    Actually, moving a vertex is hard because edges incident to it must be covered.
        #    If we move vertex v from C1 to C2:
        #    - Edges (v, u) for u in C1 \ {v} must be covered by other cliques. They are currently in C1.
        #    - This implies we must also move those edges or re-partition.
        #    Splitting is safer.
        
        # Strategy A: Split
        # Pick a random clique of size > 2.
        # Split it into two non-empty subsets S1, S2.
        # For S1 and S2 to be valid cliques, they must be cliques in G.
        # Since the original clique is a clique, any subset is a clique.
        # So we can split any clique into any two subsets.
        # However, this increases the count by 1. We hope that subsequent merges can find a better configuration.
        
        splittable = [c for c in current_partition if len(c) > 2]
        
        if not splittable:
            # All cliques are edges (size 2) or singletons?
            # Task says "at least two vertices". So min size is 2.
            # If all are size 2, we can't split.
            # Try swapping vertices between two cliques?
            # This is complex. Let's just break if we can't split.
            # Or try a different perturbation: Randomly reorder and re-greedy?
            # Let's try a "Kick": Pick two cliques, try to swap one vertex from each.
            # This is risky. Let's just break the loop if we can't split.
            # Actually, we can try to "re-greedy" the whole thing with a random seed.
            if iterations % 10 == 0:
                 # Re-greedy
                 current_partition = initial_greedy()
                 current_count = len(current_partition)
                 if current_count < best_count:
                     best_count = current_count
                     best_partition = current_partition
                 continue
            else:
                break
                
        # Pick a random splittable clique
        c = rng.choice(splittable)
        
        # Pick a random split point
        # Shuffle the vertices to get a random subset
        shuffled_c = list(c)
        rng.shuffle(shuffled_c)
        split_idx = rng.randint(1, len(shuffled_c) - 1)
        
        s1 = shuffled_c[:split_idx]
        s2 = shuffled_c[split_idx:]
        
        # Replace c with s1 and s2 in current_partition
        new_partition = []
        for cli in current_partition:
            if cli == c:
                new_partition.append(s1)
                new_partition.append(s2)
            else:
                new_partition.append(cli)
                
        current_partition = new_partition
        current_count = len(current_partition)
        
        # Do not update best_count here as it's worse.
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
