"""
Clique Partition via Greedy Set Cover and Simulated Annealing on Vertex-Clique Assignments.

Strategy:
1. Generate the graph.
2. Find all maximal cliques (or a large set of candidate cliques) using a Bron-Kerbosch variant 
   or simple enumeration for small graphs. Given n=32,64, we can enumerate triangles and K4s efficiently.
3. Initial Solution: Greedy Set Cover. 
   - Maintain a set of uncovered edges.
   - While uncovered edges exist:
       - Find the clique that covers the most uncovered edges.
       - Add it to the partition.
       - Remove its edges from the uncovered set.
   - This minimizes the number of cliques by packing edges into the largest possible structures.
4. Optimization: Simulated Annealing on the partition.
   - State: A list of cliques (sets of vertices).
   - Move: 
       a) Merge two cliques if their union is a clique.
       b) Split a clique into two valid cliques (if possible) to allow better merging later? 
          Actually, splitting increases count, so it's a "bad" move accepted with probability exp(-delta/T).
       c) Move a vertex from one clique to another (if the new clique remains a clique and the old one 
          remains a clique with >=2 vertices, or if the old one becomes empty).
   - Energy: Number of cliques in the partition.
   
   Note: The "Move a vertex" move is tricky because removing a vertex might break the clique property 
   of the remaining set? No, a subset of a clique is always a clique. So removing a vertex always 
   leaves a valid clique (if size >= 2). Adding a vertex to a clique requires checking if it is 
   connected to all existing vertices in that clique.
   
   This vertex-based representation is much more compact and allows for structural changes (merging/splitting)
   that are hard to do in edge-space without complex bookkeeping.
"""

import time
import random
import math
import sys
from itertools import combinations

_START = time.time()
DEADLINE = 145.0

def _remaining():
    return DEADLINE - time.time()

def _generate_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [[False] * n for _ in range(n)]
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u][v] = adj[v][u] = True
                edge_list.append((u, v))
    return adj, edge_list

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _is_clique(vertices, adj):
    if len(vertices) < 2:
        return False
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not adj[vertices[i]][vertices[j]]:
                return False
    return True

def _find_candidate_cliques(adj, n):
    """
    Find all triangles and K4s in the graph.
    For n=32, number of triples is ~5000, K4s ~20000. This is fast.
    We also include all edges as size-2 cliques as fallback.
    """
    candidates = []
    # Triangles
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]:
                continue
            for w in range(v + 1, n):
                if adj[u][w] and adj[v][w]:
                    candidates.append((u, v, w))
    # K4s
    # To find K4s efficiently, we can check if any triangle is part of a K4
    # Or just iterate K4 combinations if n is small. n=64 -> C(64,4) ~ 600k. Too slow?
    # Better: For each triangle (u,v,w), check if there is a common neighbor x.
    # If so, (u,v,w,x) is a K4.
    # We can do this by intersecting neighbor sets.
    tri_set = set(tuple(sorted(t)) for t in candidates)
    k4s = []
    for (u, v, w) in tri_set:
        # Common neighbors of u, v, w
        common = [x for x in range(n) if x not in (u,v,w) and adj[u][x] and adj[v][x] and adj[w][x]]
        for x in common:
            k4 = tuple(sorted((u, v, w, x)))
            k4s.append(k4)
    
    # Deduplicate K4s
    unique_k4s = list(set(k4s))
    
    # Add K4s to candidates
    for k4 in unique_k4s:
        candidates.append(k4)
        
    # Note: We don't explicitly add all edges as candidates in the main list to keep it focused,
    # but the greedy cover will fall back to edges if no larger clique covers them.
    # Actually, for the greedy step, we should consider all possible cliques.
    # But for SA, we work with the current partition.
    
    return candidates

def _solve_one_graph(adj, edge_list, time_budget=50.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []
    
    m = len(edge_list)
    edge_set = set(_edge_key(u, v) for u, v in edge_list)
    
    # 1. Greedy Set Cover Initialization
    # We want to pick a set of cliques that covers all edges.
    # Candidates: All triangles, all K4s.
    # If an edge is not covered by any triangle/K4, we must use the edge itself (size 2).
    
    candidates = _find_candidate_cliques(adj, n)
    
    # Precompute edges covered by each candidate
    cand_edges = []
    for c in candidates:
        edges = set()
        for u, v in combinations(c, 2):
            edges.add(_edge_key(u, v))
        cand_edges.append(edges)
    
    # Sort candidates by size descending, then by edge count (which is size*(size-1)/2)
    # To make greedy efficient, we can just iterate.
    
    uncovered = set(edge_set)
    partition = []
    
    # For efficiency in greedy step, we might want to pick the candidate covering the most uncovered edges.
    # Since candidates are fixed, we can just loop.
    # With ~10k candidates and ~1k edges, checking intersection is fast enough for a few iterations.
    # However, standard greedy set cover is O(N * M).
    
    # Optimization: Pre-compute which candidates cover which edges?
    # Or just do a simple greedy:
    # While uncovered:
    #   Find best candidate (max intersection with uncovered)
    #   Add to partition
    #   Remove covered edges
    
    # To speed up, we can restrict candidates to those that actually cover some uncovered edges.
    # Let's do a simple version first.
    
    # To avoid O(N*M) per step, we can just pick a random good candidate or use a heuristic.
    # Given the deadline, a simple greedy might be slow if N is large.
    # Let's use a heuristic: Pick the largest clique that covers at least one uncovered edge.
    # Ties broken randomly.
    
    # Actually, for n=32/64, the number of triangles is not huge.
    # Let's try a faster greedy:
    # Group candidates by their vertex set.
    
    # A better initial solution:
    # 1. Start with all edges as separate cliques (size 2).
    # 2. Try to merge them into larger cliques.
    # This is the "Edge SA" approach.
    
    # Let's stick to the Vertex SA approach as planned, but initialize with Greedy Set Cover.
    
    # To make Greedy Set Cover fast:
    # We'll just iterate through candidates sorted by size descending.
    # If a candidate covers any uncovered edges, we take it.
    # This is a "First Fit" style greedy. It might not be optimal but it's fast.
    # Better: Sort by (size, random) and pick the first one that covers max uncovered edges?
    # Let's just pick the candidate with the maximum intersection with `uncovered`.
    
    # To make this O(1) or O(k) per step, we can't easily do it without data structures.
    # Given time constraints, let's use a simpler initialization:
    # Randomly assign edges to cliques, then run SA.
    # OR, use the Greedy Set Cover but limit the number of candidates checked.
    
    # Let's try: 
    # 1. Create a list of all triangles.
    # 2. While uncovered edges exist:
    #    - Pick a random triangle that covers at least one uncovered edge.
    #    - If none, pick a random uncovered edge and make it a clique.
    
    # This is fast and produces a valid partition.
    
    triangles = [c for c in candidates if len(c) == 3]
    k4s = [c for c in candidates if len(c) == 4]
    
    # Combine into a pool, weighted by size?
    # Let's just use all candidates.
    
    # Efficient Greedy:
    # Maintain a list of active candidates.
    # At each step, find the candidate with the most uncovered edges.
    # This is expensive.
    
    # Alternative: 
    # Initialize with a random valid partition (e.g., all edges).
    # Then run SA to merge.
    # This is safer and simpler to implement correctly.
    
    # Let's initialize with all edges as cliques.
    cliques = [list(edge) for edge in edge_list]
    
    # Verify
    def _verify_partition(cliques, edge_list, adj):
        if not cliques:
            return len(edge_list) == 0
        edge_idx = {_edge_key(u, v): i for i, (u, v) in enumerate(edge_list)}
        edge_cover = [0] * len(edge_list)
        for clique in cliques:
            if len(clique) < 2:
                return False
            for a in range(len(clique)):
                for b in range(a + 1, len(clique)):
                    if not adj[clique[a]][clique[b]]:
                        return False
                    e = _edge_key(clique[a], clique[b])
                    if e in edge_idx:
                        edge_cover[edge_idx[e]] += 1
        for c in edge_cover:
            if c != 1:
                return False
        return True

    if not _verify_partition(cliques, edge_list, adj):
        return [[u, v] for u, v in edge_list]

    best_cliques = [c[:] for c in cliques]
    best_count = len(cliques)
    
    start_time = time.time()
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # State: list of sets of vertices
    current_cliques = [set(c) for c in cliques]
    current_count = len(current_cliques)
    
    # Precompute adjacency for quick clique checks
    # adj is already a list of lists.
    
    def _can_add_vertex(clique_set, v, adj):
        for u in clique_set:
            if not adj[u][v]:
                return False
        return True

    def _get_vertices(clique_set):
        return sorted(clique_set)

    # SA Loop
    while time.time() - start_time < time_budget and T > min_T:
        if _remaining() < 5:
            break
            
        # Move 1: Merge two cliques
        # Pick two random cliques i, j.
        # If union is a clique, merge them.
        # This reduces count by 1.
        
        # Move 2: Split a clique
        # Pick a clique. If size > 2, try to split into two cliques.
        # This increases count by 1.
        
        # Move 3: Move a vertex
        # Pick a vertex in clique A, move to clique B (if valid).
        # Count stays same (unless A becomes empty or B was empty, but B is existing).
        # If A becomes empty, count decreases by 1.
        
        move_type = random.random()
        
        if move_type < 0.5 and current_count > 1:
            # Try Merge
            i, j = random.sample(range(current_count), 2)
            set_i = current_cliques[i]
            set_j = current_cliques[j]
            
            # Check if union is a clique
            # We need to check all pairs between set_i and set_j.
            # If sizes are small, this is fast.
            is_clique = True
            for u in set_i:
                for v in set_j:
                    if not adj[u][v]:
                        is_clique = False
                        break
                if not is_clique:
                    break
            
            if is_clique:
                # Perform merge
                # Remove j, add its elements to i
                set_i.update(set_j)
                current_cliques[j] = set()
                # Compact? No, just keep empty sets and filter later.
                # But indices shift if we remove. Let's keep the list size constant 
                # and just mark as empty.
                
                new_count = sum(1 for s in current_cliques if s)
                delta_E = new_count - current_count # -1
                
                if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_cliques = [_get_vertices(s) for s in current_cliques if s]
                else:
                    # Revert
                    # Remove elements of set_j from set_i
                    # We need the original set_j. We lost it?
                    # We did set_i.update(set_j). We can't easily revert without storing old state.
                    # Let's store old state for reversion.
                    # This is inefficient. 
                    # Better: Perform move on a copy? No, too slow.
                    # Let's just not allow reversion for merges? 
                    # Or, we can just accept merges if they are valid? 
                    # Merges always reduce count, so they are always "good" in terms of objective.
                    # Why would we revert? We wouldn't. 
                    # So we can just accept all valid merges.
                    pass # Already accepted implicitly by not reverting.
            
        elif move_type < 0.7:
            # Try Split
            # Pick a clique with size >= 3
            large_cliques = [i for i, s in enumerate(current_cliques) if len(s) >= 3]
            if not large_cliques:
                T *= cooling_rate
                continue
                
            i = random.choice(large_cliques)
            set_i = current_cliques[i]
            
            # Try to split into two non-empty cliques
            # Randomly partition the vertices of set_i into two sets A and B
            # Check if A is a clique and B is a clique.
            # If yes, replace set_i with A and add B as a new clique.
            
            verts = list(set_i)
            random.shuffle(verts)
            # Try split sizes: 1 vs k-1, 2 vs k-2, etc.
            # Just try a random split point
            split_idx = random.randint(1, len(verts) - 1)
            A = set(verts[:split_idx])
            B = set(verts[split_idx:])
            
            if _is_clique(list(A), adj) and _is_clique(list(B), adj):
                # Perform split
                # Replace set_i with A
                # Add B as new clique
                current_cliques[i] = A
                current_cliques.append(B)
                
                new_count = sum(1 for s in current_cliques if s)
                delta_E = new_count - current_count # +1
                
                if random.random() < math.exp(-delta_E / T):
                    current_count = new_count
                else:
                    # Revert
                    current_cliques.pop()
                    current_cliques[i] = A | B
        
        else:
            # Try Move Vertex
            # Pick a random non-empty clique i
            non_empty = [i for i, s in enumerate(current_cliques) if s]
            if not non_empty:
                break
            i = random.choice(non_empty)
            set_i = current_cliques[i]
            if len(set_i) <= 2:
                T *= cooling_rate
                continue # Can't move vertex if size 2 (would become size 1, invalid)
            
            v = random.choice(list(set_i))
            
            # Pick a target clique j (can be i? no, move to another)
            # Or create new? No, move to existing.
            other_cliques = [j for j in range(current_count) if j != i and current_cliques[j]]
            if not other_cliques:
                T *= cooling_rate
                continue
                
            j = random.choice(other_cliques)
            set_j = current_cliques[j]
            
            # Check if v can be added to set_j
            if _can_add_vertex(set_j, v, adj):
                # Perform move
                set_i.discard(v)
                set_j.add(v)
                
                # Check if set_i is still valid (size >= 2)
                # If size < 2, it becomes invalid/empty.
                # If it becomes empty, count decreases.
                # If it becomes size 1, it's invalid. We must revert or fix.
                # Let's just revert if set_i becomes size 1.
                if len(set_i) < 2:
                    # Revert
                    set_j.discard(v)
                    set_i.add(v)
                else:
                    new_count = sum(1 for s in current_cliques if s)
                    delta_E = new_count - current_count
                    
                    if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                        current_count = new_count
                        if current_count < best_count:
                            best_count = current_count
                            best_cliques = [_get_vertices(s) for s in current_cliques if s]
                    else:
                        # Revert
                        set_j.discard(v)
                        set_i.add(v)
                        
        T *= cooling_rate

    # Final cleanup
    final_cliques = [_get_vertices(s) for s in current_cliques if s]
    
    if not _verify_partition(final_cliques, edge_list, adj):
        return best_cliques if best_cliques else [[u, v] for u, v in edge_list]
        
    return final_cliques

def construct():
    global _START
    _START = time.time()
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if _remaining() < 20:
            adj, edge_list = _generate_graph(n, p, seed)
            results.append([[u, v] for u, v in edge_list])
            continue
        
        adj, edge_list = _generate_graph(n, p, seed)
        
        time_budget = min(55.0, _remaining() - 10)
        
        cliques = _solve_one_graph(adj, edge_list, time_budget)
        
        if not _verify_partition(cliques, edge_list, adj):
            cliques = [[u, v] for u, v in edge_list]
        
        results.append(cliques)
    
    return results
