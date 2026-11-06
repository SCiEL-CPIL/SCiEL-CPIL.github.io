"""
Clique Partition via Minimum Set Cover (ILP/Greedy) and Edge-Coverage Local Search.

Strategy:
1. Generate the graph.
2. Identify all maximal cliques (or a sufficient set of candidate cliques: triangles, K4s, etc.).
3. Formulate as a Set Cover problem:
   - Universe: Edges
   - Sets: Cliques (each set covers the edges within it)
   - Goal: Minimum number of sets covering all edges exactly once (partition).
   Note: Standard set cover allows overlap. Here we require exact partition.
   However, if we minimize the number of cliques, overlapping cliques are wasteful.
   We can solve the "Exact Cover" variant or just Set Cover and then repair overlaps.
   Actually, a clique partition is a specific case of edge coloring/covering.
   
   Let's use a hybrid:
   A. Greedy Set Cover to get a fast initial partition (ignoring the "exact" constraint initially, 
      then resolving overlaps by splitting or removing redundant edges? No, that's hard).
      
   Better Approach:
   The problem is equivalent to finding a partition of the edge set into cliques.
   This is the "Clique Partition" problem.
   
   New Structural Idea:
   Use a **Maximal Clique Enumeration** to get the pool of candidates.
   Then, use a **Greedy Algorithm** that picks the clique covering the most uncovered edges.
   This gives a valid partition if we ensure we don't pick a clique that covers an already covered edge.
   Wait, if we pick a clique, it covers its edges. If we pick another clique that shares an edge, 
   that edge is covered twice -> Invalid.
   
   So, we must pick a set of *disjoint* (in terms of edges) cliques that cover all edges.
   This is exactly the "Exact Cover by Cliques" problem.
   
   Algorithm:
   1. Enumerate all cliques of size >= 2. (For n=64, this can be large, but we can limit to 
      "maximal" cliques or cliques up to size K, plus edges).
      Actually, any clique can be decomposed into smaller cliques. 
      But minimizing count favors larger cliques.
      
   2. **Greedy Exact Cover**:
      - Start with all edges uncovered.
      - While uncovered edges exist:
        - Find the clique C that covers the maximum number of *currently uncovered* edges 
          and *zero* *currently covered* edges.
        - If such a clique exists, add it to the partition and mark its edges as covered.
        - If no such clique exists (all cliques covering uncovered edges also cover covered edges),
          we are stuck. This implies we need to "undo" some previous choices or use a smaller clique.
          This suggests a backtracking or local search approach is needed for the exactness.
          
   3. **Local Search on Edge Assignments**:
      - Represent the solution as an assignment of each edge to a clique ID.
      - Move: Reassign a set of edges to a different clique.
      - Objective: Minimize number of distinct clique IDs used.
      
   Let's try a different, more robust approach: **Simulated Annealing on Edge-to-Clique Mapping**.
   - State: A dictionary mapping each edge to a clique (identified by its vertex set or an ID).
   - But the clique identity must be consistent (all edges in a clique must form a clique).
   
   Alternative: **Vertex-to-Clique-Group Mapping**? No, vertices can be in multiple cliques.
   
   Let's stick to **Greedy + Repair + Local Search**.
   
   1. **Candidate Generation**: 
      - All edges (size 2).
      - All triangles (size 3).
      - All K4s (size 4).
      - Maybe K5s if n is small enough.
      
   2. **Initial Solution**: 
      - Randomized Greedy:
        - Shuffle candidates.
        - Iterate: If candidate's edges are all uncovered, pick it.
        - This yields a valid partition.
        
   3. **Optimization (Local Search)**:
      - We want to reduce the number of cliques.
      - Move: **Merge**. If two cliques C1 and C2 have a union that is a clique, merge them.
      - Move: **Split & Reassign**. Split a large clique C into C1 and C2. Then try to merge C1 with another clique C3, and C2 with C4. If this results in a valid partition with fewer total cliques (or same count but better structure for future merges), accept.
      - Move: **Swap**. Replace a clique C1 with two cliques C2 and C3 such that C2 U C3 = C1 (as edge sets) and C2, C3 are cliques. This increases count, so only accept if it enables a merge later? Or just use for diversification.
      
      Actually, the most effective move for reducing count is **Merging**.
      The bottleneck is that greedy might pick small cliques that prevent large merges later.
      
      **New Strategy: Maximal Clique Prioritization + Backtracking/Greedy with Lookahead**
      
      1. Find all maximal cliques.
      2. Sort them by size descending.
      3. Greedy selection:
         - Pick the largest maximal clique that doesn't conflict with already picked cliques (edge-disjoint).
         - Remove its edges from the graph.
         - Repeat on the remaining graph.
      4. This is fast and often good.
      5. Refine with Local Search:
         - Try to merge any two cliques in the partition if their union is a clique.
         - Try to "unpick" a clique and repick using the remaining edges, if it leads to a better solution.
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

def _find_maximal_cliques(adj, n):
    """
    Find all maximal cliques using Bron-Kerbosch with pivoting.
    For n=32, 64, this is feasible if the graph isn't too dense (but it is dense).
    Dense graphs have fewer maximal cliques (often just the whole graph or large parts of it).
    """
    cliques = []
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        
        # Choose pivot u from P U X to minimize branching
        # Simple pivot: pick first element
        if P:
            u = next(iter(P))
        elif X:
            u = next(iter(X))
        else:
            return
            
        # Candidates are P \ N(u)
        candidates = [v for v in P if not adj[u][v]]
        
        for v in candidates:
            bron_kerbosch(
                R | {v}, 
                P & {w for w in P if adj[v][w]}, 
                X & {w for w in X if adj[v][w]}
            )
            P.remove(v)
            X.add(v)
            
    bron_kerbosch(frozenset(), frozenset(range(n)), frozenset())
    return cliques

def _solve_one_graph(adj, edge_list, time_budget=50.0):
    n = max(max(u, v) for u, v in edge_list) + 1 if edge_list else 0
    if not edge_list:
        return []
    
    m = len(edge_list)
    edge_set = set(_edge_key(u, v) for u, v in edge_list)
    
    # 1. Find Maximal Cliques
    # For very dense graphs, the number of maximal cliques is small.
    # For sparse, it's larger. p=0.6, 0.55 are dense.
    max_cliques = _find_maximal_cliques(adj, n)
    
    # Also add all edges and triangles as candidates in case maximal cliques are too large 
    # or don't cover everything efficiently? 
    # Actually, any clique is a subset of a maximal clique.
    # If we partition into maximal cliques, we might leave gaps.
    # The greedy approach handles gaps by picking smaller cliques from the remaining edges.
    
    # To be safe and allow more flexibility, let's include:
    # - All maximal cliques
    # - All triangles (size 3)
    # - All edges (size 2)
    
    candidates = list(max_cliques)
    
    # Add triangles
    for u in range(n):
        for v in range(u + 1, n):
            if not adj[u][v]: continue
            for w in range(v + 1, n):
                if adj[u][w] and adj[v][w]:
                    candidates.append(frozenset([u, v, w]))
                    
    # Add edges
    for u, v in edge_list:
        candidates.append(frozenset([u, v]))
        
    # Deduplicate candidates
    unique_candidates = list(set(candidates))
    
    # Precompute edges for each candidate
    cand_edges = []
    for c in unique_candidates:
        edges = set()
        for u, v in combinations(c, 2):
            edges.add(_edge_key(u, v))
        cand_edges.append(edges)
        
    # 2. Greedy Set Cover (Exact Partition)
    # We want to pick a subset of candidates that are edge-disjoint and cover all edges.
    # Greedy: 
    #   uncovered = set(edge_set)
    #   while uncovered:
    #      find candidate c with max |c.edges & uncovered| such that c.edges & covered == 0
    #      if none, fail (shouldn't happen if edges are candidates)
    #      add c to partition
    #      uncovered -= c.edges
    #      covered |= c.edges
      
    # To speed this up, we can sort candidates by size descending initially.
    # But the "max uncovered" criterion is dynamic.
    
    # Optimization: 
    # Since n is small (32, 64), and we have at most ~10k-50k candidates, 
    # and m is ~1k-2k edges.
    # A naive loop over candidates for each step is O(Steps * Candidates * EdgeCheck).
    # Steps <= m. Candidates ~ 50k. EdgeCheck is set intersection.
    # This might be too slow in Python if not careful.
    
    # Faster Greedy:
    # Group candidates by their smallest edge? No.
    # Just iterate.
    
    uncovered = set(edge_set)
    covered = set()
    partition = []
    
    # Sort candidates by size descending to prefer larger cliques in ties
    # Index mapping
    cand_idx = list(range(len(unique_candidates)))
    cand_idx.sort(key=lambda i: -len(unique_candidates[i]))
    
    start_time = time.time()
    
    while uncovered:
        best_cand = -1
        best_count = 0
        best_edges = None
        
        # Check all candidates
        # Optimization: Skip candidates that have any edge in 'covered'
        # We can pre-filter? 
        # For each candidate, check if cand_edges[i] & covered is empty.
        # Then check len(cand_edges[i] & uncovered).
        
        for i in cand_idx:
            c_edges = cand_edges[i]
            # If it touches any covered edge, it's invalid for exact cover
            if c_edges & covered:
                continue
            
            # Count how many uncovered edges it covers
            cnt = len(c_edges & uncovered)
            if cnt > best_count:
                best_count = cnt
                best_cand = i
                best_edges = c_edges
                
        if best_cand == -1:
            # Should not happen if edges are in candidates
            # Fallback: pick any uncovered edge
            e = next(iter(uncovered))
            u, v = e
            c = frozenset([u, v])
            partition.append(list(c))
            uncovered.discard(e)
            covered.add(e)
        else:
            partition.append(list(unique_candidates[best_cand]))
            uncovered -= best_edges
            covered |= best_edges
            
        if time.time() - start_time > time_budget * 0.5:
            break
            
    # If we broke out early, finish with edges
    if uncovered:
        for e in uncovered:
            u, v = e
            partition.append([u, v])
            covered.add(e)
            
    # Verify partition
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

    if not _verify_partition(partition, edge_list, adj):
        return [[u, v] for u, v in edge_list]
        
    best_partition = [c[:] for c in partition]
    best_count = len(best_partition)
    
    # 3. Local Search: Merge Optimizations
    # Try to merge any two cliques if their union is a clique.
    # Repeat until no more merges are possible.
    
    # To make this efficient, we can maintain a list of cliques.
    # Merging reduces the count.
    
    # We can also try "Split and Merge":
    # Split a large clique C into C1, C2.
    # Try to merge C1 with C3 and C2 with C4.
    # If valid and count doesn't increase (or decreases), accept.
    
    # Given time constraints, simple Merge is powerful.
    
    current_cliques = [set(c) for c in partition]
    
    def _try_merge_all(cliques):
        merged = True
        while merged:
            merged = False
            n_cliques = len(cliques)
            for i in range(n_cliques):
                for j in range(i + 1, n_cliques):
                    union = cliques[i] | cliques[j]
                    if _is_clique(list(union), adj):
                        # Merge
                        cliques[i] = union
                        cliques[j] = set()
                        merged = True
                        break
                if merged:
                    break
            # Remove empty
            cliques = [c for c in cliques if c]
        return cliques
        
    current_cliques = _try_merge_all(current_cliques)
    
    if len(current_cliques) < best_count:
        best_count = len(current_cliques)
        best_partition = [list(c) for c in current_cliques]
        
    # 4. Advanced Local Search: Simulated Annealing on Clique Assignments
    # State: List of sets (cliques)
    # Moves:
    #   1. Merge (if valid) -> always accept (reduces count)
    #   2. Split a clique into two valid cliques -> accept with prob exp(-1/T) (increases count)
    #   3. Move a vertex from C1 to C2 (if C2 becomes clique and C1 remains clique or empty)
    #      - If C1 becomes empty, count decreases -> accept
    #      - If C1 remains, count same -> accept with prob 0.5? Or if it enables future merges?
    #        Let's accept if it doesn't break validity.
        
    T = 1.0
    cooling = 0.99
    min_T = 0.05
    
    # Precompute adjacency matrix as bitmasks for speed? 
    # n=64 fits in a Python int.
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in range(n):
            if adj[u][v]:
                mask |= (1 << v)
        adj_mask[u] = mask
        
    def _is_clique_fast(vertices):
        if len(vertices) < 2:
            return False
        # Check if all pairs are connected
        # For small sets, loop is fine.
        # For larger, we can check if the intersection of neighbors is the set itself.
        # Let's stick to loop for simplicity and correctness, n is small.
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if not adj[vertices[i]][vertices[j]]:
                    return False
        return True
        
    start_sa = time.time()
    
    while time.time() - start_sa < time_budget * 0.5 and T > min_T:
        if _remaining() < 5:
            break
            
        move_type = random.random()
        
        if move_type < 0.6 and len(current_cliques) > 1:
            # Merge attempt
            i, j = random.sample(range(len(current_cliques)), 2)
            union = current_cliques[i] | current_cliques[j]
            if _is_clique_fast(list(union)):
                current_cliques[i] = union
                current_cliques[j] = set()
                current_cliques = [c for c in current_cliques if c]
                if len(current_cliques) < best_count:
                    best_count = len(current_cliques)
                    best_partition = [list(c) for c in current_cliques]
                    
        elif move_type < 0.8:
            # Split attempt
            large = [i for i, c in enumerate(current_cliques) if len(c) >= 3]
            if not large:
                T *= cooling
                continue
            i = random.choice(large)
            verts = list(current_cliques[i])
            random.shuffle(verts)
            split_at = random.randint(1, len(verts) - 1)
            A = set(verts[:split_at])
            B = set(verts[split_at:])
            
            if _is_clique_fast(list(A)) and _is_clique_fast(list(B)):
                current_cliques[i] = A
                current_cliques.append(B)
                # Count increased by 1. Accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    pass # Keep
                else:
                    # Revert
                    current_cliques.pop()
                    current_cliques[i] = A | B
                    
        else:
            # Vertex Move
            non_empty = [i for i, c in enumerate(current_cliques) if len(c) >= 2]
            if not non_empty:
                break
            i = random.choice(non_empty)
            if len(current_cliques[i]) <= 2:
                T *= cooling
                continue
            v = random.choice(list(current_cliques[i]))
            
            # Target
            others = [j for j in range(len(current_cliques)) if j != i and current_cliques[j]]
            if not others:
                T *= cooling
                continue
            j = random.choice(others)
            
            # Check if v can join j
            can_join = True
            for u in current_cliques[j]:
                if not adj[v][u]:
                    can_join = False
                    break
            
            if can_join:
                # Perform move
                current_cliques[i].discard(v)
                current_cliques[j].add(v)
                
                # Check validity of i
                if len(current_cliques[i]) < 2:
                    # i became invalid or empty
                    if len(current_cliques[i]) == 0:
                        # i removed
                        current_cliques[i] = set()
                        current_cliques = [c for c in current_cliques if c]
                        if len(current_cliques) < best_count:
                            best_count = len(current_cliques)
                            best_partition = [list(c) for c in current_cliques]
                    else:
                        # i size 1, invalid. Revert.
                        current_cliques[j].discard(v)
                        current_cliques[i].add(v)
                else:
                    # i is still valid. Count same.
                    # Accept with some probability to allow exploration?
                    # Or always accept? 
                    # Let's accept 50% of the time to mix states.
                    if random.random() < 0.5:
                        pass # Keep
                    else:
                        current_cliques[j].discard(v)
                        current_cliques[i].add(v)
                        
        T *= cooling
        
    # Final Verification
    final_cliques = [list(c) for c in current_cliques if c]
    if not _verify_partition(final_cliques, edge_list, adj):
        return best_partition
        
    if len(final_cliques) < best_count:
        best_count = len(final_cliques)
        best_partition = final_cliques
        
    return best_partition

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
        
        time_budget = min(60.0, _remaining() - 10)
        
        cliques = _solve_one_graph(adj, edge_list, time_budget)
        
        # Final safety check
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
            cliques = [[u, v] for u, v in edge_list]
            
        results.append(cliques)
        
    return results

if __name__ == "__main__":
    construct()
