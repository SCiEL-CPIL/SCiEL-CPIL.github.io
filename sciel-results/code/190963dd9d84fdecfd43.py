import time
import random
import numpy as np
from collections import defaultdict
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adjacency(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_cliques_containing_edge(edge, adj, remaining_edges_set, n):
        """Find all cliques that contain the given edge and only use edges in remaining_edges_set."""
        u, v = edge
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        
        # We need to find all subsets of {u, v} ∪ S where S ⊆ common
        # such that all edges within the clique are in remaining_edges_set
        # and the clique is a valid clique in the graph.
        
        # For efficiency, we'll use a recursive approach or generate candidates
        # Since graphs are dense but n is small (32, 64), we can be smarter.
        
        # Actually, for branch and bound, we don't need ALL cliques.
        # We need a good set of candidate cliques to branch on.
        # Let's generate maximal cliques containing (u,v) within the remaining graph.
        
        # A clique containing (u,v) is {u,v} ∪ S where S ⊆ common and S is a clique in the graph
        # and all edges between {u,v} and S are in remaining_edges_set, and all edges within S are in remaining_edges_set.
        
        # Let's filter common to only vertices w such that (u,w) and (v,w) are in remaining_edges_set
        valid_common = set()
        for w in common:
            if (min(u,w), max(u,w)) in remaining_edges_set and (min(v,w), max(v,w)) in remaining_edges_set:
                valid_common.add(w)
        
        if not valid_common:
            return [[u, v]]
        
        # Now find all cliques within valid_common (in the induced subgraph)
        # This is the maximum clique problem on valid_common
        # For small sets, we can enumerate all subsets, but that's exponential.
        # Instead, let's just try a few promising cliques:
        # 1. Just {u, v}
        # 2. {u, v} + one vertex from valid_common (if it forms a triangle with remaining edges)
        # 3. {u, v} + two vertices if they form a K4
        # 4. Greedy maximal clique
        
        candidates = [[u, v]]
        
        # Try adding single vertices
        single_candidates = []
        for w in valid_common:
            # Check if (u,w) and (v,w) are in remaining (already checked in valid_common)
            single_candidates.append([u, v, w])
        
        # Try pairs of vertices from valid_common
        pair_candidates = []
        valid_list = list(valid_common)
        for i in range(len(valid_list)):
            for j in range(i+1, len(valid_list)):
                w1, w2 = valid_list[i], valid_list[j]
                # Check if w1-w2 edge is in remaining
                if (min(w1,w2), max(w1,w2)) in remaining_edges_set:
                    pair_candidates.append([u, v, w1, w2])
        
        # Add single and pair candidates
        candidates.extend(single_candidates)
        candidates.extend(pair_candidates)
        
        # Also try a greedy maximal clique
        greedy_clique = [u, v]
        remaining_valid = set(valid_common)
        while remaining_valid:
            # Pick the vertex with highest degree in remaining_valid that can be added
            best_w = None
            best_deg = -1
            for w in remaining_valid:
                # Check if w is connected to all in greedy_clique
                can_add = True
                for x in greedy_clique:
                    if w not in adj[x]:
                        can_add = False
                        break
                    # Check edge in remaining
                    if (min(x,w), max(x,w)) not in remaining_edges_set:
                        can_add = False
                        break
                if can_add:
                    # Count how many other remaining_valid vertices are connected to w
                    deg = 0
                    for y in remaining_valid:
                        if y != w and (min(w,y), max(w,y)) in remaining_edges_set:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_w = w
            if best_w is None:
                break
            greedy_clique.append(best_w)
            remaining_valid.discard(best_w)
        
        if len(greedy_clique) > 2 and greedy_clique not in candidates:
            candidates.append(greedy_clique)
        
        # Remove duplicates and sort by size descending
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(sorted(c))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        unique_candidates.sort(key=len, reverse=True)
        
        return unique_candidates

    def lower_bound(remaining_edges_set, adj, n):
        """Lower bound on number of cliques needed to cover remaining edges."""
        if not remaining_edges_set:
            return 0
        
        # Each clique can cover at most (max_clique_size choose 2) edges
        # Find the maximum clique in the remaining graph
        # For a lower bound, we can use a greedy approach to estimate max clique size
        
        # Simple lower bound: each edge needs to be in at least one clique
        # A clique of size k covers k*(k-1)/2 edges
        # So we need at least ceil(|E| / max_possible_edges_per_clique)
        
        # Estimate max clique size using a simple greedy method
        # Actually, for a tighter bound, let's find a few large cliques
        
        # Very simple bound: if there are E edges, and the largest possible clique has size K,
        # then we need at least E / (K*(K-1)/2) cliques
        # But finding K is hard. Let's use a simpler bound.
        
        # Each vertex can be in multiple cliques, so this is tricky.
        # Alternative: use the fact that each clique has at least 1 edge, so LB = |E|
        # That's too weak.
        
        # Better: find the maximum degree in the remaining graph
        # A clique containing a vertex of degree d can have at most d+1 vertices
        # So max edges per clique is at most (d+1)*d/2
        
        # For now, use a simple bound: LB = number of edges / average clique size estimate
        # Let's just use LB = |E| / 10 as a very loose bound, or better, find a few cliques
        
        # Actually, let's use a proper lower bound based on edge coloring or similar
        # For simplicity, use: LB = ceil(|E| / max_possible_edges)
        # where max_possible_edges is estimated from the graph structure
        
        # Find approximate max clique size
        # Use a simple greedy max clique
        max_clique_size = 2
        vertices = list(set())
        for u, v in remaining_edges_set:
            vertices.add(u)
            vertices.add(v)
        
        if not vertices:
            return 0
        
        # Greedy max clique
        best_clique = set()
        for start_v in vertices:
            clique = {start_v}
            candidates = [v for v in vertices if v != start_v and v in adj[start_v] and (min(start_v,v), max(start_v,v)) in remaining_edges_set]
            while candidates:
                # Pick vertex with max connections to current clique
                best_c = None
                best_score = -1
                for c in candidates:
                    score = sum(1 for x in clique if c in adj[x] and (min(x,c), max(x,c)) in remaining_edges_set)
                    if score > best_score:
                        best_score = score
                        best_c = c
                if best_c is None:
                    break
                clique.add(best_c)
                candidates = [v for v in candidates if v != best_c and v in adj[best_c] and (min(v,best_c), max(v,best_c)) in remaining_edges_set]
            if len(clique) > len(best_clique):
                best_clique = clique
        
        max_k = max(2, len(best_clique))
        max_edges_per_clique = max_k * (max_k - 1) // 2
        num_edges = len(remaining_edges_set)
        
        lb = (num_edges + max_edges_per_clique - 1) // max_edges_per_clique
        return max(lb, 1)

    def branch_and_bound(edges, adj, n, time_budget):
        """Branch and bound to find minimum clique partition."""
        remaining = set(edges)
        best_solution = None
        best_count = len(edges)  # Trivial solution: each edge is its own clique
        
        # Get all edges sorted
        edge_list = sorted(remaining)
        
        # Precompute for speed
        def solve(remaining_set, current_cliques, count_so_far):
            nonlocal best_solution, best_count
            
            if time.time() > time_budget:
                return
            
            if not remaining_set:
                if count_so_far < best_count:
                    best_count = count_so_far
                    best_solution = current_cliques[:]
                return
            
            # Lower bound
            lb = lower_bound(remaining_set, adj, n)
            if count_so_far + lb >= best_count:
                return
            
            # Pick the first edge to branch on
            edge = min(remaining_set)
            u, v = edge
            
            # Get candidate cliques containing this edge
            candidates = get_cliques_containing_edge(edge, adj, remaining_set, n)
            
            # Sort by size descending (prefer larger cliques)
            candidates.sort(key=len, reverse=True)
            
            for clique in candidates:
                # Remove all edges in this clique from remaining
                new_remaining = set(remaining_set)
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        e = (min(a,b), max(a,b))
                        new_remaining.discard(e)
                
                if new_remaining == remaining_set:
                    continue  # No progress
                
                # Recurse
                solve(new_remaining, current_cliques + [clique], count_so_far + 1)
                
                if time.time() > time_budget:
                    return
        
        solve(remaining, [], 0)
        
        if best_solution is None:
            # Fallback: each edge as its own clique
            best_solution = [list(e) for e in edges]
            best_count = len(edges)
        
        return best_solution, best_count

    def greedy_initial(edges, adj, n):
        """Create a good initial solution using greedy approach."""
        remaining = set(edges)
        cliques = []
        
        # Sort edges by degree sum
        def edge_degree_sum(e):
            u, v = e
            return len(adj[u]) + len(adj[v])
        
        edge_list = sorted(remaining, key=edge_degree_sum, reverse=True)
        
        for edge in edge_list:
            if edge not in remaining:
                continue
            u, v = edge
            # Try to extend
            candidates = sorted(adj[u] & adj[v], key=lambda x: len(adj[x]), reverse=True)
            clique = [u, v]
            for w in candidates:
                can_add = True
                for x in clique:
                    if w not in adj[x]:
                        can_add = False
                        break
                    if (min(x,w), max(x,w)) not in remaining:
                        can_add = False
                        break
                if can_add:
                    clique.append(w)
            
            if len(clique) >= 2:
                cliques.append(clique)
                for i in range(len(clique)):
                    for j in range(i+1, len(clique)):
                        a, b = clique[i], clique[j]
                        remaining.discard((min(a,b), max(a,b)))
        
        # Add remaining edges
        for edge in remaining:
            cliques.append(list(edge))
        
        return cliques

    def local_search(cliques, edges, adj, n, time_budget):
        """Improve solution using local search."""
        current = cliques[:]
        current_count = len(current)
        best = current[:]
        best_count = current_count
        
        rng = random.Random(42)
        
        while time.time() < time_budget:
            # Try to merge two cliques
            if len(current) < 2:
                break
            
            i, j = rng.sample(range(len(current)), 2)
            c1, c2 = current[i], current[j]
            merged = list(set(c1) | set(c2))
            
            # Check if merged is a clique
            is_valid = True
            for a in merged:
                for b in merged:
                    if a >= b:
                        continue
                    if b not in adj[a]:
                        is_valid = False
                        break
                    if (a, b) not in set(edges):
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if not is_valid:
                continue
            
            # Check no edge duplication
            merged_edges = set()
            for a in merged:
                for b in merged:
                    if a < b:
                        merged_edges.add((a, b))
            
            other_edges = set()
            for k, clique in enumerate(current):
                if k == i or k == j:
                    continue
                for a in clique:
                    for b in clique:
                        if a < b:
                            other_edges.add((a, b))
            
            if merged_edges & other_edges:
                continue
            
            new_current = current[:]
            del new_current[max(i, j)]
            del new_current[min(i, j)]
            new_current.append(merged)
            
            if len(new_current) < best_count:
                best = new_current[:]
                best_count = len(new_current)
                current = new_current[:]
        
        return best

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges = generate_graph(n, p, seed)
        adj = build_adjacency(edges, n)
        
        # Get initial solution
        initial = greedy_initial(edges, adj, n)
        
        # Try branch and bound with time limit
        time_per_bb = min(30.0, (deadline - time.time()) / 2.0)
        if time.time() + time_per_bb < deadline:
            bb_solution, bb_count = branch_and_bound(edges, adj, n, time.time() + time_per_bb)
            
            # Use the better of BB and initial
            if bb_count < len(initial):
                current = bb_solution
            else:
                current = initial
        else:
            current = initial
        
        # Local search with remaining time
        remaining_time = deadline - time.time() - 5.0  # 5s safety
        if remaining_time > 1:
            improved = local_search(current, edges, adj, n, time.time() + remaining_time)
            results.append(improved)
        else:
            results.append(current)
    
    return results
