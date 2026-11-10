import numpy as np
import time
import random
import sys
import heapq
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 620  # Safety margin for 630s limit

    def generate_candidates(R):
        """Generate all integer vectors with squared norm <= R"""
        candidates = set()
        
        # Norm 1: e_i, -e_i
        for i in range(d):
            v = [0]*d
            v[i] = 1
            candidates.add(tuple(v))
            v[i] = -1
            candidates.add(tuple(v))
        
        # Norm 2: e_i +/- e_j
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        v = [0]*d
                        v[i] = s1
                        v[j] = s2
                        candidates.add(tuple(v))
        
        # Norm 3: e_i +/- e_j +/- e_k
        if R >= 3:
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = [0]*d
                                    v[i] = s1
                                    v[j] = s2
                                    v[k] = s3
                                    candidates.add(tuple(v))
        
        # Norm 4: 2e_i and e_i +/- e_j +/- e_k +/- e_l
        if R >= 4:
            # Type 1: 2e_i
            for i in range(d):
                v = [0]*d
                v[i] = 2
                candidates.add(tuple(v))
                v[i] = -2
                candidates.add(tuple(v))
            # Type 2: e_i +/- e_j +/- e_k +/- e_l
            for i in range(d):
                for j in range(i + 1, d):
                    for k in range(j + 1, d):
                        for l in range(k + 1, d):
                            for s1 in [1, -1]:
                                for s2 in [1, -1]:
                                    for s3 in [1, -1]:
                                        for s4 in [1, -1]:
                                            v = [0]*d
                                            v[i] = s1
                                            v[j] = s2
                                            v[k] = s3
                                            v[l] = s4
                                            candidates.add(tuple(v))
        
        # Convert to numpy array
        if not candidates:
            return np.array([], dtype=np.int64).reshape(0, d)
        
        arr = np.array(list(candidates), dtype=np.int64)
        return arr

    def solve_mis_branch_and_bound(adj, n, time_limit=10, init_bound=0):
        """
        Exact branch-and-bound for Maximum Independent Set.
        adj: adjacency matrix (n x n boolean)
        Returns the maximum independent set as a list of vertex indices.
        """
        if n == 0:
            return []
        
        # Use bitsets for efficiency
        # adj[i] is a bitset of neighbors of i
        adj_bits = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_bits[i] |= (1 << j)
        
        # Complement graph: independent set in G = clique in G_complement
        # But we'll work directly with independent set
        # A vertex v can be added to set S if no vertex in S is adjacent to v
        
        best_set = []
        best_size = init_bound
        
        # Upper bound: use greedy coloring of the remaining graph
        # For independent set, we can use the fact that in any independent set,
        # at most one vertex from each clique can be chosen
        # Simple bound: number of non-adjacent pairs
        
        def greedy_color_upper_bound(vertices):
            """
            Greedy coloring of the subgraph induced by vertices.
            The number of colors is an upper bound on the clique number,
            which is a lower bound on the chromatic number.
            For independent set in G, we need clique in G_complement.
            Actually, for MIS in G, we can use: |MIS| <= n - min_vertex_cover
            Simpler: use the bound that |MIS| <= sum over vertices of 1/(deg+1) is not tight.
            Let's use: in any independent set, if we order vertices, each vertex
            can "block" at most its non-neighbors... 
            Actually, a standard bound: |MIS(G)| <= |V| - tau(G) where tau is min vertex cover.
            But computing that is hard.
            
            Alternative: use the fact that if we have a set of vertices, the max independent set
            is at most the size of a maximal independent set found by greedy, but that's not an upper bound.
            
            Let's use a simple bound: the independence number is at most the number of vertices
            minus the size of a minimum vertex cover. A lower bound on vertex cover is the maximum
            matching size. But that's complex.
            
            Simpler approach: use the bound that in any independent set S, for each vertex v,
            v and its neighbors cannot all be in S. So |S| <= n - |N(S)|... not helpful.
            
            Let's just use: |MIS| <= n, and prune when current_size + len(remaining) <= best_size.
            For better pruning, we can use: if we have a clique C in the complement (i.e., independent set in G),
            then |MIS| <= |V| - |C| + 1... no.
            
            Actually, the standard bound for branch-and-bound MIS: 
            If we have a set of candidates C, and we know that any independent set from C
            has size at most |C|, and we have current set S, then |S| + |C| is an upper bound.
            We can tighten: if there's a vertex v in C that is non-adjacent to all in S, 
            then we can at least add v. But for upper bound, |S| + |C| is valid.
            
            For tighter bound: the independence number of the subgraph induced by C
            is at most the size of a greedy independent set on C, but that's a lower bound on alpha(C), not upper.
            
            Upper bound on alpha(C): use the fact that alpha(C) <= |C|, and also alpha(C) <= 
            sum over connected components of alpha(component).
            
            Let's just use |S| + |C| as the bound for now, and rely on good branching.
            """
            return len(vertices)
        
        def branch_and_bound(candidates_bitset, current_set, current_size):
            nonlocal best_set, best_size
            
            if time.time() > start_time + time_limit:
                return
            
            if candidates_bitset == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                return
            
            # Upper bound: current_size + number of remaining candidates
            remaining_count = bin(candidates_bitset).count('1')
            if current_size + remaining_count <= best_size:
                return
            
            # Choose a vertex to branch on
            # Heuristic: choose vertex with highest degree in the induced subgraph
            # Get list of candidate vertices
            candidates_list = [i for i in range(n) if (candidates_bitset >> i) & 1]
            
            if not candidates_list:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                return
            
            # Compute degrees in the induced subgraph
            best_vertex = candidates_list[0]
            best_deg = -1
            for v in candidates_list:
                # Count neighbors of v that are in candidates
                deg = 0
                for u in candidates_list:
                    if u != v and adj[v][u]:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_vertex = v
            
            v = best_vertex
            
            # Branch 1: include v
            new_set = current_set + [v]
            # Remove v and all its neighbors from candidates
            new_candidates = candidates_bitset & ~(1 << v)
            for u in range(n):
                if adj[v][u] and (candidates_bitset >> u) & 1:
                    new_candidates &= ~(1 << u)
            
            branch_and_bound(new_candidates, new_set, current_size + 1)
            
            # Branch 2: exclude v
            new_candidates = candidates_bitset & ~(1 << v)
            branch_and_bound(new_candidates, current_set, current_size)
        
        # Initial: all vertices are candidates
        all_candidates = (1 << n) - 1
        branch_and_bound(all_candidates, [], 0)
        
        return best_set

    def solve_mis(candidates, R, time_limit=10):
        """
        Find Maximum Independent Set in the conflict graph.
        Two vertices conflict if their squared distance < R.
        """
        n = len(candidates)
        if n < 2:
            return []
        
        # Compute norms
        norms_sq = np.sum(candidates**2, axis=1)
        
        # Filter for valid candidates (norm_sq <= R)
        valid_mask = norms_sq <= R
        if not np.all(valid_mask):
            candidates = candidates[valid_mask]
            norms_sq = norms_sq[valid_mask]
            n = len(candidates)
            if n < 2:
                return []
        
        # Compute distance matrix
        dots = candidates @ candidates.T
        dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
        
        # Conflict if dist_sq < R
        conflict_mask = dist_sq < R
        np.fill_diagonal(conflict_mask, False)
        
        # Convert to list of lists for branch and bound
        adj = [list(conflict_mask[i]) for i in range(n)]
        
        # First get a good initial solution with greedy
        deg = conflict_mask.sum(axis=1)
        sorted_indices = np.argsort(deg)
        available = np.ones(n, dtype=bool)
        selected = []
        
        while available.any():
            pick_idx = -1
            for idx in sorted_indices:
                if available[idx]:
                    pick_idx = idx
                    break
            
            if pick_idx == -1:
                break
            
            selected.append(pick_idx)
            available[pick_idx] = False
            neighbors = np.where(conflict_mask[pick_idx])[0]
            available[neighbors] = False
        
        init_bound = len(selected)
        
        # Now run branch and bound
        result = solve_mis_branch_and_bound(adj, n, time_limit=time_limit, init_bound=init_bound)
        
        return result

    # Main search loop
    best_set_global = []
    best_size_global = 0
    
    # Try different R values
    for R in [3, 4, 5]:
        if time.time() > deadline - 10:
            break
        
        candidates = generate_candidates(R)
        if len(candidates) < 2:
            continue
        
        # Solve MIS
        time_remaining = deadline - time.time()
        time_limit = max(10, min(120, time_remaining - 5))
        result_indices = solve_mis(candidates, R, time_limit=time_limit)
        
        if len(result_indices) > best_size_global:
            best_size_global = len(result_indices)
            best_set_global = [candidates[i] for i in result_indices]
    
    # Verify and return
    if best_size_global >= 2:
        result = np.array(best_set_global, dtype=np.int64)
        # Final verification
        max_norm = np.max(np.sum(result**2, axis=1))
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d2 = np.dot(result[i] - result[j], result[i] - result[j])
                if d2 < max_norm:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return result
    
    # Fallback
    return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

if __name__ == "__main__":
    res = kissing_number11()
    print(f"Size: {len(res)}")
