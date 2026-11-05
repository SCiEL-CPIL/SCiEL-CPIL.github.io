import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    N = 256
    # 0-indexed internally, 1-indexed in problem
    # adj[i] is bitset of neighbors of i in conflict graph G
    adj = [0] * N
    
    # Build conflict graph
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            # Check if (a*b) % (a+b) == 0
            if (a * b) % (a + b) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Precompute non_adj for G^c (complement graph)
    # non_adj[i] contains bits j such that j != i and j is NOT in adj[i]
    all_mask = (1 << N) - 1
    non_adj = [0] * N
    for i in range(N):
        # All bits except i and adj[i]
        non_adj[i] = all_mask ^ (1 << i) ^ adj[i]
        
    # Greedy heuristic to find initial large independent set
    # This is equivalent to finding a large clique in G^c
    # Greedy: pick vertex, add to clique, remove non-neighbors (in G^c) i.e. keep neighbors
    # Wait, in G^c, we want a clique.
    # Greedy Clique:
    # 1. Pick vertex v.
    # 2. Add v to clique.
    # 3. Candidates = Candidates & non_adj[v] (neighbors in G^c)
    # 4. Repeat
    
    def greedy_clique(candidates):
        clique = []
        while candidates:
            # Pick vertex with max degree in candidates? Or just first?
            # Let's pick the one with max degree in candidates to maximize future options
            # But calculating degree is O(N).
            # For initial bound, simple is fine.
            v = candidates & -candidates # lowest set bit
            v_idx = v.bit_length() - 1
            clique.append(v_idx)
            candidates = candidates & non_adj[v_idx]
        return len(clique)
        
    best = 0
    # Try greedy from each vertex? Or just one?
    # One greedy run is fast.
    best = greedy_clique(all_mask)
    
    # To get a better initial bound, we can try greedy with different orderings
    # But let's start with the search.
    
    # Tomita's Algorithm for Max Clique
    # State: candidates (bitset), current_clique_size
    # We need to store the best clique found? No, just the size is required for the score,
    # but the task asks to "Return a nonempty subset".
    # So we must return the set of vertices.
    
    best_clique = []
    
    def update_best(clique_list, size):
        nonlocal best, best_clique
        if size > best:
            best = size
            best_clique = clique_list[:]
            
    # We need to track the actual vertices for the solution.
    # Modifying the recursive function to pass the list of vertices.
    
    def search(clique_list, candidates, current_size):
        nonlocal best, best_clique
        if time.time() > deadline:
            return
            
        if candidates == 0:
            if current_size > best:
                best = current_size
                best_clique = clique_list[:]
            return
            
        # Color candidates
        # We need a list of (vertex_idx, color)
        col_list = []
        remaining = candidates
        color_idx = 0
        
        # Greedy coloring of the induced subgraph of G^c on 'candidates'
        # We want to assign colors such that adjacent vertices in G^c have different colors.
        # This partitions 'candidates' into independent sets of G^c (which are cliques in G).
        # Wait.
        # Coloring G^c: Adjacent in G^c -> Different Colors.
        # This means each color class is an Independent Set in G^c.
        # Independent Set in G^c = Clique in G.
        # So each color class is a Clique in G.
        # A Clique in G is a set of mutually conflicting vertices.
        # We are looking for a Clique in G^c (Independent Set in G).
        # In a Clique in G^c, all vertices are adjacent in G^c.
        # Therefore, they must have different colors in any valid coloring of G^c.
        # So the size of the Clique in G^c is at most the number of colors used.
        # Yes, this bound is valid.
        
        while remaining:
            color_idx += 1
            # Pick a vertex to assign this color
            # Heuristic: pick vertex with max degree in 'remaining' subgraph of G^c
            # To save time, pick the first one.
            v_bit = remaining & -remaining
            v_idx = v_bit.bit_length() - 1
            
            # Remove v and its neighbors in G^c from 'remaining'
            # Neighbors in G^c are non_adj[v_idx]
            remaining = remaining & ~(non_adj[v_idx] | v_bit)
            
            col_list.append((v_idx, color_idx))
            
        # Sort by color descending
        # Vertices with higher colors are more "constrained" or part of larger potential cliques?
        # Standard Tomita sorts by color descending.
        col_list.sort(key=lambda x: x[1], reverse=True)
        
        for v_idx, c in col_list:
            # Upper bound check
            if current_size + c <= best:
                break
                
            # Include v_idx
            # New candidates: neighbors of v_idx in G^c that are in current candidates
            # Note: We must use the 'candidates' from BEFORE removing v_idx in the loop?
            # In Tomita's algorithm, we iterate through the colored list.
            # The candidates for the recursive call are the intersection of the current candidate set
            # and the neighbors of v in G^c.
            # However, we have been removing vertices from 'candidates' as we iterate?
            # No, standard implementation:
            # The loop iterates through the list.
            # The 'candidates' variable is updated by removing the current vertex after processing.
            # But the recursive call should use the neighbors of v within the *original* candidate set
            # minus the vertices already processed?
            # Actually, since we process in order, and we remove v from candidates,
            # the neighbors of v that are still in 'candidates' are the valid next steps.
            
            new_candidates = candidates & non_adj[v_idx]
            
            search(clique_list + [v_idx], new_candidates, current_size + 1)
            
            # Remove v_idx from candidates for subsequent iterations in this loop
            candidates = candidates & ~(1 << v_idx)
            
            # Check time
            if time.time() > deadline:
                return
                
    # Initialize search
    search([], all_mask, 0)
    
    # If best_clique is empty (should not happen as N>=1), return [0]
    if not best_clique:
        return [1]
        
    # Convert 0-indexed to 1-indexed
    return [x + 1 for x in best_clique]

# EVOLVE-BLOCK-END
