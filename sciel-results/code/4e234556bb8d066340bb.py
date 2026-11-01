"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from functools import lru_cache

def construct(q, n, radius):
    # Deadline guard: keep a safety margin before the 120s evaluator limit.
    deadline = time.time() + 110.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: greedy set cover.
        return _greedy_set_cover(q, n, radius, deadline)

    # For q=3, n=6, radius=1: use branch-and-bound for maximum independent set
    # in the graph where vertices are 729 words, edges connect pairs at distance 1.
    # A covering code is a maximal independent set; we want the largest one.
    
    total_words = q ** n  # 729
    
    # Precompute adjacency: adj[i] = set of words at distance 1 from i
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Build adjacency list
    adj = [set() for _ in range(total_words)]
    for i in range(total_words):
        w = all_words[i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    j = word_to_idx(nw)
                    adj[i].add(j)
    
    # Convert to sorted lists for faster iteration
    adj_list = [sorted(adj[i]) for i in range(total_words)]
    
    # Symmetry breaking: fix first codeword to all-zeros (index 0)
    # We'll search for maximum independent set containing vertex 0
    
    # State: (remaining_vertices_frozenset, current_size)
    # Use memoization with LRU cache
    
    # For branch-and-bound, we need:
    # 1. Upper bound on remaining independent set size
    # 2. Greedy lower bound
    
    # Upper bound: for a set of vertices, the maximum independent set size 
    # is at most the sum of 1/(1+max_degree_in_induced_subgraph) for each vertex
    # Simpler: use greedy coloring bound
    
    def greedy_color_bound(vertices):
        """Upper bound on independent set size using greedy coloring."""
        if not vertices:
            return 0
        
        # Greedy coloring of the induced subgraph
        # Color classes give an upper bound on independent set size
        uncolored = set(vertices)
        color_count = 0
        
        while uncolored:
            color_count += 1
            # Find a maximal independent set in uncolored
            current_color = set()
            remaining = set(uncolored)
            
            while remaining:
                # Pick vertex with minimum degree in remaining
                min_deg = float('inf')
                pick = None
                for v in remaining:
                    deg_in_remaining = len(adj[v] & remaining)
                    if deg_in_remaining < min_deg:
                        min_deg = deg_in_remaining
                        pick = v
                
                if pick is None:
                    break
                    
                current_color.add(pick)
                remaining.remove(pick)
                # Remove neighbors of pick from remaining
                remaining -= adj[pick]
            
            uncolored -= current_color
        
        return color_count
    
    # Better upper bound using clique cover or simpler degree-based
    def degree_bound(vertices):
        """Upper bound based on maximum degree in induced subgraph."""
        if not vertices:
            return 0
        
        # Simple bound: |V| / (1 + avg_degree)
        v_list = list(vertices)
        if not v_list:
            return 0
        
        total_deg = 0
        for v in v_list:
            total_deg += len(adj[v] & set(vertices))
        
        avg_deg = total_deg / len(v_list)
        return int(len(v_list) / (1 + avg_deg)) + 1
    
    # Branch and bound
    best_size = 0
    best_set = set()
    
    def branch_and_bound(remaining, current_set, current_size):
        nonlocal best_size, best_set, deadline
        
        if time.time() > deadline - 5:
            return
        
        if not remaining:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_set)
            return
        
        # Upper bound
        ub = current_size + degree_bound(remaining)
        if ub <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with maximum degree in remaining
        v_list = list(remaining)
        max_deg = -1
        pivot = None
        for v in v_list:
            deg = len(adj[v] & remaining)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return
        
        # Branch 1: include pivot
        new_remaining = remaining - adj[pivot] - {pivot}
        branch_and_bound(new_remaining, current_set | {pivot}, current_size + 1)
        
        # Branch 2: exclude pivot
        new_remaining = remaining - {pivot}
        branch_and_bound(new_remaining, current_set, current_size)
    
    # Start with vertex 0 (all-zeros) included due to symmetry
    initial_remaining = set(range(1, total_words)) - adj[0]
    branch_and_bound(initial_remaining, {0}, 1)
    
    # Convert best_set to list of codewords
    if best_set:
        result = [all_words[i] for i in sorted(best_set)]
        return result
    
    # Fallback if B&B didn't find anything good
    return _greedy_set_cover(q, n, radius, deadline)


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n
    
    # Precompute neighbors within radius
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    # For each word, find all words within radius
    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered
    
    uncovered = set(range(total))
    code = []
    
    while uncovered and time.time() < deadline - 5:
        # Find word that covers most uncovered
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(all_words[best_i])
        uncovered -= coverage[best_i]
    
    return code

def _improve_with_local_search(code, q, n, radius, deadline):
    """Local search improvement: try to remove redundant codewords."""
    if not code:
        return code
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    total = q ** n
    all_words = [idx_to_word(i) for i in range(total)]
    
    # Build adjacency (distance 1)
    adj = [set() for _ in range(total)]
    for i in range(total):
        w = all_words[i]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    j = word_to_idx(nw)
                    adj[i].add(j)
    
    # Convert code to set of indices
    code_set = set(word_to_idx(cw) for cw in code)
    
    # Coverage count
    coverage_count = [0] * total
    for idx in code_set:
        coverage_count[idx] += 1
        for nb in adj[idx]:
            coverage_count[nb] += 1
    
    # Iteratively remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in list(code_set):
            # Check if removable
            removable = True
            for w_idx in [idx] + list(adj[idx]):
                if coverage_count[w_idx] <= 1:
                    removable = False
                    break
            
            if removable:
                # Remove it
                coverage_count[idx] -= 1
                for nb in adj[idx]:
                    coverage_count[nb] -= 1
                code_set.remove(idx)
                improved = True
                break
    
    return [all_words[i] for i in sorted(code_set)]
# EVOLVE-BLOCK-END
