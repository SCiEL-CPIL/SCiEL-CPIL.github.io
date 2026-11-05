import time
import math
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            bad_triples.append((a, b, c))
    
    # Build witness conflict graph
    # Edge (u, v) exists if there is a w such that {u, v, w} is a bad triple
    # Use 0-indexed: vertex i represents number i+1
    # adj[i] is a set of neighbors of vertex i
    adj = [set() for _ in range(N)]
    
    for a, b, c in bad_triples:
        a_idx, b_idx, c_idx = a - 1, b - 1, c - 1
        adj[a_idx].add(b_idx)
        adj[a_idx].add(c_idx)
        adj[b_idx].add(a_idx)
        adj[b_idx].add(c_idx)
        adj[c_idx].add(a_idx)
        adj[c_idx].add(b_idx)
    
    # Convert adj to sorted lists for faster iteration
    adj_sorted = [sorted(adj[i]) for i in range(N)]
    
    # Maximum Independent Set via Branch and Bound (Tomita's algorithm style)
    # We want to find the maximum independent set in the graph
    
    # Use bitset representation for speed
    # Python integers can serve as bitsets
    # vertex i corresponds to bit i
    
    # Precompute neighbor bitsets
    neighbor_bits = [0] * N
    for i in range(N):
        bits = 0
        for j in adj_sorted[i]:
            bits |= (1 << j)
        neighbor_bits[i] = bits
    
    best_size = 0
    best_set = [False] * N
    current_set = [False] * N
    
    # Greedy coloring for upper bound
    # We color the subgraph induced by candidate vertices
    # The number of colors gives an upper bound on the independent set size
    
    def greedy_color(candidates):
        """
        candidates: list of vertex indices
        Returns: list of colors (one per vertex in candidates)
        """
        if not candidates:
            return []
        
        # Order by degree in the subgraph (descending)
        subgraph_degree = {}
        for v in candidates:
            deg = 0
            for u in candidates:
                if u != v and u in adj[v]:
                    deg += 1
            subgraph_degree[v] = deg
        
        # Sort by degree descending
        ordered = sorted(candidates, key=lambda v: -subgraph_degree[v])
        
        colors = {}
        next_color = 1
        for v in ordered:
            # Find the smallest color not used by neighbors
            used = set()
            for u in adj_sorted[v]:
                if u in colors:
                    used.add(colors[u])
            while next_color in used:
                next_color += 1
            colors[v] = next_color
            next_color += 1
        
        # Convert to list in original order
        color_list = [colors[v] for v in candidates]
        return color_list
    
    # Better: use a proper greedy coloring that minimizes colors
    def greedy_color_min(candidates):
        """
        Greedy coloring that tries to minimize the number of colors.
        Returns: max_color used
        """
        if not candidates:
            return 0
        
        # Order by degree in subgraph (descending)
        cand_set = set(candidates)
        subgraph_degree = {}
        for v in candidates:
            deg = 0
            for u in cand_set:
                if u != v and u in adj[v]:
                    deg += 1
            subgraph_degree[v] = deg
        
        # Sort by degree descending
        ordered = sorted(candidates, key=lambda v: -subgraph_degree[v])
        
        colors = {}
        for v in ordered:
            used = set()
            for u in adj_sorted[v]:
                if u in colors:
                    used.add(colors[u])
            # Find smallest available color
            c = 1
            while c in used:
                c += 1
            colors[v] = c
        
        return max(colors.values()) if colors else 0
    
    # Branch and Bound
    # candidates: set of vertices that can still be added
    # current: set of vertices already in the independent set
    def branch_and_bound(candidates, current_size, current_set_local):
        nonlocal best_size, best_set, deadline, start_time
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set_local)
            return
        
        # Upper bound: current_size + number of candidates
        # Better bound: current_size + greedy_color(candidates)
        if current_size + len(candidates) <= best_size:
            return
        
        # Get greedy coloring bound
        color_bound = greedy_color_min(candidates)
        if current_size + color_bound <= best_size:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in candidate subgraph
        cand_list = list(candidates)
        cand_set = set(candidates)
        
        # Compute degrees in candidate subgraph
        degs = {}
        for v in cand_list:
            deg = 0
            for u in cand_list:
                if u != v and u in adj[v]:
                    deg += 1
            degs[v] = deg
        
        # Pick vertex with max degree
        v = max(cand_list, key=lambda x: degs[x])
        
        # Branch 1: Include v
        # Remove v and its neighbors from candidates
        new_candidates = set(candidates)
        new_candidates.remove(v)
        for u in adj_sorted[v]:
            if u in new_candidates:
                new_candidates.remove(u)
        
        current_set_local[v] = True
        branch_and_bound(new_candidates, current_size + 1, current_set_local)
        current_set_local[v] = False
        
        # Branch 2: Exclude v
        new_candidates2 = set(candidates)
        new_candidates2.remove(v)
        branch_and_bound(new_candidates2, current_size, current_set_local)
    
    # Initial greedy solution
    def greedy_initial():
        """Greedy independent set: pick vertices in order of degree (ascending)."""
        selected = [False] * N
        remaining = set(range(N))
        
        # Order by degree ascending
        degs = [len(adj[i]) for i in range(N)]
        order = sorted(range(N), key=lambda i: degs[i])
        
        for v in order:
            if v not in remaining:
                continue
            # Check if v is compatible with all selected vertices
            can_add = True
            for u in range(N):
                if selected[u] and u in adj[v]:
                    can_add = False
                    break
            if can_add:
                selected[v] = True
                remaining.remove(v)
        
        return selected
    
    # Get initial greedy solution
    greedy_sol = greedy_initial()
    best_size = sum(greedy_sol)
    best_set = list(greedy_sol)
    
    # Run branch and bound
    # Start with all vertices as candidates
    candidates = set(range(N))
    current_set_local = [False] * N
    
    branch_and_bound(candidates, 0, current_set_local)
    
    # Convert best_set to list of numbers
    if best_size > 0:
        result = [i + 1 for i in range(N) if best_set[i]]
    else:
        result = [1]
    
    return result
