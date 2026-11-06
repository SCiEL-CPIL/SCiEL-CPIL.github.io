"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 20
    
    def popcount(x):
        """Fast popcount for Python integers."""
        return bin(x).count('1')
    
    def can_add(adj_bits, u, v):
        """
        Check if adding edge (u,v) to graph with adjacency bitmasks adj_bits
        maintains K2,3-freeness.
        
        adj_bits[i] is a bitmask of neighbors of vertex i.
        
        Conditions:
        1. |N(u) ∩ N(v)| <= 2 (unchanged by adding (u,v) itself)
        2. For each w in N(v): |N(u) ∩ N(w)| + 1 <= 2, i.e., |N(u) ∩ N(w)| <= 1
        3. For each w in N(u): |N(v) ∩ N(w)| + 1 <= 2, i.e., |N(v) ∩ N(w)| <= 1
        """
        # Condition 1: common neighbors of u and v
        common_uv = popcount(adj_bits[u] & adj_bits[v])
        if common_uv > 2:
            return False
        
        # Condition 2: for each w in N(v), w != u
        nv = adj_bits[v]
        w = nv
        while w:
            low = w & (-w)
            wi = low.bit_length() - 1
            w ^= low
            if wi == u:
                continue
            # N(u) gains v, and v~w, so v becomes common neighbor of u and w
            common_uw = popcount(adj_bits[u] & adj_bits[wi])
            if common_uw > 1:
                return False
        
        # Condition 3: for each w in N(u), w != v
        nu = adj_bits[u]
        w = nu
        while w:
            low = w & (-w)
            wi = low.bit_length() - 1
            w ^= low
            if wi == v:
                continue
            common_vw = popcount(adj_bits[v] & adj_bits[wi])
            if common_vw > 1:
                return False
        
        return True
    
    def count_tight_pairs(adj_bits):
        """
        Count the number of pairs (i,j) with exactly 2 common neighbors.
        These are 'tight' pairs that will block future edge additions.
        """
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                common = popcount(adj_bits[i] & adj_bits[j])
                if common == 2:
                    count += 1
        return count
    
    def greedy_min_tight(adj_bits, edges):
        """
        Greedily add edges choosing the one that minimizes the increase in tight pairs.
        This is the key heuristic: prioritize edges that keep the graph 'flexible'.
        """
        improved = True
        while improved:
            improved = False
            best_edge = None
            best_tight_delta = float('inf')
            best_tight_count = float('inf')
            
            for u in range(n):
                for v in range(u+1, n):
                    if adj_bits[u] & (1 << v):
                        continue  # already an edge
                    
                    if not can_add(adj_bits, u, v):
                        continue
                    
                    # Simulate adding the edge and count tight pairs
                    old_tight = count_tight_pairs(adj_bits)
                    
                    # Try adding
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    
                    new_tight = count_tight_pairs(adj_bits)
                    
                    # Revert
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    
                    delta = new_tight - old_tight
                    if delta < best_tight_delta or (delta == best_tight_delta and new_tight < best_tight_count):
                        best_tight_delta = delta
                        best_tight_count = new_tight
                        best_edge = (u, v)
            
            if best_edge is not None:
                u, v = best_edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
                improved = True
        
        return edges, adj_bits
    
    def greedy_random(adj_bits, edges, rng):
        """Standard random greedy fill."""
        all_missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not (adj_bits[i] & (1 << j))]
        rng.shuffle(all_missing)
        for u, v in all_missing:
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
        return edges, adj_bits
    
    def local_search_swap(adj_bits, edges, rng, max_iters=50):
        """
        Local search: try removing one edge and adding two different edges.
        Or: remove an edge, then do greedy fill to see if we get more edges.
        """
        best_edges = list(edges)
        best_adj = list(adj_bits)
        best_count = len(best_edges)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            cur_adj = list(adj_bits)
            cur_edges = list(edges)
            
            # Remove a random edge
            if not cur_edges:
                break
            rem_idx = rng.randint(0, len(cur_edges) - 1)
            u, v = cur_edges[rem_idx]
            cur_adj[u] &= ~(1 << v)
            cur_adj[v] &= ~(1 << u)
            cur_edges.pop(rem_idx)
            
            # Try greedy fill with min-tight heuristic
            cur_edges, cur_adj = greedy_min_tight(cur_adj, cur_edges)
            
            if len(cur_edges) > best_count:
                best_count = len(cur_edges)
                best_edges = cur_edges
                best_adj = cur_adj
                # Update the working state
                adj_bits = cur_adj
                edges = cur_edges
            else:
                # Revert
                adj_bits = list(best_adj)
                edges = list(best_edges)
                # If we can't improve, try a different removal
                continue
        
        return edges, adj_bits, best_count
    
    def structured_construction():
        """
        Use a structured approach based on finite field / polarity graph ideas.
        For K2,3-free graphs, we can use the construction from finite geometry.
        
        One approach: partition vertices into groups and use a regular structure.
        For n=20, try a (5,5)-bipartite-like structure or use the fact that
        K2,3-free graphs relate to bipartite Ramsey numbers.
        
        Actually, let's try a simpler structured approach:
        - Start with a cycle C20 (20 edges)
        - Add chords carefully
        """
        adj_bits = [0] * n
        edges = []
        
        # Add a 20-cycle
        for i in range(n):
            j = (i + 1) % n
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
            edges.append([i, j])
        
        # Now add chords using min-tight greedy
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        return edges, adj_bits
    
    def structured_complement():
        """
        Try a different structure: start with a complete bipartite-like base
        and add within-part edges carefully.
        """
        adj_bits = [0] * n
        edges = []
        
        # Split into two groups of 10
        group_a = list(range(10))
        group_b = list(range(10, 20))
        
        # Add a perfect matching between groups
        for i in range(10):
            u = group_a[i]
            v = group_b[i]
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges.append([u, v])
        
        # Add within-group edges (paths or cycles)
        for i in range(9):
            u = group_a[i]
            v = group_a[i+1]
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges.append([u, v])
        
        for i in range(9):
            u = group_b[i]
            v = group_b[i+1]
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            edges.append([u, v])
        
        # Fill with min-tight greedy
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        return edges, adj_bits
    
    # Main strategy: multiple approaches, keep the best
    best_edges_overall = []
    best_count_overall = 0
    
    # Approach 1: Pure min-tight greedy from empty
    if time.time() < deadline - 20:
        adj_bits = [0] * n
        edges = []
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 2: Structured (cycle base) + min-tight fill
    if time.time() < deadline - 20:
        edges, adj_bits = structured_construction()
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 3: Structured (bipartite base) + min-tight fill
    if time.time() < deadline - 20:
        edges, adj_bits = structured_complement()
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 4: Multiple random restarts with min-tight greedy
    rng = random.Random(42)
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        adj_bits = [0] * n
        edges = []
        
        # Random initial seed edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_edges)
        
        # Add a random subset first
        num_seed = rng.randint(10, 25)
        for u, v in all_edges[:num_seed]:
            if can_add(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
        
        # Min-tight greedy fill
        edges, adj_bits = greedy_min_tight(adj_bits, edges)
        
        # Local search
        if time.time() < deadline - 5:
            edges, adj_bits, _ = local_search_swap(adj_bits, edges, rng, max_iters=20)
        
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Approach 5: More random restarts with pure random greedy (fallback)
    for r in range(30):
        if time.time() > deadline - 5:
            break
        
        adj_bits = [0] * n
        edges = []
        edges, adj_bits = greedy_random(adj_bits, edges, rng)
        
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_edges_overall = edges
    
    # Final local search on the best solution
    if time.time() < deadline - 3 and best_edges_overall:
        # Reconstruct adj_bits from best_edges_overall
        adj_bits = [0] * n
        for u, v in best_edges_overall:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        best_edges_overall, adj_bits, _ = local_search_swap(
            adj_bits, best_edges_overall, rng, max_iters=30
        )
    
    return best_edges_overall
# EVOLVE-BLOCK-END
