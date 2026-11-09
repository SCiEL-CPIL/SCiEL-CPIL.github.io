"""Stronger algorithm for dense K3,3-free graphs on 20 vertices."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20
    
    # Polarity graph seed: PG(2,4) has 21 points, we take 20
    # The polarity graph of PG(2,4) is K3,3-free and dense
    # Construction: vertices are 1D subspaces of GF(4)^3
    # GF(4) = {0,1,2,3} with 2^2=0, 2^3=1, etc.
    # We use a known polarity: (x,y,z) -> (x, y, z) with form xz + y^2
    # Vertices: all nonzero vectors in GF(4)^3 modulo scalar, grouped by polarity
    
    # Simpler approach: use a circulant-like structure based on quadratic residues
    # For n=20, use edges (i,j) where (j-i) mod 20 is in a carefully chosen set
    # This creates a regular graph that's often K3,3-free
    
    def polarity_seed():
        """Create a structured seed based on polarity graph ideas."""
        adj = [0] * n
        # Use a set of differences that gives a regular graph
        # For K3,3-freedom, we need no three vertices with 3 common neighbors
        # Try: connect i to i+d mod 20 for d in {1,2,4,5,8,10,12,15,16,18,19}
        # This gives degree ~11
        diffs = {1, 2, 4, 5, 8, 10, 12, 15, 16, 18, 19}
        for i in range(n):
            for d in diffs:
                j = (i + d) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Fast K3,3-free check using bitsets
    def is_k33_free(adj_bits):
        """Fast K3,3 check: no three vertices share 3+ common neighbors."""
        # Precompute common neighbor bitsets for all pairs
        for i in range(n):
            for j in range(i+1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                # Check if any other vertex k shares 3+ neighbors with both i and j
                # i.e., |N(i) ∩ N(j) ∩ N(k)| >= 3
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def count_edges(adj_bits):
        return sum(b.bit_count() for b in adj_bits) // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def get_missing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i+1, n)
                if not (adj_bits[i] >> j) & 1]

    def local_search(adj_bits, deadline):
        """Enhanced hill climbing with degree-prioritized 2-opt and 3-opt."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        
        # Precompute missing and existing edges
        missing = get_missing(current)
        random.shuffle(missing)
        existing = [(i, j) for i in range(n) for j in range(i+1, n)
                    if (current[i] >> j) & 1]
        
        # Degree array for prioritization
        def get_degrees(adj):
            return [adj[i].bit_count() for i in range(n)]
        
        improved = True
        rounds = 0
        while improved and time.time() < deadline - 5:
            improved = False
            rounds += 1
            random.shuffle(missing)
            
            # Priority: try adding edges between low-degree vertices first
            degs = get_degrees(current)
            missing.sort(key=lambda e: (degs[e[0]] + degs[e[1]]))
            
            # Try adding edges (prioritized)
            new_missing = []
            for u, v in missing:
                if time.time() > deadline - 5:
                    new_missing.append((u, v))
                    continue
                add_edge(current, u, v)
                if is_k33_free(current):
                    current_edges += 1
                    improved = True
                    existing.append((u, v))
                else:
                    remove_edge(current, u, v)
                    new_missing.append((u, v))
            
            if improved:
                missing = new_missing
                continue
            
            # 2-opt: swap one existing edge for one missing edge
            if existing and missing and time.time() < deadline - 5:
                # Prioritize swaps that balance degrees
                degs = get_degrees(current)
                swap_attempts = 0
                max_swaps = 500 if rounds < 10 else 200
                while swap_attempts < max_swaps and time.time() < deadline - 5:
                    swap_attempts += 1
                    # Pick existing edge from high-degree vertices, missing from low-degree
                    e1 = random.choice(existing)
                    e2 = random.choice(missing)
                    u1, v1 = e1
                    u2, v2 = e2
                    
                    remove_edge(current, u1, v1)
                    add_edge(current, u2, v2)
                    if is_k33_free(current):
                        existing.remove(e1)
                        existing.append(e2)
                        missing.remove(e2)
                        missing.append(e1)
                        improved = True
                    else:
                        remove_edge(current, u2, v2)
                        add_edge(current, u1, v1)
            
            # 3-opt: remove 2 existing edges, add 2 missing edges
            if len(existing) >= 2 and len(missing) >= 2 and time.time() < deadline - 5:
                three_opt_attempts = 0
                max_3opt = 100 if rounds < 10 else 50
                while three_opt_attempts < max_3opt and time.time() < deadline - 5:
                    three_opt_attempts += 1
                    e1 = random.choice(existing)
                    e2 = random.choice([e for e in existing if e != e1])
                    m1 = random.choice(missing)
                    m2 = random.choice([m for m in missing if m != m1])
                    
                    u1, v1 = e1
                    u2, v2 = e2
                    w1, x1 = m1
                    w2, x2 = m2
                    
                    remove_edge(current, u1, v1)
                    remove_edge(current, u2, v2)
                    add_edge(current, w1, x1)
                    add_edge(current, w2, x2)
                    if is_k33_free(current):
                        existing.remove(e1)
                        existing.remove(e2)
                        existing.extend([m1, m2])
                        missing.remove(m1)
                        missing.remove(m2)
                        missing.extend([e1, e2])
                        improved = True
                    else:
                        remove_edge(current, w1, x1)
                        remove_edge(current, w2, x2)
                        add_edge(current, u1, v1)
                        add_edge(current, u2, v2)
        
        return current, current_edges

    def init_random_greedy(deadline):
        """Build a random maximal K3,3-free graph via greedy edge addition."""
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline - 10:
                break
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    def init_bipartite_random(deadline):
        """Initialize with a random bipartite graph (naturally K3,3-free if we're careful,
        but a complete bipartite K10,10 has K3,3, so we need sparse bipartite)."""
        adj = [0] * n
        A = list(range(10))
        B = list(range(10, 20))
        # Add random edges between A and B
        for a in A:
            # Each vertex in A connects to a random subset of B
            for b in B:
                if random.random() < 0.4:
                    add_edge(adj, a, b)
        return adj

    def init_complement_of_turan_like(deadline):
        """Try a structure based on Turan-type ideas."""
        adj = [0] * n
        # Start with a random graph and prune
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.6:
                    add_edge(adj, i, j)
        # Prune to make K3,3-free
        # Simple approach: remove edges that create K3,3
        # This is a rough initialization; local search will fix it
        return adj

    best_adj = None
    best_edges = -1

    # Restart 1: Polarity graph seed (structured, likely K3,3-free or close)
    if time.time() < deadline - 30:
        adj1 = polarity_seed()
        # Verify and clean if needed
        if not is_k33_free(adj1):
            # Prune: remove edges that create K3,3
            all_edges1 = [(i, j) for i in range(n) for j in range(i+1, n)
                          if (adj1[i] >> j) & 1]
            random.shuffle(all_edges1)
            adj1_clean = [0] * n
            for u, v in all_edges1:
                add_edge(adj1_clean, u, v)
                if not is_k33_free(adj1_clean):
                    remove_edge(adj1_clean, u, v)
            adj1 = adj1_clean
        adj1, e1 = local_search(adj1, deadline - 20)
        if e1 > best_edges:
            best_edges = e1
            best_adj = adj1[:]

    # Restart 2: Random greedy from empty
    if time.time() < deadline - 25:
        adj2 = init_random_greedy(deadline)
        adj2, e2 = local_search(adj2, deadline - 15)
        if e2 > best_edges:
            best_edges = e2
            best_adj = adj2[:]

    # Restart 3: Perturbed polarity seed
    if time.time() < deadline - 20:
        adj3 = polarity_seed()
        # Randomly perturb: remove some edges, add others
        all_edges3 = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges3)
        adj3_perturbed = [0] * n
        for u, v in all_edges3:
            if time.time() > deadline - 15:
                break
            if random.random() < 0.7:  # Keep ~70% of edges
                add_edge(adj3_perturbed, u, v)
                if not is_k33_free(adj3_perturbed):
                    remove_edge(adj3_perturbed, u, v)
        adj3, e3 = local_search(adj3_perturbed, deadline - 10)
        if e3 > best_edges:
            best_edges = e3
            best_adj = adj3[:]

    # Restart 4: Another random greedy
    if time.time() < deadline - 15:
        adj4 = init_random_greedy(deadline)
        adj4, e4 = local_search(adj4, deadline - 8)
        if e4 > best_edges:
            best_edges = e4
            best_adj = adj4[:]

    # Restart 5: Dense random start with pruning
    if time.time() < deadline - 12:
        adj5 = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    add_edge(adj5, i, j)
        all_edges5 = [(i, j) for i in range(n) for j in range(i+1, n)
                      if (adj5[i] >> j) & 1]
        random.shuffle(all_edges5)
        adj5_clean = [0] * n
        for u, v in all_edges5:
            add_edge(adj5_clean, u, v)
            if not is_k33_free(adj5_clean):
                remove_edge(adj5_clean, u, v)
        adj5, e5 = local_search(adj5_clean, deadline - 5)
        if e5 > best_edges:
            best_edges = e5
            best_adj = adj5[:]

    # Final polish on the best
    if best_adj is not None and time.time() < deadline - 2:
        best_adj, best_edges = local_search(best_adj, deadline - 2)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result

# EVOLVE-BLOCK-END
