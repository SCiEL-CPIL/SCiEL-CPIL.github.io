"""
Strategy: ADMM-based global optimization for C6-free graphs.
We model the graph as a binary adjacency matrix X.
Objective: Maximize sum(X) subject to C6-free constraint.
Method:
1. Initialize X with a random or heuristic graph.
2. ADMM Iteration:
   a. Update X via gradient ascent on the objective (sum of edges) + penalty for C6s.
   b. Project X onto the feasible set (C6-free graphs) using a greedy repair.
   c. Update Lagrange multipliers.
3. The "Projection" step uses the fast bitset C6 detector to remove edges that create C6s.
4. Final local search to maximize edges.
"""

import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    n = 40
    
    def build_adjacency(edge_list):
        """Build adjacency matrix (numpy bool array) and bitsets."""
        adj = np.zeros((n, n), dtype=bool)
        adj_bits = [0] * n
        for u, v in edge_list:
            adj[u, v] = True
            adj[v, u] = True
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        return adj, adj_bits

    def count_c6s(adj_bits):
        """
        Count the number of C6s in the graph.
        This is used as a penalty term in the ADMM objective.
        For efficiency, we only count a subset or use a fast approximation if needed, 
        but for n=40, we can afford a relatively fast count.
        Actually, counting all C6s is O(n^6) worst case, which is too slow.
        Instead, we will use the "has_c6" check for specific edges during repair.
        For the ADMM penalty, we will use a proxy: the number of "almost-C6" configurations 
        or simply rely on the projection step to ensure validity.
        
        Revised Strategy:
        The ADMM step will try to add edges based on a "potential" score.
        The projection step will remove edges that create C6s.
        We don't need to count all C6s in the objective, we just need the projection to be effective.
        So, the ADMM update will be:
        X_new = X_old + alpha * (gradient of objective - penalty * gradient of C6_count)
        Since counting C6s is hard, we will use a simpler heuristic for the gradient:
        Prefer edges that connect low-degree nodes (to avoid creating dense clusters that form C6s).
        """
        pass

    def has_c6_bitset(adj_bits, u, v):
        """
        Check if adding edge (u,v) creates a C6.
        Path: u-w-x-y-z-v with all distinct.
        """
        mask_all = (1 << n) - 1
        mask_uv = mask_all ^ (1 << u) ^ (1 << v)
        
        nu = adj_bits[u] & mask_uv
        if nu == 0:
            return False
        
        nv = adj_bits[v] & mask_uv
        
        w = nu
        while w:
            lb = w & (-w)
            w ^= lb
            wi = lb.bit_length() - 1
            if wi == v:
                continue
            nw = adj_bits[wi] & mask_uv & ~(1 << wi)
            if nw == 0:
                continue
            x = nw
            while x:
                lbx = x & (-x)
                x ^= lbx
                xi = lbx.bit_length() - 1
                nx = adj_bits[xi] & mask_uv & ~(1 << wi) & ~(1 << xi)
                if nx == 0:
                    continue
                y = nx
                while y:
                    lby = y & (-y)
                    y ^= lby
                    yi = lby.bit_length() - 1
                    ny = adj_bits[yi] & mask_uv & ~(1 << wi) & ~(1 << xi) & ~(1 << yi)
                    if ny & nv:
                        return True
        return False

    def project_to_c6_free(adj_bits, edges_to_add):
        """
        Given a set of edges to add, try to add as many as possible without creating C6s.
        This is the 'Projection' step in ADMM.
        """
        added = []
        for u, v in edges_to_add:
            if time.time() > deadline:
                break
            # Check if edge already exists
            if adj_bits[u] & (1 << v):
                continue
            # Check if adding creates C6
            if not has_c6_bitset(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                added.append((u, v))
        return added, adj_bits

    def run_admm(seed_val):
        """
        Run ADMM-style optimization.
        1. Start with a sparse graph (e.g., random or 8 K5s).
        2. Iteratively propose edges based on a score (degree balance + random).
        3. Project by adding valid edges.
        4. Repeat until time limit or convergence.
        """
        random.seed(seed_val)
        
        # Initial graph: 8 disjoint K5s (104 edges) is a strong start.
        # Or start sparser to allow more freedom?
        # Let's start with 8 K5s, it's known to be C6-free.
        edge_set = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edge_set.add((5 * k + a, 5 * k + b))
        
        adj_bits = [0] * n
        for u, v in edge_set:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
        # Degrees for heuristic
        degrees = [0] * n
        for u in range(n):
            degrees[u] = bin(adj_bits[u]).count('1')
            
        # All possible edges not in initial set
        all_edges = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set and (b, a) not in edge_set]
        
        # ADMM Parameters
        alpha = 0.5  # Learning rate for probability updates
        beta = 1.0   # Penalty weight (not explicitly used in this simplified version)
        
        # We will maintain a "probability" or "score" for each edge
        scores = {e: 0.0 for e in all_edges}
        
        # Initial scores: prefer edges between low-degree nodes
        for u, v in all_edges:
            scores[(u, v)] = -degrees[u] - degrees[v] + random.uniform(-1, 1)
            
        current_edges = len(edge_set)
        
        while time.time() < deadline:
            # Sort edges by score (highest first)
            sorted_edges = sorted(all_edges, key=lambda e: scores[e], reverse=True)
            
            # Try to add top edges
            added_this_iter = 0
            for u, v in sorted_edges:
                if time.time() > deadline:
                    break
                if adj_bits[u] & (1 << v):
                    continue
                if not has_c6_bitset(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edge_set.add((u, v))
                    added_this_iter += 1
                    # Update degrees
                    degrees[u] += 1
                    degrees[v] += 1
                    # Update scores for other edges? 
                    # In ADMM, we would update the dual variables.
                    # Here, we just re-score based on new degrees.
                    # This is a greedy step, but the "score" provides a global ordering.
                    
            # If no edges were added, we might be stuck in a local optimum.
            # ADMM would update the Lagrange multipliers.
            # Here, we can "shake" the scores to escape local optima.
            if added_this_iter == 0:
                # Perturb scores
                for e in all_edges:
                    scores[e] += random.uniform(-1, 1)
                # Or, try removing a random edge to allow different additions?
                # For now, just perturb scores.
                # Also, update base scores based on current degrees
                for u, v in all_edges:
                    if not (adj_bits[u] & (1 << v)):
                        scores[(u, v)] = -degrees[u] - degrees[v] + random.uniform(-1, 1)
                        
            # If we added edges, update scores for remaining edges
            if added_this_iter > 0:
                for u, v in all_edges:
                    if not (adj_bits[u] & (1 << v)):
                        scores[(u, v)] = -degrees[u] - degrees[v] + random.uniform(-1, 1)
            
            # Check for convergence (no more edges can be added)
            # If we go through the whole list and add nothing, we are stuck.
            # The perturbation above helps escape.
            
        # Final local search: try to add any remaining edges
        # The loop above already does this, but let's do a final pass with random order
        remaining = [e for e in all_edges if not (adj_bits[e[0]] & (1 << e[1]))]
        random.shuffle(remaining)
        for u, v in remaining:
            if time.time() > deadline:
                break
            if not has_c6_bitset(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                
        return list(edge_set)

    best_result = []
    best_count = 0
    
    # Run multiple ADMM trials
    for trial in range(20):
        if time.time() > deadline:
            break
        result = run_admm(trial)
        if len(result) > best_count:
            best_count = len(result)
            best_result = result
            if best_count >= 105:
                break
                
    # If we haven't found a good result, fall back to simple greedy
    if best_count < 100:
        # Simple greedy fallback
        edge_set = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    edge_set.add((5 * k + a, 5 * k + b))
        adj_bits = [0] * n
        for u, v in edge_set:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
        all_possible = [(a, b) for a in range(n) for b in range(a + 1, n) if (a, b) not in edge_set and (b, a) not in edge_set]
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            if not has_c6_bitset(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edge_set.add((u, v))
                
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_result = list(edge_set)

    return [[a, b] for (a, b) in sorted(best_result)]
