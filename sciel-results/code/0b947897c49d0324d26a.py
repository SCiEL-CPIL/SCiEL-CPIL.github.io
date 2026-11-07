"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # 1. Fast Violation Counter
    # Violation: path s-u-v-t where deg(s) == deg(t).
    # We maintain:
    # deg[u]: current degree of u
    # adj[u]: set of neighbors of u
    # For fast checking of adding edge (u,v):
    # We need to count pairs (s, t) such that s in N(u)\{v}, t in N(v)\{u}, and deg(s) == deg(t).
    # Let's precompute for each node u, a histogram of degrees of its neighbors.
    # neigh_deg_counts[u][d] = number of neighbors w of u such that deg(w) == d
    
    def count_violations_for_edge(u, v, deg, adj, neigh_deg_counts):
        """
        Returns the number of violations created by adding edge (u, v).
        Assumes (u, v) is NOT currently in the graph.
        """
        # Violations are paths s-u-v-t.
        # s must be a neighbor of u (not v).
        # t must be a neighbor of v (not u).
        # We need deg(s) == deg(t).
        
        # Get degrees of neighbors of u (excluding v)
        # We can iterate over neighbors of u.
        # To optimize, we can use the precomputed counts if we update them carefully,
        # but since we are just checking a hypothetical addition, we can iterate.
        # However, iterating over neighbors for every candidate edge in greedy might be slow.
        # n=15, max degree 14. 105 edges. 105 * 14 * 14 is small enough?
        # 105 * 196 = ~20k ops per full pass. Very fast.
        
        viol = 0
        # Iterate over neighbors of u
        for s in adj[u]:
            if s == v:
                continue
            ds = deg[s]
            # Now check neighbors of v
            # We need to count how many t in N(v)\{u} have deg(t) == ds
            # We can just iterate N(v)
            for t in adj[v]:
                if t == u:
                    continue
                if deg[t] == ds:
                    viol += 1
        return viol

    def is_valid_graph(edges, deg, adj):
        """
        Check if the graph defined by edges, deg, adj is valid.
        Violation: any path s-u-v-t with deg(s)==deg(t).
        """
        total_viol = 0
        for u, v in edges:
            # Check paths through edge (u,v)
            # s in N(u)\{v}, t in N(v)\{u}, deg(s)==deg(t)
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u:
                        continue
                    if deg[t] == ds:
                        total_viol += 1
        return total_viol == 0

    # 2. Greedy Construction with Dynamic Costing
    # We want to add edges one by one.
    # Cost of adding edge (u,v) = number of violations it creates.
    # We prefer edges with cost 0. If multiple, maybe prefer those that don't increase "potential" conflicts?
    # Or just pick any valid edge (cost 0).
    # If no cost 0 edge, we stop? Or pick min cost?
    # If we pick min cost > 0, the graph becomes invalid.
    # So we should only add edges with cost 0.
    
    def greedy_construct():
        deg = [0] * n
        adj = [set() for _ in range(n)]
        current_edges = []
        remaining_edges = set(all_edges)
        
        # To make it deterministic but varied, we can shuffle the remaining edges or use a priority queue.
        # Let's use a simple loop: in each iteration, find an edge with cost 0.
        # To avoid O(M^2 * K) complexity, we can just scan. M=105.
        
        while remaining_edges:
            valid_edge_found = False
            # Shuffle remaining edges to break symmetry and explore different structures
            list_remaining = list(remaining_edges)
            random.shuffle(list_remaining)
            
            for e in list_remaining:
                u, v = e
                # Check if adding (u,v) creates any violation
                # We need to check against current graph.
                viol = count_violations_for_edge(u, v, deg, adj, None)
                if viol == 0:
                    # Add it
                    current_edges.append(e)
                    deg[u] += 1
                    deg[v] += 1
                    adj[u].add(v)
                    adj[v].add(u)
                    remaining_edges.remove(e)
                    valid_edge_found = True
                    break
            
            if not valid_edge_found:
                break
                
        return current_edges, deg, adj

    # 3. Local Search: 2-Swap
    # If greedy gets stuck, we can try to swap edges.
    # Remove e1, add e2.
    # This changes degrees and adjacency.
    # We need to check validity of the new graph.
    
    def try_swap(current_edges, deg, adj, all_edges_set):
        """
        Try to improve the graph by removing one edge and adding another.
        Returns (new_edges, new_deg, new_adj) if successful, else None.
        """
        if not current_edges:
            return None
        
        # Pick a random edge to remove
        e1_idx = random.randint(0, len(current_edges) - 1)
        e1 = current_edges[e1_idx]
        u1, v1 = e1
        
        # Candidate edges to add
        # We should prefer edges that are "cheap" to add after removal.
        # But calculating cost after removal is complex.
        # Let's just try random missing edges.
        missing = [e for e in all_edges if e not in all_edges_set]
        if not missing:
            return None
        
        # Shuffle missing to try different ones
        random.shuffle(missing)
        
        # Limit number of attempts to keep it fast
        max_attempts = 20
        for i in range(max_attempts):
            e2 = missing[i]
            u2, v2 = e2
            
            # Check if e2 is already present (should be in missing, so no)
            # Construct new graph state
            # Remove e1
            new_edges = current_edges[:]
            new_edges.pop(e1_idx)
            
            # Check if e2 is in new_edges? No, e2 was missing.
            
            # Add e2
            new_edges.append(e2)
            
            # Update degrees and adj
            new_deg = deg[:]
            new_adj = [set(s) for s in adj]
            
            # Remove u1-v1
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            new_adj[u1].remove(v1)
            new_adj[v1].remove(u1)
            
            # Add u2-v2
            new_deg[u2] += 1
            new_deg[v2] += 1
            new_adj[u2].add(v2)
            new_adj[v2].add(u2)
            
            # Check validity
            if is_valid_graph(new_edges, new_deg, new_adj):
                return new_edges, new_deg, new_adj
                
        return None

    # 4. Main Loop
    best_edges = []
    best_deg = [0] * n
    best_adj = [set() for _ in range(n)]
    
    # Multiple restarts to find a good structure
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        edges, deg, adj = greedy_construct()
        
        if len(edges) > len(best_edges):
            best_edges = edges
            best_deg = deg
            best_adj = adj
            
        # Local Search on the current solution
        current_edges = edges
        current_deg = deg
        current_adj = adj
        
        # Try to improve with swaps
        # We can run a few iterations of swaps
        swap_attempts = 100
        for s in range(swap_attempts):
            if time.time() > deadline:
                break
                
            current_edges_set = set(current_edges)
            res = try_swap(current_edges, current_deg, current_adj, current_edges_set)
            if res:
                new_edges, new_deg, new_adj = res
                # Only accept if it improves or is equal? 
                # Since we only add valid edges, size stays same.
                # Wait, swap keeps size same.
                # So we don't improve size.
                # But it might enable further additions?
                # The greedy construction stopped because NO edge could be added.
                # A swap changes the structure, so maybe an edge that couldn't be added before can be added now.
                # So we should accept swaps even if size doesn't change, to explore.
                current_edges = new_edges
                current_deg = new_deg
                current_adj = new_adj
                
                # After a swap, try to add more edges (Greedy again)
                # This is a "Greedy + Swap" hybrid
                improved = True
                while improved:
                    improved = False
                    if time.time() > deadline:
                        break
                    current_edges_set = set(current_edges)
                    remaining = [e for e in all_edges if e not in current_edges_set]
                    random.shuffle(remaining)
                    for e in remaining:
                        u, v = e
                        viol = count_violations_for_edge(u, v, current_deg, current_adj, None)
                        if viol == 0:
                            # Add it
                            current_edges.append(e)
                            current_deg[u] += 1
                            current_deg[v] += 1
                            current_adj[u].add(v)
                            current_adj[v].add(u)
                            improved = True
                            break
                            
                    if len(current_edges) > len(best_edges):
                        best_edges = current_edges
                        best_deg = current_deg
                        best_adj = current_adj
                        
    # Final Polish: Try to add any remaining edge that keeps it valid
    # This is already done in the loop above, but let's do a final pass on best_edges
    if best_edges:
        best_edges_set = set(best_edges)
        # Reconstruct deg/adj for best_edges if they got out of sync (they shouldn't)
        # But let's be safe
        final_deg = [0] * n
        final_adj = [set() for _ in range(n)]
        for u, v in best_edges:
            final_deg[u] += 1
            final_deg[v] += 1
            final_adj[u].add(v)
            final_adj[v].add(u)
            
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            remaining = [e for e in all_edges if e not in best_edges_set]
            random.shuffle(remaining)
            for e in remaining:
                u, v = e
                viol = count_violations_for_edge(u, v, final_deg, final_adj, None)
                if viol == 0:
                    best_edges.append(e)
                    best_edges_set.add(e)
                    final_deg[u] += 1
                    final_deg[v] += 1
                    final_adj[u].add(v)
                    final_adj[v].add(u)
                    improved = True
                    break

    if not best_edges:
        # Fallback: Star graph is always valid?
        # Star: center 0, leaves 1..14.
        # Degrees: deg(0)=14, deg(i)=1 for i>0.
        # Path s-u-v-t.
        # If u=0, v=i. s must be neighbor of 0 (leaf), t neighbor of i (only 0).
        # s=leaf, t=0.
        # Path: leaf - 0 - i - 0? No, simple path.
        # Path s-u-v-t with distinct vertices.
        # In a star, max path length is 2 (leaf-center-leaf).
        # So no 3-edge paths exist.
        # Thus star is valid.
        best_edges = [(0, i) for i in range(1, n)]

    # Ensure types are correct
    return [list(e) for e in best_edges]

# EVOLVE-BLOCK-END
