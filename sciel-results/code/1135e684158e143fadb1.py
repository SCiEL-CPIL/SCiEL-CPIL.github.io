import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin under 120s limit
    
    n = 40
    
    # --- C6 Checking Logic (Fast Bitmask) ---
    def can_add_edge(adj, u, v):
        """
        Check if adding edge (u,v) to the graph represented by adj creates a C6.
        Returns True if safe, False if C6 created.
        """
        # Neighbors of u excluding v
        nu_mask = adj[u] & ~(1 << v)
        # Neighbors of v excluding u
        nv_mask = adj[v] & ~(1 << u)
        
        # We need to check if there is a path of length 3 between u and v
        # u - a - b - c - d - v
        # a in N(u), d in N(v), a != d
        # b in N(a) \ {u,v,d}, c in N(d) \ {u,v,a}
        # b ~ c
        
        a = nu_mask
        while a:
            lb_a = a & (-a)
            ai = lb_a.bit_length() - 1
            a ^= lb_a
            
            d = nv_mask & ~lb_a  # d != a
            while d:
                ld_d = d & (-d)
                di = ld_d.bit_length() - 1
                d ^= ld_d
                
                # Candidates for b: neighbors of a, excluding u, v, d, a
                b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
                
                # Candidates for c: neighbors of d, excluding u, v, a, d
                c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)
                
                # Check if any neighbor of any b in b_candidates is in c_candidates
                # Let U = union of neighbors of all b in b_candidates
                # If U & c_candidates is non-empty, then a C6 exists.
                
                union_neighbors = 0
                bm = b_candidates
                while bm:
                    lb_b = bm & (-bm)
                    bi = lb_b.bit_length() - 1
                    bm ^= lb_b
                    union_neighbors |= adj[bi]
                    
                    # Early exit if we already found a connection
                    if union_neighbors & c_candidates:
                        return False
                        
        return True

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += adj[i].bit_count()
        return count // 2

    # --- Initialization: 8 Disjoint K5s ---
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)
                
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    best_adj = None
    best_count = 0
    
    # We will perform multiple restarts.
    # Each restart consists of:
    # 1. Greedy construction with random ordering.
    # 2. Local search (simulated annealing style) to escape local optima.
    
    max_restarts = 200 # Limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5: # Leave 5s for final cleanup/return
            break
            
        current_adj = base_adj[:]
        current_edges = 80
        
        # --- Phase 1: Greedy Addition ---
        # Create a list of candidate edges (those not in base)
        candidate_edges = []
        for u, v in all_edges:
            if not ((current_adj[u] >> v) & 1):
                candidate_edges.append((u, v))
                
        random.seed(1000 + restart)
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if time.time() > deadline - 10:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges += 1
                    
        # --- Phase 2: Local Search (Edge Swaps) ---
        # We try to swap existing edges with non-existing edges to potentially
        # open up space for more additions, or simply rearrange to a better local optimum.
        
        # Maintain lists for fast random selection
        # Rebuild lists after greedy phase
        existing_edges = []
        non_existing_edges = []
        for u, v in all_edges:
            if (current_adj[u] >> v) & 1:
                existing_edges.append((u, v))
            else:
                non_existing_edges.append((u, v))
                
        # SA parameters
        T = 5.0
        cooling_rate = 0.99
        min_T = 0.5
        
        # We run local search for a fixed number of steps or until time is up
        # Steps should be small enough to allow many restarts
        max_sa_steps = 500
        
        steps = 0
        while steps < max_sa_steps and time.time() < deadline - 5:
            steps += 1
            if not existing_edges or not non_existing_edges:
                break
                
            # Pick a random existing edge to remove
            idx_rem = random.randint(0, len(existing_edges) - 1)
            u1, v1 = existing_edges[idx_rem]
            
            # Pick a random non-existing edge to add
            idx_add = random.randint(0, len(non_existing_edges) - 1)
            u2, v2 = non_existing_edges[idx_add]
            
            # Skip if same edge (shouldn't happen given lists, but safe)
            if (u1, v1) == (u2, v2):
                continue
                
            # Temporarily remove (u1, v1)
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            
            # Check if (u2, v2) can be added
            if can_add_edge(current_adj, u2, v2):
                # Add (u2, v2)
                current_adj[u2] |= (1 << v2)
                current_adj[v2] |= (1 << u2)
                
                # Update lists
                # Remove (u1, v1) from existing, add to non-existing
                # Add (u2, v2) to existing, remove from non-existing
                
                # For efficiency, we can just swap in the lists if we track indices,
                # but since lists are small (< 800), simple remove/append is okay for now.
                # However, random selection from list requires index.
                
                # Let's do a simple swap in the lists:
                # (u1, v1) moves from existing to non_existing
                # (u2, v2) moves from non_existing to existing
                
                existing_edges[idx_rem] = (u2, v2)
                non_existing_edges[idx_add] = (u1, v1)
                
                # Note: Edge count remains constant in a swap.
                # This explores the space of graphs with the same number of edges.
                
                # Occasionally, try to add an edge if possible (greedy step within SA)
                # This is a deviation from pure SA, but helps maximize edges.
                # If we find a valid add, we accept it immediately.
                # Let's try to add a random non-edge after a swap?
                # No, let's stick to pure swaps for the local search phase to keep it simple,
                # and rely on the greedy phase for adding edges.
                # However, pure swaps don't increase edge count.
                # To increase edge count, we need to find a configuration where an add is valid.
                
                # Strategy: After a few swaps, try to add an edge.
                # If we successfully add an edge, we update the best.
                
                # Let's modify: Every 10 swaps, try to add a random non-edge.
                if steps % 10 == 0:
                    if non_existing_edges:
                        idx_try = random.randint(0, len(non_existing_edges) - 1)
                        u_try, v_try = non_existing_edges[idx_try]
                        if can_add_edge(current_adj, u_try, v_try):
                            current_adj[u_try] |= (1 << v_try)
                            current_adj[v_try] |= (1 << u_try)
                            current_edges += 1
                            
                            # Update lists
                            # (u_try, v_try) moves from non_existing to existing
                            # We need a new non_existing entry? No, we just remove it from non_existing.
                            # To keep lists balanced, we can pick a random existing edge to move to non_existing?
                            # No, that would reduce edge count.
                            # We just need to update the list of non_existing edges.
                            # Since we only need to pick from non_existing, we can just remove the entry.
                            # But we need to maintain the list for future picks.
                            # Simplest: Rebuild lists if edge count changes significantly or periodically.
                            # For now, let's just mark it.
                            # Actually, maintaining dynamic lists with random access is tricky.
                            # Let's rebuild lists every 50 steps or when edge count changes.
                            
                            # Rebuild lists
                            existing_edges = []
                            non_existing_edges = []
                            for u, v in all_edges:
                                if (current_adj[u] >> v) & 1:
                                    existing_edges.append((u, v))
                                else:
                                    non_existing_edges.append((u, v))
                                    
                            # Update best if improved
                            if current_edges > best_count:
                                best_count = current_edges
                                best_adj = current_adj[:]
                                if best_count >= 105:
                                    # Early exit if we hit the target
                                    return [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
            else:
                # Reject swap, re-add (u1, v1)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                
            # Cooling
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        # After local search, check if we have a new best
        if current_edges > best_count:
            best_count = current_edges
            best_adj = current_adj[:]
            if best_count >= 105:
                return [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
        
        # If we have time, we could do a final greedy pass on the current_adj
        # But the greedy phase is already done.
        
    # Return the best graph found
    if best_adj is None:
        # Fallback to base
        best_adj = base_adj[:]
        
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])
                
    return result

# Helper for quick testing (not used in construct)
if __name__ == "__main__":
    edges = construct()
    print(f"Edges found: {len(edges)}")
