import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 118  # safety margin for 120s limit

    n = 20
    num_pairs = n * (n - 1) // 2
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_triples = [(i, j, k) for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)]
    num_triples = len(all_triples)

    # Precompute for each triple, the list of pairs (edge indices) that form the "common neighbors"
    # Actually, for a triple (i,j,k), common neighbors are vertices v such that (i,v),(j,v),(k,v) are all edges.
    # We need to count how many such v exist. If >= 3, it's a violation.
    # To make this fast in SA, we can precompute for each edge e, which triples it participates in as a common neighbor edge?
    # That's complex. Let's stick to a reasonably fast check.
    
    # Better approach for SA: 
    # State: adjacency matrix as list of bitmasks (length 20).
    # Objective: Maximize edges - penalty * (number of violating triples).
    # Or simply: maximize edges subject to validity, but SA needs a soft objective.
    # Let's use: score = num_edges - lambda * sum(max(0, common_neighbors(t) - 2) for t in triples)
    # This way, we can have invalid states during SA, but the penalty guides us away.

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def count_violations(adj):
        """Count total excess common neighbors over 2 for all triples."""
        total_excess = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                cij = ai & adj[j]
                bc = cij.bit_count()
                if bc < 3:
                    continue
                # For each k > j, check common neighbors of i,j,k
                for k in range(j + 1, n):
                    cijk = cij & adj[k]
                    bc_ijk = cijk.bit_count()
                    if bc_ijk > 2:
                        total_excess += (bc_ijk - 2)
        return total_excess

    def objective(adj, penalty):
        e = count_edges(adj)
        v = count_violations(adj)
        return e - penalty * v

    def is_valid(adj):
        return count_violations(adj) == 0

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            v = adj[u]
            while v:
                low = v & (-v)
                vv = low.bit_length() - 1
                if vv > u:
                    edges.append([u, vv])
                v ^= low
        return edges

    # Precompute neighbor lists for each vertex for faster common neighbor calc?
    # The bitmask approach is already quite fast in Python for n=20.

    # Simulated Annealing
    random.seed(42)
    
    # Start with a random dense graph
    adj = [0] * n
    for u, v in all_pairs:
        if random.random() < 0.8:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
    # Ensure symmetry
    for u in range(n):
        for v in range(n):
            if adj[u] & (1 << v):
                adj[v] |= (1 << u)
            else:
                adj[v] &= ~(1 << u)

    current_adj = adj[:]
    current_edges = count_edges(current_adj)
    current_viol = count_violations(current_adj)
    current_score = current_edges - 100.0 * current_viol  # Start with high penalty
    
    best_adj = current_adj[:]
    best_edges = current_edges
    best_viol = current_viol
    best_score = current_score

    # SA Parameters
    T_start = 50.0
    T_end = 0.01
    alpha = 0.999  # Cooling factor per step, but we will adjust based on time
    
    # We need to determine the number of steps we can do.
    # Let's estimate time per step.
    # A step involves:
    # 1. Flip one edge (O(1) update to adj, O(1) update to edge count)
    # 2. Recalculate violations. This is O(n^3) bit operations.
    #    n=20, triples=1140. Bit ops are fast.
    #    Let's measure roughly.
    
    # To be safe, let's limit the number of steps or use time.
    # We will run SA until time runs out.
    
    # Optimization: Instead of recalculating ALL violations from scratch,
    # we can update the violation count incrementally.
    # When we flip edge (u,v):
    # It affects triples that contain u or v.
    # Specifically, for any triple (i,j,k), the common neighbors change only if u or v is in the triple OR if u/v is a common neighbor.
    # This is complex. Given n=20 is small, full recalculation might be fast enough.
    # Let's try full recalculation first. If too slow, we optimize.
    
    # Let's benchmark count_violations
    t0 = time.time()
    for _ in range(100):
        count_violations(current_adj)
    t1 = time.time()
    time_per_check = (t1 - t0) / 100.0
    
    # If time_per_check is ~0.001s, we can do ~1000 steps/s.
    # In 100s, we can do 100,000 steps. That's decent.
    
    # Let's implement the SA loop with time guard.
    # We need a dynamic penalty that decreases over time? 
    # Or fixed penalty? 
    # If penalty is too high, it gets stuck in valid but sparse states.
    # If penalty is too low, it stays invalid.
    # Let's use a penalty that starts high and decreases.
    
    T = T_start
    # We will adjust alpha to reach T_end by deadline.
    # Number of steps estimate:
    # Let's just run until time is up, cooling T exponentially.
    # T_new = T * alpha
    # We want T to reach T_end in roughly (deadline - now) seconds.
    # Steps per second ~ 1/time_per_step.
    # Total steps ~ (deadline - now) / time_per_step.
    # alpha = (T_end / T_start) ** (1 / total_steps)
    
    # Let's just use a fixed alpha and see.
    # Or better, keep T high enough to allow moves, but lower it slowly.
    
    # Let's try a different approach: 
    # Use a fixed penalty P = 10.
    # Maximize E - 10*V.
    # As E increases, V might increase, but we want to push E up.
    
    P = 10.0
    current_obj = current_edges - P * current_viol
    best_obj = current_obj
    best_adj = current_adj[:]
    
    # Track best valid graph separately
    best_valid_adj = None
    best_valid_edges = 0
    if current_viol == 0:
        best_valid_adj = current_adj[:]
        best_valid_edges = current_edges

    # Precompute list of all edges for random selection
    # We flip a random edge.
    
    while time.time() < deadline - 2:
        # Choose a random edge to flip
        u, v = random.choice(all_pairs)
        
        # Flip
        if adj[u] & (1 << v):
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            delta_e = -1
        else:
            # Add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            delta_e = 1
            
        # Calculate new violations
        # Optimization: Only recalculate violations for triples involving u or v?
        # No, let's just recalculate all for simplicity first.
        new_viol = count_violations(adj)
        new_edges = current_edges + delta_e
        new_obj = new_edges - P * new_viol
        
        delta_obj = new_obj - current_obj
        
        # Acceptance probability
        if delta_obj > 0 or random.random() < math.exp(delta_obj / T):
            # Accept
            current_edges = new_edges
            current_viol = new_viol
            current_obj = new_obj
            
            if new_viol == 0 and current_edges > best_valid_edges:
                best_valid_edges = current_edges
                best_valid_adj = adj[:]
                
            if current_obj > best_obj:
                best_obj = current_obj
                best_adj = adj[:]
            
            # Cool down
            T *= 0.9995
            if T < T_end:
                T = T_end
        else:
            # Reject: Flip back
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            # T doesn't change on rejection usually, or cools slower?
            # Standard SA cools regardless.
            T *= 0.9995
            if T < T_end:
                T = T_end

    # After SA, try to improve the best valid graph with greedy addition
    if best_valid_adj is None:
        # If we never found a valid graph, take the best_adj and try to repair it?
        # Or just return the best valid one we can find quickly.
        # Let's try to repair best_adj
        adj = best_adj[:]
        # Simple repair: remove edges from violating triples
        for _ in range(1000):
            viol = count_violations(adj)
            if viol == 0:
                break
            # Find a violating triple
            # Re-use find_violation logic
            found = False
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    cij = ai & adj[j]
                    if cij.bit_count() < 3:
                        continue
                    for k in range(j + 1, n):
                        cijk = cij & adj[k]
                        if cijk.bit_count() >= 3:
                            # Remove an edge
                            x = random.choice([b for b in range(n) if cijk & (1 << b)])
                            adj[i] &= ~(1 << x)
                            adj[x] &= ~(1 << i)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        if count_violations(adj) == 0:
            best_valid_adj = adj
            best_valid_edges = count_edges(adj)
        else:
            # Fallback: return a simple valid graph
            return [[i, i+1] for i in range(n-1)]

    # Greedy add to best_valid_adj
    adj = best_valid_adj[:]
    
    def greedy_add(adj):
        improved = True
        while improved:
            improved = False
            for u, v in all_pairs:
                if adj[u] & (1 << v):
                    continue
                old_u, old_v = adj[u], adj[v]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if count_violations(adj) == 0:
                    improved = True
                    break
                else:
                    adj[u] = old_u
                    adj[v] = old_v
    
    greedy_add(adj)
    
    # Check if improved
    final_edges = count_edges(adj)
    if final_edges > best_valid_edges:
        best_valid_adj = adj
        best_valid_edges = final_edges

    return adj_to_edges(best_valid_adj)
