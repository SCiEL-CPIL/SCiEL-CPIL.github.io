import time
import random

def construct():
    n = 20
    deadline = time.time() + 110
    
    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def get_edges_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def count_violations(adj):
        v_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] & adj[v]).bit_count() > 2:
                    v_count += 1
        return v_count

    def greedy_init():
        adj = [0] * n
        random.shuffle(all_edges)
        for u, v in all_edges:
            # Check if adding (u,v) is valid
            # 1. Check (u,v) pair itself? No, common neighbors don't include u,v.
            # 2. Check pairs (u, w) and (v, w) for all w != u, v
            valid = True
            # Check if u and v have > 2 common neighbors
            if (adj[u] & adj[v]).bit_count() > 2:
                valid = False
            else:
                # Check neighbors of u
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    w = lsb.bit_length() - 1
                    if w != v:
                        # Adding (u,v) makes v a neighbor of u.
                        # Does this create a new common neighbor for (u, w)?
                        # Common neighbors of u and w are N(u) intersect N(w).
                        # New N(u) = Old N(u) U {v}.
                        # So new common neighbors = Old common neighbors U ({v} intersect N(w)).
                        # So if v is in N(w), the count increases by 1.
                        if adj[w] & (1 << v):
                            if (adj[u] & adj[w]).bit_count() > 1: # If it was 2, it becomes 3
                                valid = False
                                break
                    mask -= lsb
                if not valid: break
                
                # Check neighbors of v
                if valid:
                    mask = adj[v]
                    while mask:
                        lsb = mask & -mask
                        w = lsb.bit_length() - 1
                        if w != u:
                            if adj[w] & (1 << u):
                                if (adj[v] & adj[w]).bit_count() > 1:
                                    valid = False
                                    break
                        mask -= lsb
            
            if valid:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def simulated_annealing(adj, time_limit):
        # State
        current_adj = adj[:]
        current_edges = get_edges_count(current_adj)
        current_violations = count_violations(current_adj)
        
        best_adj = current_adj[:]
        best_score = current_edges - 100 * current_violations
        
        # Precompute affected pairs for each edge
        # affected_pairs[u][v] = list of pairs (a,b) that need checking if (u,v) is toggled
        # Pairs are (u, w) and (v, w) for w != u, v
        # And the pair (u, v) itself? 
        # Toggling (u,v) does NOT change N(u) intersect N(v).
        # So we don't need to check (u,v) for violation change?
        # Wait. The violation is defined on the pair (u,v).
        # The violation count for pair (u,v) is 1 if |N(u) intersect N(v)| > 2.
        # As established, N(u) intersect N(v) does not change when toggling (u,v).
        # So the violation status of pair (u,v) is constant during the toggle.
        # We only need to update violations for pairs (u, w) and (v, w).
        
        affected = {}
        for u, v in all_edges:
            pairs = []
            for w in range(n):
                if w != u and w != v:
                    pairs.append((u, w))
                    pairs.append((v, w))
            # Remove duplicates? (u,w) and (v,w) are distinct if u!=v.
            # But if we iterate all edges, we might double count?
            # No, we just need the list of pairs to check.
            affected[(u, v)] = pairs

        T = 10.0
        cooling = 0.9999 # Adjust based on steps
        steps_per_sec = 50000 # Estimate
        max_steps = int(time_limit * steps_per_sec)
        
        for i in range(max_steps):
            if time.time() > time_limit:
                break
            
            # Pick random edge
            u, v = random.choice(all_edges)
            
            # Calculate change
            # Toggle (u,v)
            # Check validity of new state for affected pairs
            
            # We need to know the OLD violation status of affected pairs
            # And calculate the NEW violation status
            
            delta_edges = 1 if not (current_adj[u] & (1 << v)) else -1
            
            # Temporarily apply change to calculate new violations
            # To avoid modifying current_adj, we can simulate
            # But it's faster to modify, calculate, and revert if rejected
            
            # Save old state of affected pairs
            old_violations = {}
            for a, b in affected[(u, v)]:
                if a > b: a, b = b, a
                # Ensure consistent ordering for lookup? 
                # adj is symmetric, so adj[a]&adj[b] is same as adj[b]&adj[a]
                old_violations[(a,b)] = 1 if (current_adj[a] & current_adj[b]).bit_count() > 2 else 0
            
            # Apply toggle
            if current_adj[u] & (1 << v):
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            
            new_violations_count = 0
            for a, b in affected[(u, v)]:
                if a > b: a, b = b, a
                new_violations_count += 1 if (current_adj[a] & current_adj[b]).bit_count() > 2 else 0
            
            # We need to update total_violations
            # We stored old_violations for these specific pairs.
            # But we need the sum of old violations for these pairs.
            old_violations_sum = sum(old_violations.values())
            delta_violations = new_violations_count - old_violations_sum
            
            new_edges = current_edges + delta_edges
            new_violations = current_violations + delta_violations
            
            new_score = new_edges - 100 * new_violations
            old_score = current_edges - 100 * current_violations
            
            delta_score = new_score - old_score
            
            accept = False
            if delta_score >= 0:
                accept = True
            else:
                if random.random() < pow(2.71828, delta_score / T):
                    accept = True
            
            if accept:
                current_edges = new_edges
                current_violations = new_violations
                # Update best
                if new_score > best_score:
                    best_score = new_score
                    best_adj = current_adj[:]
            else:
                # Revert
                if current_adj[u] & (1 << v):
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
            
            T *= cooling
            
        return best_adj

    # Main loop
    best_adj = None
    best_edges = 0
    
    # Run multiple SA instances
    while time.time() < deadline:
        # Initialize
        adj = greedy_init()
        
        # Run SA for a fixed time chunk or until convergence?
        # Let's run for 5 seconds per instance
        sa_time = min(5.0, deadline - time.time())
        if sa_time <= 0: break
        
        res_adj = simulated_annealing(adj, time.time() + sa_time)
        
        edges = get_edges_count(res_adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = res_adj[:]
            
    # Convert best_adj to list of edges
    if best_adj is None:
        best_adj = greedy_init()
        
    best_edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                best_edges_list.append([u, v])
                
    return best_edges_list
