import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s kill
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def count_edges():
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def is_valid_add(u, v):
        # Check if adding (u, v) maintains K2,3-free property
        # 1. u and v must not already be connected
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Direct constraint: |N(u) cap N(v)| <= 2
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 3. Indirect constraints
        # For each w in N(u) (w != v), adding v to N(u) means u becomes a neighbor of v.
        # The new common neighbors of (w, v) are (N(w) cap N(v)) U {u}.
        # We need |(N(w) cap N(v)) U {u}| <= 2.
        
        # Iterate neighbors of u
        mask_u = adj[u]
        # We don't need to exclude v from mask_u for the loop logic if we handle it, 
        # but v is not in adj[u] yet.
        temp = mask_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # w is a neighbor of u
            # Check constraint for pair (w, v)
            # Common neighbors of w and v in current graph: adj[w] & adj[v]
            # After adding (u, v), u is in N(v).
            # Is u in N(w)? Yes, because w in N(u) implies u in N(w).
            # So u is a common neighbor of w and v.
            # The set of common neighbors becomes (adj[w] & adj[v]) | (1 << u)
            # But wait, adj[w] & adj[v] already contains u if u is in both.
            # Since w in N(u) and we are adding u to N(v), u will be in N(w) and N(v).
            # So the new intersection is (adj[w] & adj[v]) | (1 << u).
            # We need this count to be <= 2.
            
            new_common_wv = (adj[w] & adj[v]) | (1 << u)
            if new_common_wv.bit_count() > 2:
                return False
            temp ^= lsb
            
        # For each w in N(v) (w != u), adding u to N(v) means v becomes a neighbor of u.
        # The new common neighbors of (u, w) are (N(u) cap N(w)) U {v}.
        mask_v = adj[v]
        temp = mask_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # w is a neighbor of v
            # Check constraint for pair (u, w)
            # Common neighbors of u and w in current graph: adj[u] & adj[w]
            # After adding (u, v), v is in N(u).
            # Is v in N(w)? Yes, because w in N(v) implies v in N(w).
            # So v is a common neighbor of u and w.
            # The set of common neighbors becomes (adj[u] & adj[w]) | (1 << v).
            
            new_common_uw = (adj[u] & adj[w]) | (1 << v)
            if new_common_uw.bit_count() > 2:
                return False
            temp ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    best_count = 0
    best_adj = [0] * n
    
    # Multi-start Simulated Annealing
    # We will run several restarts.
    # Each restart:
    # 1. Greedy random initialization
    # 2. SA optimization
    
    num_restarts = 10  # Adjust based on time per restart
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Reset adjacency
        adj = [0] * n
        current_count = 0
        
        # 1. Greedy Initialization
        # Shuffle edges
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if is_valid_add(u, v):
                add_edge(u, v)
                current_count += 1
                
        # Track present and missing edges for SA
        present_edges = []
        missing_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    present_edges.append((i, j))
                else:
                    missing_edges.append((i, j))
                    
        # 2. Simulated Annealing
        T = 1.0
        alpha = 0.99995
        min_T = 0.01
        
        # SA Loop
        while time.time() < deadline - 5:
            # Pick a move
            # 60% Add, 20% Remove, 20% Swap
            move_type = random.random()
            
            if move_type < 0.6:
                # Try to add a random missing edge
                if not missing_edges:
                    continue
                idx = random.randint(0, len(missing_edges) - 1)
                u, v = missing_edges[idx]
                
                if is_valid_add(u, v):
                    # Valid add: Accept always (improvement)
                    add_edge(u, v)
                    current_count += 1
                    missing_edges.pop(idx)
                    present_edges.append((u, v))
                    
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj[:]
                else:
                    # Invalid: Do nothing
                    pass
                    
            elif move_type < 0.8:
                # Try to remove a random present edge
                if not present_edges:
                    continue
                idx = random.randint(0, len(present_edges) - 1)
                u, v = present_edges[idx]
                
                # Accept with probability exp(-1/T)
                # Delta = -1 (worsening)
                # P = exp(-Delta/T) = exp(1/T) ... Wait.
                # Standard SA: Minimize Energy. Here Maximize Edges.
                # Let Energy = -Edges.
                # Move: Remove edge. Edges -> Edges - 1. Energy -> Energy + 1.
                # Delta_E = +1.
                # P = exp(-Delta_E / T) = exp(-1/T).
                
                if T > 0.01:
                    prob = math.exp(-1.0 / T)
                else:
                    prob = 0.0
                    
                if random.random() < prob:
                    remove_edge(u, v)
                    current_count -= 1
                    present_edges.pop(idx)
                    missing_edges.append((u, v))
                    
            else:
                # Swap: Remove one present, try to add one missing
                if not present_edges or not missing_edges:
                    continue
                
                idx_p = random.randint(0, len(present_edges) - 1)
                u, v = present_edges[idx_p]
                
                idx_m = random.randint(0, len(missing_edges) - 1)
                x, y = missing_edges[idx_m]
                
                # Temporarily remove (u, v)
                remove_edge(u, v)
                
                # Try to add (x, y)
                if is_valid_add(x, y):
                    # Valid swap: Count remains same.
                    # Accept with probability 1.0 to allow exploration?
                    # Or accept with prob exp(0/T) = 1.0.
                    add_edge(x, y)
                    
                    # Update lists
                    present_edges.pop(idx_p)
                    present_edges.append((x, y))
                    missing_edges.pop(idx_m)
                    missing_edges.append((u, v))
                    
                    # Even if count is same, structure changes.
                    # We don't update best_count here as it's not an improvement.
                    # But we might escape a local optimum for future adds.
                else:
                    # Invalid swap: Revert
                    add_edge(u, v)
                    
            # Cooling
            T *= alpha
            if T < min_T:
                T = min_T

    # Return best result
    res = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                res.append([i, j])
                
    return res

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    # Validate
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    valid = True
    for i in range(20):
        for j in range(i + 1, 20):
            if (adj[i] & adj[j]).bit_count() > 2:
                valid = False
                print(f"Violation: {i}, {j} have {(adj[i] & adj[j]).bit_count()} common neighbors")
    if valid:
        print("Valid")
    else:
        print("Invalid")
