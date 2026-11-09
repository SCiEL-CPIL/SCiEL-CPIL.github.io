import time
import numpy as np
import itertools
import random

def paley_graph(q):
    """Construct the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return q, edges

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin for 330s limit
    
    # Fallback: Paley graph on 37 vertices
    n_seed, edges_seed = paley_graph(37)
    best_n = n_seed
    best_edges = edges_seed
    
    # Initialize with Paley 37
    n_current = 37
    # adj_mask[i] is a bitmask of neighbors of i
    adj_mask = [0] * n_current
    for u, v in edges_seed:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Helper to get neighbors and non-neighbors
    def get_neighbors(i, n):
        return [j for j in range(n) if i != j and (adj_mask[i] >> j) & 1]
        
    def get_non_neighbors(i, n):
        return [j for j in range(n) if i != j and not ((adj_mask[i] >> j) & 1)]

    # Pre-compute K4s and Independent K4s for the current graph
    # This is expensive to recompute, but we can update incrementally.
    # For n=37, number of 4-sets is C(37,4) ~ 66k.
    # Let's just recompute K4s and I4s for the core when adding a vertex?
    # Or better, maintain lists of K4s and I4s.
    
    k4s = []
    i4s = []
    
    def rebuild_k4_i4(n):
        nonlocal k4s, i4s
        k4s = []
        i4s = []
        # Iterate over all 4-sets
        # This is slow in Python.
        # C(37,4) = 66045.
        # C(42,4) = 111930.
        # It's manageable if we do it once per vertex addition.
        
        # Optimization: Use bitmasks for K4 check?
        # A set {a,b,c,d} is a K4 if adj[a] has b,c,d and adj[b] has a,c,d etc.
        # We can just check edges.
        
        for combo in itertools.combinations(range(n), 4):
            a, b, c, d = combo
            is_k4 = True
            is_i4 = True
            # Check edges
            if not ((adj_mask[a] >> b) & 1): is_k4 = False
            if not ((adj_mask[a] >> c) & 1): is_k4 = False
            if not ((adj_mask[a] >> d) & 1): is_k4 = False
            if not ((adj_mask[b] >> c) & 1): is_k4 = False
            if not ((adj_mask[b] >> d) & 1): is_k4 = False
            if not ((adj_mask[c] >> d) & 1): is_k4 = False
            
            if ((adj_mask[a] >> b) & 1): is_i4 = False
            if ((adj_mask[a] >> c) & 1): is_i4 = False
            if ((adj_mask[a] >> d) & 1): is_i4 = False
            if ((adj_mask[b] >> c) & 1): is_i4 = False
            if ((adj_mask[b] >> d) & 1): is_i4 = False
            if ((adj_mask[c] >> d) & 1): is_i4 = False
            
            if is_k4:
                k4s.append(combo)
            if is_i4:
                i4s.append(combo)
                
    # Initial build for n=37
    rebuild_k4_i4(n_current)

    # Function to try to add a new vertex
    def try_add_vertex(n_target, time_limit):
        nonlocal adj_mask, k4s, i4s, n_current
        
        if time.time() > time_limit:
            return False
            
        # We want to add vertex n_current (index)
        # We need to find a bitmask S (subset of 0..n_current-1) to connect to.
        # Constraints:
        # 1. S does not contain any K4 from k4s.
        # 2. (All \ S) does not contain any I4 from i4s.
        
        # Local Search on the bitmask S
        # Represent S as an integer or list of bools.
        # Since n_current <= 42, we can use an integer bitmask.
        
        # Initialize S randomly or based on some heuristic
        # Let's try random initialization
        S_mask = 0
        for i in range(n_current):
            if random.random() < 0.5:
                S_mask |= (1 << i)
                
        best_S_mask = S_mask
        best_cost = float('inf')
        
        T = 1.0
        min_T = 0.01
        cooling = 0.999
        
        # Cost function: number of K4s fully contained in S + number of I4s fully contained in ~S
        def calculate_cost(mask):
            cost = 0
            # Check K4s
            for k4 in k4s:
                # Check if all 4 vertices are in mask
                if all((mask >> v) & 1 for v in k4):
                    cost += 1
            # Check I4s
            for i4 in i4s:
                # Check if all 4 vertices are NOT in mask
                if all(not ((mask >> v) & 1) for v in i4):
                    cost += 1
            return cost
            
        current_cost = calculate_cost(S_mask)
        best_cost = current_cost
        
        stagnation = 0
        max_stagnation = 2000
        
        while time.time() < time_limit:
            # Pick a random vertex in the existing graph to flip connection
            v = random.randint(0, n_current - 1)
            
            # Flip bit v in S_mask
            S_mask ^= (1 << v)
            
            new_cost = calculate_cost(S_mask)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_S_mask = S_mask
                    stagnation = 0
                    if best_cost == 0:
                        break
                else:
                    stagnation += 1
            else:
                # Revert
                S_mask ^= (1 << v)
                stagnation += 1
                
            T = max(min_T, T * cooling)
            
            if stagnation > max_stagnation:
                # Restart
                S_mask = 0
                for i in range(n_current):
                    if random.random() < 0.5:
                        S_mask |= (1 << i)
                current_cost = calculate_cost(S_mask)
                T = 1.0
                stagnation = 0
                
        if best_cost == 0:
            # Accept the new vertex
            new_v = n_current
            adj_mask.append(0)
            for i in range(n_current):
                if (best_S_mask >> i) & 1:
                    adj_mask[new_v] |= (1 << i)
                    adj_mask[i] |= (1 << new_v)
            
            n_current += 1
            # Update k4s and i4s
            # New K4s involving new_v: {new_v} + K3 in S
            # New I4s involving new_v: {new_v} + I3 in ~S
            # We need to find K3s in S and I3s in ~S from the OLD graph's structure?
            # Actually, we can just rebuild k4s and i4s for the new n_current.
            # It's O(n^4) which is ~100k operations. Fast enough.
            rebuild_k4_i4(n_current)
            
            # Update best result
            if n_current > best_n:
                best_n = n_current
                # Convert to edges
                edges = []
                for i in range(n_current):
                    for j in range(i + 1, n_current):
                        if (adj_mask[i] >> j) & 1:
                            edges.append((i, j))
                best_edges = edges
                
            return True
        else:
            return False

    # Try to extend from 37 to 42
    # We have time.
    for target_n in range(38, 43):
        if time.time() > deadline - 10:
            break
            
        # Allocate some time for this step
        # If we fail, we might want to try again with different params?
        # For now, just try once per target_n with a time budget.
        time_budget = min(50, deadline - time.time() - 5)
        
        # We might need multiple attempts if the first one fails
        success = False
        attempts = 0
        while time.time() < deadline - 5 and attempts < 5:
            attempts += 1
            # Reset random seed for reproducibility in attempt? No, let it be random.
            if try_add_vertex(target_n, time.time() + time_budget / attempts):
                success = True
                break
            
        if not success:
            # If we can't add this vertex, we can't add larger ones
            # Unless we restart from a different seed?
            # The directive says "improve", so if we stuck at 37, score is 1.0.
            # If we get to 38, score is 38/37 > 1.0.
            break

    return best_n, best_edges
