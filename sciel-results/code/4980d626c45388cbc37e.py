"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _count_new_c4s(adj, u, v):
    """
    Count the number of C4s created by adding edge (u, v).
    A C4 is u-v-w-z-u.
    We need w in N(v) and z in N(u) such that (w, z) is an edge.
    Note: u and v are not in N(u) or N(v) respectively in this context of 'new' C4s 
    involving the edge (u,v) as one side.
    Actually, the C4 is u-v-w-z-u.
    So w must be a neighbor of v (other than u), and z must be a neighbor of u (other than v).
    And (w, z) must be an edge.
    """
    nu = adj[u]
    nv = adj[v]
    
    # We iterate over neighbors of u (excluding v) and neighbors of v (excluding u)
    # Since degrees are low, we can just do a nested loop.
    count = 0
    for z in nu:
        if z == v:
            continue
        # Check if z is connected to any w in nv (excluding u)
        for w in nv:
            if w == u:
                continue
            if w in adj[z]:
                count += 1
    return count

def _remove_c4s_count(adj, u, v):
    """
    Count the number of C4s destroyed by removing edge (u, v).
    Same logic as _count_new_c4s.
    """
    return _count_new_c4s(adj, u, v)

def _construct_sa(n, min_deg_req, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Initial temperature
    T = 10.0
    cooling_rate = 0.99
    
    # Start with a random graph with average degree approx min_deg_req
    # Or start with empty graph and add edges?
    # Random graph with density p = min_deg_req / n is a good start.
    p = min_deg_req / n
    adj = [set() for _ in range(n)]
    edges = set()
    
    # Generate initial random graph
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
    
    # Calculate initial energy
    # Energy = C4_count * 100 + max(0, min_deg_req - min_degree) * 10
    # We want C4_count = 0 and min_degree >= min_deg_req
    
    def get_energy():
        c4_count = 0
        min_deg = min(len(adj[i]) for i in range(n))
        # Counting all C4s is expensive O(n^4).
        # Instead, we can track C4s incrementally?
        # For SA, we need the global C4 count to know if we are valid.
        # However, for acceptance, we can use the delta.
        # But we need to know if we reached 0.
        # Let's just count C4s periodically or use a fast approximation?
        # No, we need exact validity.
        # We can maintain a global C4 count variable.
        pass
    
    # To maintain global C4 count efficiently, we would need to update it on every move.
    # Delta C4 = - (C4s destroyed) + (C4s created).
    
    global_c4_count = 0
    # Initialize global_c4_count
    # This is O(n^4) or O(n * m) if done smartly.
    # For n=47, n^4 is ~ 5M. Might be okay for init.
    # But let's just run the SA and check validity at the end of each epoch or every K steps.
    # Actually, we can just use the delta for acceptance and check full validity every 100 steps.
    
    # Let's implement the SA loop with incremental C4 tracking.
    
    # Re-initialize global_c4_count properly
    # A C4 is a set of 4 vertices.
    # It's easier to just check validity every step if we are fast enough?
    # No, checking validity is O(n^4).
    # Let's use the delta method.
    
    # We need to know the initial C4 count.
    # We can count it by iterating all pairs of edges?
    # Or just start with 0 and assume we are far from valid, and let SA find its way?
    # No, we need to know when we hit 0.
    
    # Let's compute initial C4 count once.
    # For each pair of vertices (u, v) that are NOT connected, count common neighbors? No.
    # For each pair of connected vertices (u, v), the number of C4s involving (u,v) is 
    # number of edges between N(u)\{v} and N(v)\{u}.
    # Summing this over all edges (u,v) counts each C4 twice (once for each of its 4 edges? No, a C4 has 4 edges.
    # Each C4 u-v-w-z-u has edges (u,v), (v,w), (w,z), (z,u).
    # For edge (u,v), the C4 is counted if we look at neighbors.
    # Wait, the formula `_count_new_c4s(adj, u, v)` counts C4s that *would be created* by adding (u,v).
    # If (u,v) is already an edge, it counts the C4s that *currently exist* and use (u,v) as an edge.
    # So `sum(_count_new_c4s(adj, u, v) for (u,v) in edges)` counts each C4 4 times?
    # Let's trace: C4 0-1-2-3-0.
    # Edge (0,1): N(0)={1,3}, N(1)={0,2}. 
    # z in N(0)\{1} is 3. w in N(1)\{0} is 2. Is (3,2) edge? Yes. Count 1.
    # Edge (1,2): N(1)={0,2}, N(2)={1,3}.
    # z in N(1)\{2} is 0. w in N(2)\{1} is 3. Is (0,3) edge? Yes. Count 1.
    # Edge (2,3): Count 1.
    # Edge (3,0): Count 1.
    # Total sum = 4.
    # So `global_c4_count = sum(...) // 4`.
    
    global_c4_count = 0
    for u, v in list(edges):
        global_c4_count += _count_new_c4s(adj, u, v)
    global_c4_count //= 4
    
    min_deg = min(len(adj[i]) for i in range(n))
    current_energy = global_c4_count * 100 + max(0, min_deg_req - min_deg) * 10
    
    # SA Loop
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Choose a move: Add edge, Remove edge
        # Bias towards fixing degree deficit
        deficient_vertices = [i for i in range(n) if len(adj[i]) < min_deg_req]
        
        if deficient_vertices and random.random() < 0.7:
            # Try to add edge incident to a deficient vertex
            u = random.choice(deficient_vertices)
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            if candidates:
                v = random.choice(candidates)
                # Calculate delta
                delta_c4 = _count_new_c4s(adj, u, v)
                delta_deg = 0 # min degree might increase
                # Check if min degree increases
                old_min_deg = min_deg
                # We don't want to calculate min_deg every step O(n).
                # We can track it, but for simplicity and speed, let's just update global state and check validity periodically.
                # Actually, we can just use the energy delta.
                # If we add (u,v), degrees of u and v increase by 1.
                # min_deg can only increase if u or v was the unique minimum.
                
                delta_energy = delta_c4 * 100
                
                # Acceptance
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    # Perform move
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.add((u, v))
                    global_c4_count += delta_c4
                    
                    # Update min_deg (approximate or exact)
                    # To be safe, recalc min_deg every 100 steps?
                    # Or just track it?
                    # Let's recalc every 50 steps to keep it cheap.
                    if steps % 50 == 0:
                        min_deg = min(len(adj[i]) for i in range(n))
                        current_energy = global_c4_count * 100 + max(0, min_deg_req - min_deg) * 10
                    else:
                        # Heuristic update
                        if len(adj[u]) < min_deg:
                            pass # min_deg didn't change
                        elif len(adj[v]) < min_deg:
                            pass
                        else:
                            # If both were >= min_deg, min_deg didn't change.
                            # If one was min_deg, it increased, so min_deg might increase if it was the ONLY one.
                            # This is getting complex. Let's just recalc every step? 
                            # n=47, min() is O(47). Very fast.
                            min_deg = min(len(adj[i]) for i in range(n))
                        # Update energy
                        current_energy = global_c4_count * 100 + max(0, min_deg_req - min_deg) * 10
                        
                        # Check success
                        if global_c4_count == 0 and min_deg >= min_deg_req:
                            edge_list = []
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        edge_list.append([u, v])
                            return {'n': n, 'edges': edge_list}
        else:
            # Remove a random edge or add a random edge
            if edges and random.random() < 0.5:
                # Remove
                u, v = random.choice(list(edges))
                delta_c4 = _count_new_c4s(adj, u, v) # This is the number of C4s destroyed
                # Removing edge reduces C4 count by delta_c4
                # Degrees decrease by 1
                delta_energy = -delta_c4 * 100
                # Degree penalty might increase
                # Let's just accept with probability based on C4 change, and check validity later.
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    adj[u].remove(v)
                    adj[v].remove(u)
                    edges.remove((u, v))
                    global_c4_count -= delta_c4
                    
                    min_deg = min(len(adj[i]) for i in range(n))
                    current_energy = global_c4_count * 100 + max(0, min_deg_req - min_deg) * 10
            else:
                # Add random edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and v not in adj[u]:
                    delta_c4 = _count_new_c4s(adj, u, v)
                    delta_energy = delta_c4 * 100
                    
                    if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.add((u, v))
                        global_c4_count += delta_c4
                        
                        min_deg = min(len(adj[i]) for i in range(n))
                        current_energy = global_c4_count * 100 + max(0, min_deg_req - min_deg) * 10
                        
                        if global_c4_count == 0 and min_deg >= min_deg_req:
                            edge_list = []
                            for u in range(n):
                                for v in adj[u]:
                                    if u < v:
                                        edge_list.append([u, v])
                            return {'n': n, 'edges': edge_list}
                            
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1 # Floor temperature to keep exploring
            
        # Periodic full check to ensure global_c4_count is correct (in case of bugs)
        if steps % 1000 == 0:
             # Recount
             recount = 0
             for u, v in list(edges):
                 recount += _count_new_c4s(adj, u, v)
             recount //= 4
             if recount != global_c4_count:
                 global_c4_count = recount
                 min_deg = min(len(adj[i]) for i in range(n))
                 current_energy = global_c4_count * 100 + max(0, min_deg_req - min_deg) * 10
                 if global_c4_count == 0 and min_deg >= min_deg_req:
                     edge_list = []
                     for u in range(n):
                         for v in adj[u]:
                             if u < v:
                                 edge_list.append([u, v])
                     return {'n': n, 'edges': edge_list}
                     
    return None

def construct():
    start_time = time.time()
    deadline = 115.0
    
    time_budgets = {
        47: 30.0,
        46: 40.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
            
        result = _construct_sa(n, min_deg_req, t_limit)
        if result is not None:
            return result
            
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
