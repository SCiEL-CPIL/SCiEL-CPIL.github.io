"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _calc_score(adj, deg, n, edge_list):
    """
    Calculate the maximum degree sum over all triangles.
    adj: list of int bitmasks
    deg: list of ints
    n: number of vertices
    edge_list: list of (u, v) tuples
    """
    max_sum = 0
    for u, v in edge_list:
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        while common:
            lsb = common & -common
            w = lsb.bit_length() - 1
            s = deg[u] + deg[v] + deg[w]
            if s > max_sum:
                max_sum = s
            common ^= lsb
    return max_sum

def _construct_baseline():
    """K_{8,8} plus one edge inside a part."""
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append((u, v))
    edges.append((0, 1))
    return edges

def _local_search(n, initial_edges, deadline):
    """
    Local search with simulated annealing.
    Optimized for speed: minimal overhead in the loop.
    """
    start_time = time.time()
    
    # Initialize state
    adj = [0] * n
    deg = [0] * n
    edge_set = set()
    edge_list = []
    
    for u, v in initial_edges:
        u, v = min(u, v), max(u, v)
        key = (u, v)
        if key not in edge_set:
            edge_set.add(key)
            edge_list.append(key)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
    current_score = _calc_score(adj, deg, n, edge_list)
    best_score = current_score
    best_edges = list(edge_list)
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    no_improve_count = 0
    last_best_time = start_time
    
    # Precompute all possible pairs for faster non-edge selection?
    # No, random sampling is fine for N=16.
    
    while time.time() < deadline:
        # 1. Identify worst triangle (or just use random move if too slow, but we need to target)
        # For speed, we can just do random swaps and rely on SA, or target the worst triangle.
        # Finding worst triangle is O(E * common). E=65, common is small. It's fast.
        
        # Let's find the worst triangle to target removal.
        worst_tri = None
        worst_val = -1
        for u, v in edge_list:
            common = adj[u] & adj[v]
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = deg[u] + deg[v] + deg[w]
                if s > worst_val:
                    worst_val = s
                    worst_tri = (u, v, w)
                common ^= lsb
        
        # Choose edge to remove
        if worst_tri and worst_val > 0:
            u, v, w = worst_tri
            # Remove one of the edges in the worst triangle
            # Prioritize edge with highest degree sum endpoints? Or just random from tri?
            # Let's pick the edge (u,v) from the worst triangle
            rem_u, rem_v = u, v
        else:
            # No triangles, random removal
            rem_u, rem_v = random.choice(edge_list)
        
        # Choose edge to add
        # Random non-edge
        add_pair = None
        for _ in range(10):
            u = random.randrange(n)
            v = random.randrange(n)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edge_set:
                add_pair = (a, b)
                break
        
        if add_pair is None:
            # Fallback: find a non-edge by scanning
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edge_set:
                        add_pair = (i, j)
                        break
                if add_pair:
                    break
            
        if add_pair is None:
            continue # Graph is complete, shouldn't happen with 65 edges
            
        add_u, add_v = add_pair
        
        # Perform Swap
        # Remove rem_u, rem_v
        key_rem = (min(rem_u, rem_v), max(rem_u, rem_v))
        edge_set.remove(key_rem)
        # Remove from edge_list efficiently? 
        # For N=16, edge_list is small. We can just rebuild or mark.
        # Rebuilding edge_list every step is O(E). E=65. That's fine.
        # But we can also just keep edge_list and filter? No, let's just rebuild for simplicity and correctness.
        # Actually, maintaining a list with removals is tricky. 
        # Let's just keep `edge_set` as the source of truth for membership, 
        # and for iteration in _calc_score, we can iterate `edge_set`.
        # Iterating a set is fast enough for 65 elements.
        
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)
        deg[rem_u] -= 1
        deg[rem_v] -= 1
        
        # Add add_u, add_v
        key_add = (min(add_u, add_v), max(add_u, add_v))
        edge_set.add(key_add)
        adj[add_u] |= (1 << add_v)
        adj[add_v] |= (1 << add_u)
        deg[add_u] += 1
        deg[add_v] += 1
        
        # Calculate new score
        new_score = _calc_score(adj, deg, n, list(edge_set))
        delta = new_score - current_score
        
        # Accept/Reject
        accept = False
        if delta < 0:
            accept = True
        elif T > T_min and delta > 0:
            if random.random() < math.exp(-delta / T):
                accept = True
        
        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = list(edge_set)
                no_improve_count = 0
                last_best_time = time.time()
            else:
                no_improve_count += 1
        else:
            # Revert
            edge_set.remove(key_add)
            adj[add_u] &= ~(1 << add_v)
            adj[add_v] &= ~(1 << add_u)
            deg[add_u] -= 1
            deg[add_v] -= 1
            
            edge_set.add(key_rem)
            adj[rem_u] |= (1 << rem_v)
            adj[rem_v] |= (1 << rem_u)
            deg[rem_u] += 1
            deg[rem_v] += 1
            
            no_improve_count += 1
        
        T *= cooling
        
        # Restart if stuck
        if no_improve_count > 500:
            # Reset temperature
            T = 3.0
            no_improve_count = 0
            # Optionally reset to best
            # adj = [0]*n ... rebuild from best_edges
            # For now, just cool down and continue

    return best_edges

def _polish(n, initial_edges, deadline):
    """
    Final polish: Hill climbing.
    """
    adj = [0] * n
    deg = [0] * n
    edge_set = set()
    
    for u, v in initial_edges:
        u, v = min(u, v), max(u, v)
        key = (u, v)
        if key not in edge_set:
            edge_set.add(key)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
    current_score = _calc_score(adj, deg, n, list(edge_set))
    best_score = current_score
    best_edges = list(edge_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find worst triangle
        worst_tri = None
        worst_val = -1
        for u, v in edge_set:
            common = adj[u] & adj[v]
            while common:
                lsb = common & -common
                w = lsb.bit_length() - 1
                s = deg[u] + deg[v] + deg[w]
                if s > worst_val:
                    worst_val = s
                    worst_tri = (u, v, w)
                common ^= lsb
        
        if not worst_tri:
            break
            
        u, v, w = worst_tri
        
        # Try removing each edge in the worst triangle and adding a random non-edge
        # To be faster, we only try a few candidates
        candidates = [(u, v), (v, w), (w, u)]
        random.shuffle(candidates)
        
        for rem_u, rem_v in candidates:
            key_rem = (min(rem_u, rem_v), max(rem_u, rem_v))
            
            # Try a few random non-edges
            for _ in range(5):
                add_u = random.randrange(n)
                add_v = random.randrange(n)
                if add_u == add_v:
                    continue
                a, b = (add_u, add_v) if add_u < add_v else (add_v, add_u)
                if (a, b) in edge_set:
                    continue
                
                key_add = (a, b)
                
                # Swap
                edge_set.remove(key_rem)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                deg[rem_u] -= 1
                deg[rem_v] -= 1
                
                edge_set.add(key_add)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                deg[a] += 1
                deg[b] += 1
                
                new_score = _calc_score(adj, deg, n, list(edge_set))
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_edges = list(edge_set)
                    improved = True
                    break # Restart loop with new state
                else:
                    # Revert
                    edge_set.remove(key_add)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    deg[a] -= 1
                    deg[b] -= 1
                    
                    edge_set.add(key_rem)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
                    deg[rem_u] += 1
                    deg[rem_v] += 1
            
            if improved:
                break
                
    return best_edges

def construct():
    """Construct a graph on 16 vertices with 65 edges minimizing max triangle degree sum."""
    start_time = time.time()
    # Evaluator kills at 120s. Use 115s for safety.
    deadline = start_time + 115
    
    n = 16
    target_edges = 65
    
    # Start with baseline
    initial_edges = _construct_baseline()
    
    # Run local search with most of the budget
    # Reserve 5s for polish
    search_deadline = start_time + (deadline - start_time) - 5
    best_edges = _local_search(n, initial_edges, search_deadline)
    
    # Final polish phase with remaining time
    polish_deadline = deadline
    best_edges = _polish(n, best_edges, polish_deadline)
    
    # Verify edge count
    edge_set = set()
    for u, v in best_edges:
        edge_set.add((min(u, v), max(u, v)))
        
    if len(edge_set) != target_edges:
        # Fallback
        best_edges = _construct_baseline()
        edge_set = set()
        for u, v in best_edges:
            edge_set.add((min(u, v), max(u, v)))
        
    return {'n': n, 'edges': [list(e) for e in sorted(edge_set)]}

# EVOLVE-BLOCK-END
