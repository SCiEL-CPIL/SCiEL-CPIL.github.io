import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill

    # Strategy:
    # 1. Structural insight: If we designate vertex 0 as a "hub" with high degree,
    #    and all other vertices as "leaves" with low degree, then any path s-u-v-t
    #    with equal degree endpoints must have s, t both in the leaf set (since hub
    #    degree is unique/maximal, it can't match a leaf).
    #    This reduces the problem to finding a dense graph on vertices 1..14 (14 vertices)
    #    such that no path of length 3 in the FULL graph has endpoints in 1..14 with equal degree.
    #    
    #    Let's analyze the constraints on the leaf subgraph (vertices 1..14).
    #    Let L be the set of leaves.
    #    Degrees in full graph: deg_full(v) = deg_L(v) + (1 if v connected to 0 else 0).
    #    If we connect 0 to ALL leaves, deg_full(v) = deg_L(v) + 1 for all v in L.
    #    Then deg_full(s) == deg_full(t) iff deg_L(s) == deg_L(t).
    #    So we need a graph on 14 vertices such that no path s-u-v-t has deg_L(s) == deg_L(t).
    #    
    #    This is a known type of problem. A very dense graph satisfying this is a "Windmill" or similar?
    #    Actually, if the leaf graph is a star (center 1, leaves 2..14), deg_L(1)=13, deg_L(others)=1.
    #    Paths:
    #    - s, t leaves (deg 1). u, v must be in L.
    #    - If u=1, v=leaf: path s-1-v-t. deg(s)=1, deg(t)=1. Conflict!
    #    So a star of leaves is NOT valid.
    #
    #    Let's try a different structure for the leaf graph.
    #    What if the leaf graph is a complete graph K14?
    #    deg_L(v) = 13 for all v. All degrees equal. Any path s-u-v-t has deg(s)=deg(t). Invalid.
    #
    #    We need distinct degrees or specific structures.
    #    Consider a "near-complete" graph with one vertex removed?
    #    
    #    Alternative: Use the current program's best found structure but verify and optimize.
    #    The current program found a score of 2.0 (which is 2.0 * seed_value). 
    #    Wait, the score is value/seed. If seed is 14 (star), score 2.0 means 28 edges.
    #    Can we get more than 28 edges?
    #    Max edges in simple graph on 15 vertices is 105.
    #    
    #    Let's use a fast local search on the bitmask representation, but with a much faster validity check.
    #    The current is_valid_adj is O(E * D^2) roughly.
    #    We can precompute degree classes.
    
    def compute_degrees(adj):
        return [bin(adj[i]).count('1') for i in range(n)]

    def is_valid_fast(adj, degrees):
        # Check: for all paths s-u-v-t, deg(s) != deg(t).
        # Equivalent: For each edge (u,v), and each s in N(u)\{v}, and each t in N(v)\{u}, deg(s) != deg(t).
        # We can group neighbors by degree.
        
        # Precompute for each vertex, the set of degrees of its neighbors?
        # No, just iterate.
        
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                
                # Get neighbors of u excluding v
                nu = au & ~(1 << v)
                # Get neighbors of v excluding u
                nv = adj[v] & ~(1 << u)
                
                # We need to check if there exists s in nu and t in nv such that deg[s] == deg[t].
                # Create a bitmask of degrees present in nu?
                # Degrees are small (0..14).
                
                deg_mask_u = 0
                while nu:
                    bit = nu & -nu
                    s = bit.bit_length() - 1
                    deg_mask_u |= (1 << degrees[s])
                    nu ^= bit
                
                # Now check if any t in nv has a degree in deg_mask_u
                while nv:
                    bit = nv & -nv
                    t = bit.bit_length() - 1
                    if deg_mask_u & (1 << degrees[t]):
                        return False
                    nv ^= bit
                    
        return True

    def count_edges(adj):
        total = 0
        for i in range(n):
            total += bin(adj[i]).count('1')
        return total // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Start with a known good structure?
    # The current program's best was 28 edges.
    # Let's try to start with a random dense graph and prune, or build up.
    # Building up is safer for validity.
    
    random.seed(42)
    best_adj = [0] * n
    best_deg = [0] * n
    best_count = 0
    
    # Greedy construction with multiple random orderings
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for trial in range(100):
        if time.time() > deadline - 5:
            break
            
        trial_adj = [0] * n
        trial_deg = [0] * n
        trial_count = 0
        
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            new_adj = trial_adj[:]
            new_deg = trial_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_fast(new_adj, new_deg):
                trial_adj = new_adj
                trial_deg = new_deg
                trial_count += 1
        
        if trial_count > best_count:
            best_count = trial_count
            best_adj = trial_adj[:]
            best_deg = trial_deg[:]
            
        # If we hit a high threshold, stop early
        if best_count >= 35:
            break

    # Local search from best
    current_adj = best_adj[:]
    current_deg = best_deg[:]
    current_count = best_count
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        no_improve += 1
        
        # Try to add a random non-edge
        added = False
        # Sample 50 random non-edges
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_fast(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_adj = current_adj[:]
                    best_deg = current_deg[:]
                added = True
                no_improve = 0
                break
        
        if added:
            continue
            
        # Try swap: remove one edge, add another
        swapped = False
        # Pick random edge to remove
        edge_to_remove = None
        # Sample 50 random edges
        for _ in range(50):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if current_adj[u] & (1 << v):
                edge_to_remove = (u, v)
                break
        
        if edge_to_remove is None:
            break
            
        u1, v1 = edge_to_remove
        
        # Pick random non-edge to add
        for _ in range(100):
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            if (u2, v2) == (u1, v1):
                continue
            if current_adj[u2] & (1 << v2):
                continue
                
            new_adj = current_adj[:]
            new_deg = current_deg[:]
            new_adj[u1] &= ~(1 << v1)
            new_adj[v1] &= ~(1 << u1)
            new_adj[u2] |= (1 << v2)
            new_adj[v2] |= (1 << u2)
            new_deg[u1] -= 1
            new_deg[v1] -= 1
            new_deg[u2] += 1
            new_deg[v2] += 1
            
            if is_valid_fast(new_adj, new_deg):
                current_adj = new_adj
                current_deg = new_deg
                # Count stays same
                swapped = True
                no_improve = 0
                break
        
        if not swapped:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart from best
            current_adj = best_adj[:]
            current_deg = best_deg[:]
            current_count = best_count
            no_improve = 0
            max_no_improve = int(max_no_improve * 1.5)
            
            # If we've been restarting a lot, try a different random seed for greedy phase?
            # For now, just break if we've been at best for too long
            if best_count < 20:
                 break

    final_edges = adj_to_edges(best_adj)
    final_edges = [[int(e[0]), int(e[1])] for e in final_edges]
    
    return final_edges
