import time
import random
import sys

def construct():
    n = 15
    start = time.time()
    deadline = start + 80.0  # Safety margin before 90s kill

    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_possible = len(all_edges)  # 105

    # Precompute for each vertex, the list of other vertices
    # And for fast bitmask operations
    # adj_mask[v] = bitmask of neighbors of v

    def check_violation(adj_mask, deg):
        """
        Check if the graph (defined by adj_mask, deg) has any violation.
        A violation: path s-u-v-t (4 distinct vertices, 3 edges) where deg(s) == deg(t).
        Uses degree-class bitmasks for speed.
        """
        # Group vertices by degree
        deg_class = {}
        for v in range(n):
            d = deg[v]
            if d not in deg_class:
                deg_class[d] = 0
            deg_class[d] |= (1 << v)

        # For each edge (u, v), check if there exist s, t with deg(s)==deg(t)
        # such that s~u, t~v, s!=t, s,t not in {u,v}
        # OR s~v, t~u, s!=t, s,t not in {u,v}
        
        # For each edge (u, v):
        #   neighbors_u = adj_mask[u]
        #   neighbors_v = adj_mask[v]
        #   s can be in neighbors_u (excluding u, v)
        #   t can be in neighbors_v (excluding u, v)
        #   Need deg(s) == deg(t)
        
        # Fast check: for each degree class d, let S_d = deg_class[d]
        # For edge (u,v):
        #   s_candidates = neighbors_u & ~(1<<u) & ~(1<<v)
        #   t_candidates = neighbors_v & ~(1<<u) & ~(1<<v)
        #   For each degree d:
        #     s_in_class = s_candidates & deg_class[d]
        #     t_in_class = t_candidates & deg_class[d]
        #     If s_in_class and t_in_class:
        #       If s_in_class != t_in_class (as sets, i.e., there exist s!=t):
        #         Actually we need s != t. 
        #         s_in_class and t_in_class are bitmasks.
        #         If (s_in_class & t_in_class) == 0, then all s in s_in_class differ from all t in t_in_class? No.
        #         We need: exists s in s_in_class, t in t_in_class, s != t.
        #         This is true if:
        #           - |s_in_class| >= 1 and |t_in_class| >= 1 and (s_in_class != t_in_class or |s_in_class| >= 2)
        #         More simply: if s_in_class & t_in_class is non-empty, then there's s=t which is bad for "s!=t" 
        #         but we just need SOME pair with s!=t.
        #         If s_in_class and t_in_class are both non-empty:
        #           If they share no vertices, then any s from s_in_class and t from t_in_class have s!=t. Valid.
        #           If they share some vertices, we need at least one s in s_in_class that is not in t_in_class, OR one t in t_in_class not in s_in_class.
        #           i.e., (s_in_class & ~t_in_class) != 0 OR (t_in_class & ~s_in_class) != 0
        #           Which is equivalent to s_in_class != t_in_class.
        
        for (u, v) in all_edges:
            if not (adj_mask[u] & (1 << v)):
                continue  # Edge not present
            
            s_cand = adj_mask[u] & ~(1 << u) & ~(1 << v)
            t_cand = adj_mask[v] & ~(1 << u) & ~(1 << v)
            
            if s_cand == 0 or t_cand == 0:
                continue
            
            # Check if there exists a degree class where both s_cand and t_cand have vertices, with s != t possible
            for d, mask in deg_class.items():
                s_in = s_cand & mask
                t_in = t_cand & mask
                if s_in == 0 or t_in == 0:
                    continue
                # Need s != t: possible if s_in != t_in
                if s_in != t_in:
                    return True

        return False

    def count_edges(adj_mask):
        count = 0
        for v in range(n):
            count += bin(adj_mask[v]).count('1')
        return count // 2

    def build_deg(adj_mask):
        deg = [0] * n
        for v in range(n):
            deg[v] = bin(adj_mask[v]).count('1')
        return deg

    def is_valid(adj_mask):
        deg = build_deg(adj_mask)
        return not check_violation(adj_mask, deg)

    # Diverse seeds
    def seed_bipartite():
        # K_{7,8}: vertices 0-6 connected to 7-14
        adj = [0] * n
        for i in range(7):
            for j in range(7, 15):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        return adj

    def seed_near_complete():
        # Start with complete graph, remove edges to fix violations
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        # Remove edges until valid
        # This is a heuristic seed; may need many removals
        # Just use it as a starting point for SA
        return adj

    def seed_random_dense():
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.7:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def seed_ring_plus_chords():
        # Ring 0-1-2-...-14-0 plus some chords
        adj = [0] * n
        for i in range(n):
            adj[i] |= (1 << ((i+1) % n))
            adj[i] |= (1 << ((i-1) % n))
        # Add some random chords
        for _ in range(10):
            u, v = random.sample(range(n), 2)
            if not (adj[u] & (1 << v)):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    seeds = [
        seed_bipartite(),
        seed_random_dense(),
        seed_ring_plus_chords(),
    ]
    # Add a few more random seeds
    for _ in range(3):
        seeds.append(seed_random_dense())

    best_adj = None
    best_count = 0

    # Simulated annealing
    def simulated_annealing(initial_adj, max_time=10.0):
        adj = initial_adj[:]
        deg = build_deg(adj)
        
        # Ensure initial is valid; if not, remove random edges until valid
        while check_violation(adj, deg):
            # Find a violating edge and remove it
            # For simplicity, remove a random edge
            present_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
            if not present_edges:
                return adj, 0
            u, v = random.choice(present_edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            deg[u] -= 1
            deg[v] -= 1
        
        current_count = count_edges(adj)
        best_adj_local = adj[:]
        best_count_local = current_count
        
        T = 10.0  # Initial temperature
        cooling = 0.9995
        
        sa_start = time.time()
        while time.time() < sa_start + max_time:
            if time.time() > deadline - 2:
                break
            
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Add an edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                if not missing:
                    continue
                u, v = random.choice(missing)
                new_adj = adj[:]
                new_adj[u] |= (1 << v)
                new_adj[v] |= (1 << u)
                new_deg = deg[:]
                new_deg[u] += 1
                new_deg[v] += 1
                new_count = current_count + 1
                
                if check_violation(new_adj, new_deg):
                    # Reject
                    T *= cooling
                    continue
                
                delta = new_count - current_count
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    adj = new_adj
                    deg = new_deg
                    current_count = new_count
                    if current_count > best_count_local:
                        best_count_local = current_count
                        best_adj_local = adj[:]
                
            elif move_type < 0.6:
                # Remove an edge
                present = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
                if not present:
                    continue
                u, v = random.choice(present)
                new_adj = adj[:]
                new_adj[u] &= ~(1 << v)
                new_adj[v] &= ~(1 << u)
                new_deg = deg[:]
                new_deg[u] -= 1
                new_deg[v] -= 1
                new_count = current_count - 1
                
                # Removing an edge always preserves validity (fewer paths)
                # Actually, removing an edge can't create new violations, only remove them.
                # So it's always valid.
                delta = new_count - current_count
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    adj = new_adj
                    deg = new_deg
                    current_count = new_count
                    if current_count > best_count_local:
                        best_count_local = current_count
                        best_adj_local = adj[:]
                        
            else:
                # Swap: remove one edge, add another
                present = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                if not present or not missing:
                    continue
                u1, v1 = random.choice(present)
                u2, v2 = random.choice(missing)
                # Ensure we're not just adding back the same edge
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                
                new_adj = adj[:]
                new_adj[u1] &= ~(1 << v1)
                new_adj[v1] &= ~(1 << u1)
                new_adj[u2] |= (1 << v2)
                new_adj[v2] |= (1 << u2)
                
                new_deg = deg[:]
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_count = current_count  # Same number of edges
                
                if check_violation(new_adj, new_deg):
                    T *= cooling
                    continue
                
                delta = 0  # Same edge count
                # For same edge count, only accept if it might lead to future improvements
                # Use a small acceptance probability for neutral moves
                if random.random() < 0.3:
                    adj = new_adj
                    deg = new_deg
                    current_count = new_count
                    if current_count > best_count_local:
                        best_count_local = current_count
                        best_adj_local = adj[:]
            
            T *= cooling
            if T < 0.01:
                T = 0.01
        
        return best_adj_local, best_count_local

    # Run SA from each seed
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        sa_time = min(10.0, max(2.0, deadline - 5 - time.time()))
        adj_result, count_result = simulated_annealing(seed, sa_time)
        if count_result > best_count:
            best_count = count_result
            best_adj = adj_result[:]

    # Final greedy improvement on best
    if best_adj is not None:
        deg = build_deg(best_adj)
        # Try to add edges greedily
        for (u, v) in all_edges:
            if time.time() > deadline - 2:
                break
            if best_adj[u] & (1 << v):
                continue
            new_adj = best_adj[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg = deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            if not check_violation(new_adj, new_deg):
                best_adj = new_adj
                deg = new_deg
                best_count += 1

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                edges.append([int(i), int(j)])
    
    return edges

