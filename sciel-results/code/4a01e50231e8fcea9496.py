import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    
    # Precompute edge index mapping
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Bitmask helper
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]

    def can_add(u, v, adj_mask):
        """
        Check if adding edge (u, v) keeps the graph K3,3-free.
        A K3,3 is formed if there exist a1, a2 in N(u) and b1, b2 in N(v)
        such that a1, a2 are both connected to b1, b2.
        """
        if (adj_mask[u] >> v) & 1:
            return True
        
        N_u = adj_mask[u]
        N_v = adj_mask[v]
        
        # We need to check pairs (b1, b2) in N_v
        # For each pair, check if there are at least 2 vertices in N_u connected to both b1 and b2.
        
        # Extract list of neighbors of v
        nv_list = mask_to_list(N_v)
        
        for i in range(len(nv_list)):
            b1 = nv_list[i]
            for j in range(i + 1, len(nv_list)):
                b2 = nv_list[j]
                # Common neighbors of b1 and b2
                common_b1_b2 = adj_mask[b1] & adj_mask[b2]
                # Check how many of these are in N_u
                # If >= 2, we have a1, a2 in N_u connected to both b1, b2.
                # This forms a K2,2 between {a1, a2} and {b1, b2}.
                # Combined with edges (u, b1), (u, b2), (a1, v), (a2, v), this is a K3,3.
                # Note: u is connected to b1, b2. v is connected to a1, a2.
                # The K3,3 partitions are {u, a1, a2} and {v, b1, b2}.
                # Edges:
                # u-b1, u-b2 (exist)
                # a1-v, a2-v (exist because a1, a2 in N_u? NO. a1, a2 are in N_u, so u-a1, u-a2 exist.
                # Wait. The logic above was:
                # a1, a2 in N(u). So edges u-a1, u-a2 exist.
                # b1, b2 in N(v). So edges v-b1, v-b2 exist.
                # We need a1-b1, a1-b2, a2-b1, a2-b2.
                # My check: common_b1_b2 contains vertices connected to both b1 and b2.
                # If a1 is in common_b1_b2, then a1-b1 and a1-b2 exist.
                # So if |common_b1_b2 & N_u| >= 2, we found a1, a2.
                # This confirms the K3,3.
                if (common_b1_b2 & N_u).bit_count() >= 2:
                    return False
        return True

    def get_edge_count(adj_mask):
        return sum(mask.bit_count() for mask in adj_mask) // 2

    def is_valid(adj_mask):
        """Check if the graph is K3,3-free."""
        for i in range(n):
            for j in range(i + 1, n):
                mij = adj_mask[i] & adj_mask[j]
                for k in range(j + 1, n):
                    if (mij & adj_mask[k]).bit_count() >= 3:
                        return False
        return True

    def find_violations(adj_mask):
        """Find all violating triples."""
        viols = []
        for i in range(n):
            for j in range(i + 1, n):
                mij = adj_mask[i] & adj_mask[j]
                for k in range(j + 1, n):
                    if (mij & adj_mask[k]).bit_count() >= 3:
                        viols.append((i, j, k))
        return viols

    def polish(adj_mask, edges_set, time_budget=10.0):
        """Local search to maximize edges."""
        polish_start = time.time()
        polish_deadline = polish_start + time_budget
        improved = True
        no_improve_count = 0
        while improved and time.time() < polish_deadline:
            improved = False
            no_improve_count += 1
            if no_improve_count > 3:
                break
            
            # 1. Try adding missing edges
            missing = [e for e in all_edges if e not in edges_set]
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > polish_deadline:
                    return adj_mask, edges_set
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges_set.add((u, v))
                    improved = True
            
            # 2. Try edge swaps (remove 1, add 1)
            if time.time() < polish_deadline - 1:
                edge_list = list(edges_set)
                random.shuffle(edge_list)
                for e1 in edge_list[:50]:  # Limit to 50 swaps per pass
                    if time.time() >= polish_deadline - 1:
                        break
                    u1, v1 = e1
                    # Remove e1
                    adj_mask[u1] &= ~(1 << v1)
                    adj_mask[v1] &= ~(1 << u1)
                    edges_set.discard(e1)
                    
                    # Try to add a different edge
                    missing = [e for e in all_edges if e not in edges_set]
                    random.shuffle(missing)
                    added = None
                    for e2 in missing:
                        if can_add(e2[0], e2[1], adj_mask):
                            added = e2
                            break
                    
                    if added:
                        u2, v2 = added
                        adj_mask[u2] |= (1 << v2)
                        adj_mask[v2] |= (1 << u2)
                        edges_set.add(added)
                        improved = True
                    else:
                        # Undo removal
                        adj_mask[u1] |= (1 << v1)
                        adj_mask[v1] |= (1 << u1)
                        edges_set.add(e1)

        return adj_mask, edges_set

    def repair_mask(adj_mask):
        """Remove edges to eliminate all violations."""
        max_iter = 5000
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            
            viols = find_violations(adj_mask)
            if not viols:
                return True
            
            # Find edges to remove - prioritize edges in violating triples
            edge_viol_count = {}
            for i, j, k in viols:
                for v in range(n):
                    if v not in (i, j, k):
                        if adj_mask[i] & (1 << v) and adj_mask[j] & (1 << v) and adj_mask[k] & (1 << v):
                            for a, b in [(i, v), (j, v), (k, v)]:
                                key = (min(a, b), max(a, b))
                                edge_viol_count[key] = edge_viol_count.get(key, 0) + 1
            
            if edge_viol_count:
                worst_edge = max(edge_viol_count, key=edge_viol_count.get)
                u, v = worst_edge
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                # Fallback: remove random edge from first violation
                i, j, k = viols[0]
                candidates = []
                for v in range(n):
                    if v != i and adj_mask[i] & (1 << v):
                        candidates.append((i, v))
                    if v != j and adj_mask[j] & (1 << v):
                        candidates.append((j, v))
                    if v != k and adj_mask[k] & (1 << v):
                        candidates.append((k, v))
                if not candidates:
                    return False
                u, v = random.choice(candidates)
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
        
        return is_valid(adj_mask)

    def mask_to_edges(adj_mask):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mask[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def edges_to_mask(edges):
        adj_mask = [0] * n
        for u, v in edges:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask

    def construct_polarity_graph():
        """
        Construct a high-density base graph using a polarity-like construction.
        For n=20, we use a structured random approach with high initial density.
        """
        adj_mask = [0] * n
        edges_set = set()
        
        # Start with a moderately dense random graph
        p = 0.5
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if random.random() < p:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges_set.add((u, v))
        
        return adj_mask, edges_set

    def construct_strongly_regular_like():
        """
        Try to construct a graph with properties similar to strongly regular graphs.
        For n=20, we aim for a graph with high edge density and low common neighbor counts.
        """
        adj_mask = [0] * n
        edges_set = set()
        
        # Use a structured pattern: connect vertices based on modular arithmetic
        # This creates a more regular structure than pure random
        for i in range(n):
            for j in range(i + 1, n):
                # Connect if the difference is in a specific set
                diff = (j - i) % n
                if diff in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                    if can_add(i, j, adj_mask):
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                        edges_set.add((i, j))
        
        return adj_mask, edges_set

    def simulated_annealing(adj_mask, time_budget=15.0):
        """Run SA on the adjacency mask."""
        sa_start = time.time()
        sa_deadline = sa_start + time_budget
        
        current_adj = adj_mask[:]
        current_edges_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if current_adj[i] & (1 << j):
                    current_edges_set.add((i, j))
        
        current_count = get_edge_count(current_adj)
        
        best_adj = current_adj[:]
        best_edges = current_edges_set.copy()
        best_count = current_count
        
        T = 30.0
        T_min = 0.01
        cooling_rate = 0.9998
        
        rng = random.Random(42)
        
        while T > T_min and time.time() < sa_deadline:
            # Pick a random edge to flip
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_val = 1 if (current_adj[i] & (1 << j)) else 0
            
            # Temporarily flip
            if old_val:
                current_adj[i] &= ~(1 << j)
                current_adj[j] &= ~(1 << i)
                current_edges_set.discard((i, j))
            else:
                current_adj[i] |= (1 << j)
                current_adj[j] |= (1 << i)
                current_edges_set.add((i, j))
            
            new_count = get_edge_count(current_adj)
            is_valid_now = is_valid(current_adj)
            
            # Energy: -count + penalty for violations
            if is_valid_now:
                new_energy = -new_count
            else:
                viols = find_violations(current_adj)
                new_energy = -new_count + 100 * len(viols)
            
            old_energy = -current_count if is_valid(adj_mask) else -current_count + 100 * len(find_violations(adj_mask))
            
            delta_E = new_energy - old_energy
            
            if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                # Accept
                current_count = new_count
                if is_valid_now and new_count > best_count:
                    best_adj = current_adj[:]
                    best_edges = current_edges_set.copy()
                    best_count = new_count
            else:
                # Reject - flip back
                if old_val:
                    current_adj[i] |= (1 << j)
                    current_adj[j] |= (1 << i)
                    current_edges_set.add((i, j))
                else:
                    current_adj[i] &= ~(1 << j)
                    current_adj[j] &= ~(1 << i)
                    current_edges_set.discard((i, j))
            
            T *= cooling_rate
        
        return best_adj, best_edges, best_count

    # Main search
    random.seed(42)
    best_mask = None
    best_count = -1
    
    # Strategy 1: Polarity-like construction + polish
    if time.time() < deadline - 30:
        adj_mask, edges_set = construct_polarity_graph()
        if repair_mask(adj_mask):
            adj_mask, edges_set = polish(adj_mask, edges_set, time_budget=min(20, deadline - time.time() - 20))
            ec = get_edge_count(adj_mask)
            if ec > best_count:
                best_count = ec
                best_mask = adj_mask[:]
        
        # SA refinement
        if time.time() < deadline - 15:
            sa_adj, sa_edges, sa_count = simulated_annealing(adj_mask, time_budget=min(15, deadline - time.time() - 5))
            if is_valid(sa_adj) and sa_count > best_count:
                best_count = sa_count
                best_mask = sa_adj[:]

    # Strategy 2: Strongly regular-like construction + polish
    if time.time() < deadline - 30:
        adj_mask, edges_set = construct_strongly_regular_like()
        if repair_mask(adj_mask):
            adj_mask, edges_set = polish(adj_mask, edges_set, time_budget=min(20, deadline - time.time() - 20))
            ec = get_edge_count(adj_mask)
            if ec > best_count:
                best_count = ec
                best_mask = adj_mask[:]
        
        if time.time() < deadline - 15:
            sa_adj, sa_edges, sa_count = simulated_annealing(adj_mask, time_budget=min(15, deadline - time.time() - 5))
            if is_valid(sa_adj) and sa_count > best_count:
                best_count = sa_count
                best_mask = sa_adj[:]

    # Strategy 3: Multiple random restarts with polish
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        
        adj_mask = [0] * n
        edges_set = set()
        
        # Random greedy initialization with varying density
        p = 0.4 + 0.1 * trial
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if random.random() < p:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
                    edges_set.add((u, v))
        
        if repair_mask(adj_mask):
            adj_mask, edges_set = polish(adj_mask, edges_set, time_budget=min(15, deadline - time.time() - 10))
            ec = get_edge_count(adj_mask)
            if ec > best_count:
                best_count = ec
                best_mask = adj_mask[:]
        
        if time.time() < deadline - 10:
            sa_adj, sa_edges, sa_count = simulated_annealing(adj_mask, time_budget=min(10, deadline - time.time() - 5))
            if is_valid(sa_adj) and sa_count > best_count:
                best_count = sa_count
                best_mask = sa_adj[:]

    # Strategy 4: Perturb best solution and re-optimize
    if best_mask is not None and time.time() < deadline - 10:
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            mask = best_mask[:]
            # Perturb: remove some random edges
            for i in range(n):
                for j in range(i + 1, n):
                    if mask[i] & (1 << j) and random.random() < 0.2:
                        mask[i] &= ~(1 << j)
                        mask[j] &= ~(1 << i)
            
            if repair_mask(mask):
                edges_set = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if mask[i] & (1 << j):
                            edges_set.add((i, j))
                
                mask, edges_set = polish(mask, edges_set, time_budget=min(10, deadline - time.time() - 5))
                ec = get_edge_count(mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = mask[:]

    if best_mask is None:
        return []
    
    edges = mask_to_edges(best_mask)
    return edges
