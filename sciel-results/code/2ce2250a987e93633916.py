import time
import random
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for the 120s limit

    def compute_common(adj):
        """Compute common neighbor matrix from adjacency matrix."""
        return (adj @ adj).astype(np.int32)

    def is_k23_free(adj, common):
        """Check if the graph is K2,3-free."""
        return np.all(common[np.triu_indices(n, k=1)] <= 2)

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    def can_add_edge(adj, common, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        if adj[u, v]:
            return False
        if common[u, v] > 2:
            return False
        # For all w in N(v), common[u,w] must stay <= 2 after adding (u,v)
        # Adding (u,v) increases common[u,w] by 1 for all w in N(v)
        nv = np.nonzero(adj[v])[0]
        nv = nv[nv != u]
        if len(nv) > 0:
            if np.any(common[u, nv] >= 2):
                return False
        # For all w in N(u), common[v,w] must stay <= 2 after adding (u,v)
        nu = np.nonzero(adj[u])[0]
        nu = nu[nu != v]
        if len(nu) > 0:
            if np.any(common[v, nu] >= 2):
                return False
        return True

    def add_edge(adj, common, u, v):
        """Add edge (u,v) and update common neighbor counts."""
        nv = np.nonzero(adj[v])[0]
        nu = np.nonzero(adj[u])[0]
        adj[u, v] = 1
        adj[v, u] = 1
        if len(nv) > 0:
            common[u, nv] += 1
            common[nv, u] += 1
        if len(nu) > 0:
            common[v, nu] += 1
            common[nu, v] += 1

    def remove_edge(adj, common, u, v):
        """Remove edge (u,v) and update common neighbor counts."""
        nv = np.nonzero(adj[v])[0]
        nu = np.nonzero(adj[u])[0]
        adj[u, v] = 0
        adj[v, u] = 0
        if len(nv) > 0:
            common[u, nv] -= 1
            common[nv, u] -= 1
        if len(nu) > 0:
            common[v, nu] -= 1
            common[nu, v] -= 1

    def greedy_add(adj, common, max_time):
        """Greedily add all possible edges while maintaining K2,3-freeness."""
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v] and can_add_edge(adj, common, u, v):
                        add_edge(adj, common, u, v)
                        improved = True
        return adj, common

    def circulant_init(steps):
        """Create a circulant graph with given step set."""
        adj = np.zeros((n, n), dtype=np.int8)
        for s in steps:
            for i in range(n):
                j = (i + s) % n
                if j > i:
                    adj[i, j] = 1
                    adj[j, i] = 1
        return adj

    def simulated_annealing(adj, common, max_time):
        """Simulated annealing with QUBO-inspired energy landscape."""
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj.copy()
        best_common = common.copy()
        
        T = 5.0  # Initial temperature
        cooling_rate = 0.9995
        
        while time.time() < max_time and T > 0.01:
            # Generate a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add an edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if adj[u, v]:
                    continue
                if can_add_edge(adj, common, u, v):
                    add_edge(adj, common, u, v)
                    new_edges = current_edges + 1
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = adj.copy()
                        best_common = common.copy()
                    current_edges = new_edges
                else:
                    # Can't add directly; try swap: remove one edge then add
                    pass
            elif move_type < 0.7:
                # Try to remove an edge (to enable future additions)
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                if not edges:
                    T *= cooling_rate
                    continue
                u, v = random.choice(edges)
                remove_edge(adj, common, u, v)
                current_edges -= 1
                # Try to add a different edge
                added = False
                for _ in range(20):
                    au = random.randint(0, n - 1)
                    av = random.randint(0, n - 1)
                    if au == av:
                        continue
                    if au > av:
                        au, av = av, au
                    if not adj[au, av] and can_add_edge(adj, common, au, av):
                        add_edge(adj, common, au, av)
                        current_edges += 1
                        added = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj.copy()
                            best_common = common.copy()
                        break
                if not added:
                    # Restore removed edge
                    add_edge(adj, common, u, v)
                    current_edges += 1
            else:
                # Swap: remove one edge, add another
                edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                if not edges or not non_edges:
                    T *= cooling_rate
                    continue
                rem_u, rem_v = random.choice(edges)
                add_u, add_v = random.choice(non_edges)
                
                # Check if we can do the swap
                if can_add_edge(adj, common, add_u, add_v):
                    # Direct swap possible
                    remove_edge(adj, common, rem_u, rem_v)
                    add_edge(adj, common, add_u, add_v)
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj.copy()
                        best_common = common.copy()
                else:
                    # Try: remove first, then add
                    remove_edge(adj, common, rem_u, rem_v)
                    if can_add_edge(adj, common, add_u, add_v):
                        add_edge(adj, common, add_u, add_v)
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj.copy()
                            best_common = common.copy()
                    else:
                        # Restore
                        add_edge(adj, common, rem_u, rem_v)
            
            T *= cooling_rate
            
            # Periodic greedy fill
            if random.random() < 0.05:
                adj, common = greedy_add(adj, common, max_time)
                current_edges = count_edges(adj)
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj.copy()
                    best_common = common.copy()
        
        return best_adj, best_common, best_edges

    def local_search_polish(adj, common, max_time):
        """Final polish: try all possible edge additions and swaps."""
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj.copy()
        best_common = common.copy()
        
        # First, greedy add all possible edges
        adj, common = greedy_add(adj, common, max_time)
        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
            best_common = common.copy()
        
        # Try systematic swaps
        while time.time() < max_time:
            improved = False
            for ru in range(n):
                for rv in range(ru + 1, n):
                    if not adj[ru, rv]:
                        continue
                    # Try removing this edge and adding a different one
                    remove_edge(adj, common, ru, rv)
                    for au in range(n):
                        for av in range(au + 1, n):
                            if not adj[au, av] and can_add_edge(adj, common, au, av):
                                add_edge(adj, common, au, av)
                                new_edges = count_edges(adj)
                                if new_edges > best_edges:
                                    best_edges = new_edges
                                    best_adj = adj.copy()
                                    best_common = common.copy()
                                    improved = True
                                # Revert for next try
                                remove_edge(adj, common, au, av)
                                break
                        else:
                            continue
                        break
                    # Restore original edge
                    add_edge(adj, common, ru, rv)
                    
                    if time.time() > max_time:
                        break
                if time.time() > max_time:
                    break
            if not improved and time.time() < max_time:
                # No improvement in full pass; try random swaps
                for _ in range(100):
                    if time.time() > max_time:
                        break
                    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
                    non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                    if not edges or not non_edges:
                        break
                    ru, rv = random.choice(edges)
                    au, av = random.choice(non_edges)
                    remove_edge(adj, common, ru, rv)
                    if can_add_edge(adj, common, au, av):
                        add_edge(adj, common, au, av)
                        new_edges = count_edges(adj)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = adj.copy()
                            best_common = common.copy()
                    else:
                        add_edge(adj, common, ru, rv)
            else:
                break
        
        return best_adj, best_common, best_edges

    # Main algorithm: multiple restarts with different initializations
    best_score = 0
    best_adj = None

    # Try different circulant initializations
    circulant_steps = [
        [1, 2, 3, 4],
        [1, 2, 3, 5],
        [1, 2, 4, 5],
        [1, 3, 4, 5],
        [1, 2, 3, 4, 5],
        [1, 2, 4, 7],
        [1, 3, 4, 7],
        [2, 3, 4, 5],
        [1, 2, 3, 6],
        [1, 3, 5, 7],
    ]

    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart < len(circulant_steps):
            # Circulant initialization
            adj = circulant_init(circulant_steps[restart])
        elif restart < len(circulant_steps) + 5:
            # Random dense graph
            adj = np.zeros((n, n), dtype=np.int8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.4:
                        adj[u, v] = 1
                        adj[v, u] = 1
        else:
            # Random sparse graph
            adj = np.zeros((n, n), dtype=np.int8)
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.25:
                        adj[u, v] = 1
                        adj[v, u] = 1
        
        common = compute_common(adj)
        
        # Repair if needed (remove edges causing K2,3)
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v] and common[u, v] > 2:
                        remove_edge(adj, common, u, v)
                        changed = True
        
        # Greedy add
        adj, common = greedy_add(adj, common, deadline)
        
        # Simulated annealing
        adj, common, count = simulated_annealing(adj, common, deadline)
        
        # Final polish
        adj, common, count = local_search_polish(adj, common, deadline)
        
        if count > best_score:
            best_score = count
            best_adj = adj.copy()
            best_common = common.copy()
    
    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
    
    return best_edges
