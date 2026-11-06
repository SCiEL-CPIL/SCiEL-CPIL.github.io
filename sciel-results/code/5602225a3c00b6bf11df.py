import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin 10s
    
    def get_seed_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    best_n, best_edges = get_seed_graph()
    
    # Targets to try, descending
    targets = [42, 41, 40, 39, 38]
    
    for target_n in targets:
        if time.time() > deadline:
            break
        
        time_budget = deadline - time.time()
        if time_budget < 10:
            break
            
        found = solve_for_n(target_n, time_budget)
        if found:
            n, edges = found
            if n > best_n:
                best_n, best_edges = n, edges
                if n >= 42:
                    break
    return best_n, best_edges

def solve_for_n(n, time_budget):
    max_restarts = max(1, int(time_budget / 20))
    random.seed(42 + n)
    np.random.seed(42 + n)
    
    # Precompute all 5-combinations
    all_combos = np.array(list(itertools.combinations(range(n), 5)), dtype=np.int32)
    num_combos = len(all_combos)
    
    for r in range(max_restarts):
        if time.time() > time_budget:
            break
        
        # Try different initialization strategies
        if r % 3 == 0:
            # Random initialization
            density = 0.5
            adj = (np.random.rand(n, n) < density).astype(np.int8)
            np.fill_diagonal(adj, 0)
            adj = np.maximum(adj, adj.T)
            np.fill_diagonal(adj, 0)
        elif r % 3 == 1:
            # Clebsch-inspired structured initialization
            adj = init_clebsch_inspired(n, np.random)
        else:
            # Random with varied density
            density = random.choice([0.4, 0.5, 0.6])
            adj = (np.random.rand(n, n) < density).astype(np.int8)
            np.fill_diagonal(adj, 0)
            adj = np.maximum(adj, adj.T)
            np.fill_diagonal(adj, 0)
        
        # Count initial violations
        violations = count_violations_np(n, adj, all_combos)
        best_violations = violations
        
        # Local search loop
        max_steps = 2000
        for step in range(max_steps):
            if time.time() > time_budget:
                break
            
            if violations == 0:
                edges_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges_list.append((i, j))
                return n, edges_list
            
            # Find a violation
            combo = find_one_violation_np(n, adj, all_combos)
            if combo is None:
                edges_list = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j]:
                            edges_list.append((i, j))
                return n, edges_list
            
            # Get all pairs in the violating combo
            pairs = list(itertools.combinations(combo, 2))
            random.shuffle(pairs)
            
            best_flip = None
            best_new_violations = violations + 100
            
            # Try up to 10 flips
            for u, v in pairs[:10]:
                has_edge = adj[u, v]
                
                adj[u, v] = 1 - has_edge
                adj[v, u] = 1 - has_edge
                
                new_violations = count_violations_np(n, adj, all_combos)
                
                adj[u, v] = has_edge
                adj[v, u] = has_edge
                
                if new_violations < best_new_violations:
                    best_new_violations = new_violations
                    best_flip = (u, v, has_edge)
            
            if best_flip:
                u, v, was_edge = best_flip
                adj[u, v] = 1 - was_edge
                adj[v, u] = 1 - was_edge
                violations = best_new_violations
                
                if violations < best_violations:
                    best_violations = violations
            else:
                # Random flip to escape local minimum
                u, v = random.sample(range(n), 2)
                if u > v:
                    u, v = v, u
                adj[u, v] = 1 - adj[u, v]
                adj[v, u] = 1 - adj[v, u]
                violations = count_violations_np(n, adj, all_combos)
    
    return None

def init_clebsch_inspired(n, rng):
    """Initialize using Clebsch graph complement structure, expanded to n vertices"""
    adj = np.zeros((n, n), dtype=np.int8)
    
    # Clebsch complement on first 16 vertices
    # Clebsch: 16 vertices (even-parity 4-bit strings), edges at Hamming dist 1 or 3
    # Complement: edges at Hamming dist 2 or 4
    for i in range(min(16, n)):
        for j in range(i + 1, min(16, n)):
            xor_val = i ^ j
            pc = bin(xor_val).count('1')
            if pc == 2 or pc == 4:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Connect remaining vertices with structured randomness
    for i in range(16, n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
        # Also connect to first 16 vertices
        for j in range(min(16, n)):
            if rng.random() < 0.5:
                adj[i, j] = 1
                adj[j, i] = 1
    
    return adj

def count_violations_np(n, adj, all_combos):
    """Count K5 and IK5 violations using numpy vectorization"""
    if len(all_combos) == 0:
        return 0
    
    a = all_combos[:, 0]
    b = all_combos[:, 1]
    c = all_combos[:, 2]
    d = all_combos[:, 3]
    e = all_combos[:, 4]
    
    edge_ab = adj[a, b]
    edge_ac = adj[a, c]
    edge_ad = adj[a, d]
    edge_ae = adj[a, e]
    edge_bc = adj[b, c]
    edge_bd = adj[b, d]
    edge_be = adj[b, e]
    edge_cd = adj[c, d]
    edge_ce = adj[c, e]
    edge_de = adj[d, e]
    
    is_k5 = (edge_ab == 1) & (edge_ac == 1) & (edge_ad == 1) & (edge_ae == 1) & \
            (edge_bc == 1) & (edge_bd == 1) & (edge_be == 1) & \
            (edge_cd == 1) & (edge_ce == 1) & (edge_de == 1)
    
    is_ik5 = (edge_ab == 0) & (edge_ac == 0) & (edge_ad == 0) & (edge_ae == 0) & \
             (edge_bc == 0) & (edge_bd == 0) & (edge_be == 0) & \
             (edge_cd == 0) & (edge_ce == 0) & (edge_de == 0)
    
    return int(np.sum(is_k5) + np.sum(is_ik5))

def find_one_violation_np(n, adj, all_combos):
    """Find one K5 or IK5 violation"""
    if len(all_combos) == 0:
        return None
    
    a = all_combos[:, 0]
    b = all_combos[:, 1]
    c = all_combos[:, 2]
    d = all_combos[:, 3]
    e = all_combos[:, 4]
    
    edge_ab = adj[a, b]
    edge_ac = adj[a, c]
    edge_ad = adj[a, d]
    edge_ae = adj[a, e]
    edge_bc = adj[b, c]
    edge_bd = adj[b, d]
    edge_be = adj[b, e]
    edge_cd = adj[c, d]
    edge_ce = adj[c, e]
    edge_de = adj[d, e]
    
    is_k5 = (edge_ab == 1) & (edge_ac == 1) & (edge_ad == 1) & (edge_ae == 1) & \
            (edge_bc == 1) & (edge_bd == 1) & (edge_be == 1) & \
            (edge_cd == 1) & (edge_ce == 1) & (edge_de == 1)
    
    if np.any(is_k5):
        idx = np.where(is_k5)[0][0]
        return tuple(all_combos[idx])
    
    is_ik5 = (edge_ab == 0) & (edge_ac == 0) & (edge_ad == 0) & (edge_ae == 0) & \
             (edge_bc == 0) & (edge_bd == 0) & (edge_be == 0) & \
             (edge_cd == 0) & (edge_ce == 0) & (edge_de == 0)
    
    if np.any(is_ik5):
        idx = np.where(is_ik5)[0][0]
        return tuple(all_combos[idx])
    
    return None
