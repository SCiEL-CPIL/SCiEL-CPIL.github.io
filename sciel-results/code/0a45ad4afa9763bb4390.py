import time
import random
import sys

N = 40
DEADLINE = 115  # Safety margin under 150s kill

# Precompute all possible edges
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_ALL_EDGES = len(ALL_EDGES)

# For fast C6 check, we need to check if adding (u,v) creates a path u-a-b-c-d-v
# We will maintain adjacency bitmasks.
# To check if adding (u,v) creates a C6, we check if there is a path of length 4 between u and v in G - {u,v} ?
# No, a C6 is u-a-b-c-d-v-u. So we need a path of length 4 from u to v in G \ {u,v} such that the internal vertices are distinct and not u,v.
# Actually, it's simpler:
# A C6 containing edge (u,v) exists iff there is a path of length 4 from u to v in G \ {u,v}.
# Path: u - x - y - z - v.
# x in N(u), z in N(v), x != z, x != v, z != u.
# y must be in N(x) ∩ N(z) and y not in {u, v, x, z}.
# Also need to ensure the path is simple (vertices distinct).
# If x, y, z are distinct and not u, v, then the path u-x-y-z-v is a simple path of length 4.
# Adding (u,v) closes it to a C6.
# So the check is:
# For each x in N(u) \ {v}:
#   For each z in N(v) \ {u, x}:
#     If N(x) ∩ N(z) contains any y such that y not in {u, v, x, z}:
#        Then adding (u,v) creates a C6.

def _creates_c6(adj, u, v):
    nu = adj[u]
    nv = adj[v]
    
    # x in N(u), x != v
    x_mask = nu & ~(1 << v)
    while x_mask:
        lsb_x = x_mask & -x_mask
        x = lsb_x.bit_length() - 1
        x_mask ^= lsb_x
        
        # z in N(v), z not in {u, x}
        z_mask = nv & ~(1 << u) & ~(1 << x)
        while z_mask:
            lsb_z = z_mask & -z_mask
            z = lsb_z.bit_length() - 1
            z_mask ^= lsb_z
            
            # y in N(x) ∩ N(z)
            # y must not be u, v, x, z
            forbidden = (1 << u) | (1 << v) | (1 << x) | (1 << z)
            y_mask = adj[x] & adj[z] & ~forbidden
            if y_mask:
                return True
    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _is_edge(adj, u, v):
    return (adj[u] >> v) & 1

def _seed_adjacency():
    """
    Create a dense seed graph.
    We want a graph that is close to C6-free but dense.
    Let's try a random graph with p=0.5, then remove edges that create C6s.
    Or better: Start with a known structure.
    The Turan graph T(40, 2) has 100 edges and is C6-free (actually C3-free).
    But we can do better.
    Let's start with a random graph and repair.
    """
    adj = [0] * N
    rng = random.Random(42)
    
    # Start with a fairly dense random graph
    # p = 0.6
    for u, v in ALL_EDGES:
        if rng.random() < 0.6:
            _add_edge(adj, u, v)
            
    # Repair: Remove edges that create C6s
    # We iterate until no more C6s are found.
    # To be efficient, we can check each edge.
    # If removing an edge doesn't help, we keep it.
    # Actually, we should remove edges that are part of a C6.
    # Let's just try to remove random edges until C6-free.
    
    # Better: Greedy removal.
    # Check all edges. If an edge is in a C6, remove it.
    # This might be too aggressive.
    
    # Alternative: Start with T(40,2) and add edges greedily.
    # T(40,2) is bipartite with parts of size 20.
    # Edges: (i, 20+i) for i in 0..19? No, complete bipartite K_{20,20}.
    # Let's construct K_{20,20}.
    adj = [0] * N
    for i in range(20):
        for j in range(20, 40):
            _add_edge(adj, i, j)
            
    # Now greedily add edges from the remaining possible edges.
    # Remaining edges are within {0..19} and within {20..39}.
    # We can also add edges between the two sets? No, K_{20,20} already has all.
    # So we only have edges within the two cliques to add? No, within the sets.
    # Let's shuffle all non-edges and try to add them.
    
    non_edges = []
    for u, v in ALL_EDGES:
        if not _is_edge(adj, u, v):
            non_edges.append((u, v))
            
    rng.shuffle(non_edges)
    
    for u, v in non_edges:
        if not _creates_c6(adj, u, v):
            _add_edge(adj, u, v)
            
    return adj

def _saturation_pass(adj, deadline):
    """
    Try to add any valid edge. Repeat until no more can be added.
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u, v in ALL_EDGES:
            if time.time() >= deadline:
                return adj
            if _is_edge(adj, u, v):
                continue
            if not _creates_c6(adj, u, v):
                _add_edge(adj, u, v)
                improved = True
    return adj

def _simulated_annealing(adj, deadline, rng):
    """
    Simulated annealing on the edge set.
    Moves: Add an edge, or Remove an edge.
    If Add creates C6, reject (or accept with low prob? No, constraint is hard).
    Actually, we must maintain C6-free invariant.
    So:
    1. Try to add a random non-edge. If valid, accept.
    2. Try to remove a random edge. Always accept (to escape local optima).
       But removing reduces edge count. We accept with probability exp(-delta/T).
       Here delta = -1 (loss of 1 edge). So prob = exp(1/T).
       Wait, standard SA: accept if new_energy < old_energy.
       Here, objective is maximize edges. So minimize -edges.
       New state has 1 fewer edge. Delta_E = +1 (worse).
       Accept with prob exp(-1/T).
       
    Strategy:
    - Start with T = 1.0
    - Cooldown: T *= 0.99
    - Min T = 0.01
    
    We need to balance adding and removing.
    If we only add, we get stuck.
    If we remove too much, we lose edges.
    
    Let's try:
    - With prob 0.5, try to add a random non-edge.
    - With prob 0.5, try to remove a random edge.
    
    For removal, we accept if rand() < exp(-1/T).
    """
    T = 1.0
    min_T = 0.01
    
    # Get current edge list for quick removal selection
    # We can just pick random u, v and check if edge exists.
    
    while time.time() < deadline and T > min_T:
        if rng.random() < 0.5:
            # Try add
            # Pick random non-edge
            # Efficient way: pick random u, v. If edge exists, retry?
            # Or maintain list of non-edges? List changes.
            # Let's just pick random u, v.
            u = rng.randint(0, N-1)
            v = rng.randint(0, N-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if _is_edge(adj, u, v):
                continue
            if not _creates_c6(adj, u, v):
                _add_edge(adj, u, v)
                # Immediate improvement, no need to check T
        else:
            # Try remove
            u = rng.randint(0, N-1)
            v = rng.randint(0, N-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if not _is_edge(adj, u, v):
                continue
            
            # Accept removal with prob exp(-1/T)
            if rng.random() < (1.0 / (1.0 + T)): # Approximation of exp(-1/T) for small T?
                # Actually exp(-1/T) drops fast.
                # Let's use standard: p = exp(-1/T)
                # If T=1, p=0.36. If T=0.1, p=0.006.
                # This might be too conservative.
                # Let's use p = 1/(1+T) as a smoother schedule?
                # Or just exp(-1/T).
                pass # Logic below
                
            # Let's stick to exp(-1/T)
            p_accept = 2.718281828 ** (-1.0 / T)
            if rng.random() < p_accept:
                _remove_edge(adj, u, v)
                
        T *= 0.995 # Slow cooling
        
    # Final saturation
    _saturation_pass(adj, deadline)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = random.Random(12345)
    
    best_adj = None
    best_count = 0
    
    # Multiple restarts
    # Each restart: Seed -> SA -> Saturation
    
    # We have ~110s.
    # SA is stochastic.
    # Let's run SA for a fixed amount of time per restart, or until convergence.
    # Since we don't know convergence, let's do time-sliced restarts.
    
    # Strategy:
    # 1. Generate seed.
    # 2. Run SA for (deadline - current_time - 5s) / remaining_restarts ?
    # No, just run SA until time is low, then saturation.
    # But we want multiple restarts to find different basins.
    
    # Let's do 5 restarts.
    num_restarts = 5
    time_per_restart = (DEADLINE - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() >= deadline - 5:
            break
            
        # 1. Seed
        adj = _seed_adjacency()
        
        # 2. Initial saturation
        _saturation_pass(adj, deadline)
        
        # 3. Simulated Annealing
        # Allocate time for SA
        sa_deadline = time.time() + time_per_restart * 0.7
        if sa_deadline > deadline - 10:
            sa_deadline = deadline - 10
            
        if time.time() < sa_deadline:
            adj = _simulated_annealing(adj, sa_deadline, rng)
            
        # 4. Final saturation
        adj = _saturation_pass(adj, deadline)
        
        # Evaluate
        count = _count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        # Update time_per_restart for remaining restarts?
        # Keep it simple.
        
    # If we still have time, do a final polish on the best solution
    if best_adj is not None and time.time() < deadline - 2:
        # Try to add any edges
        _saturation_pass(best_adj, deadline)
        best_count = _count_edges(best_adj)
        
    # Reconstruct edge list
    if best_adj is None:
        return []
        
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges

if __name__ == "__main__":
    construct()
