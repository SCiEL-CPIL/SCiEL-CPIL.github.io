"""Dense K2,3-free graph on 20 vertices via simulated annealing with bitmask representation."""
import time
import random

N = 20
DEADLINE_MARGIN = 5.0  # seconds to stay under the 150s wall-clock kill
TOTAL_EDGES = N * (N - 1) // 2  # 190

# Precompute edge indices
EDGE_LIST = [(u, v) for u in range(N) for v in range(u + 1, N)]
EDGE_INDEX = {(u, v): i for i, (u, v) in enumerate(EDGE_LIST)}
EDGE_INDEX.update({(v, u): i for i, (u, v) in enumerate(EDGE_LIST)})

def _common_neighbours_bitmask(adj_bitmasks, u, v):
    """Return count of common neighbours using bitmasks."""
    return bin(adj_bitmasks[u] & adj_bitmasks[v]).count('1')

def _is_valid_bitmask(adj_bitmasks):
    """Check no two vertices have 3+ common neighbours using bitmasks."""
    for u in range(N):
        for v in range(u + 1, N):
            if bin(adj_bitmasks[u] & adj_bitmasks[v]).count('1') >= 3:
                return False
    return True

def _count_edges_bitmask(adj_bitmasks):
    """Count edges from bitmask representation."""
    return sum(bin(adj_bitmasks[u]).count('1') for u in range(N)) // 2

def _bitmask_to_edge_list(adj_bitmasks):
    """Convert bitmask representation to edge list."""
    edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if adj_bitmasks[u] & (1 << v):
                edges.append([u, v])
    return edges

def _edge_list_to_bitmask(edge_list):
    """Convert edge list to bitmask representation."""
    adj_bitmasks = [0] * N
    for u, v in edge_list:
        adj_bitmasks[u] |= (1 << v)
        adj_bitmasks[v] |= (1 << u)
    return adj_bitmasks

def _can_add_edge_bitmask(adj_bitmasks, u, v):
    """Check if edge (u,v) can be added without violating K2,3 constraint."""
    if adj_bitmasks[u] & (1 << v):
        return True  # Already present
    
    # Check pair (u, v): common neighbours must be <= 2
    if bin(adj_bitmasks[u] & adj_bitmasks[v]).count('1') >= 3:
        return False
    
    # For each w in N(v), check |N(u) ∩ N(w)| <= 1
    for w in range(N):
        if w == u or w == v:
            continue
        if adj_bitmasks[v] & (1 << w):
            if bin(adj_bitmasks[u] & adj_bitmasks[w]).count('1') >= 2:
                return False
    
    # For each w in N(u), check |N(v) ∩ N(w)| <= 1
    for w in range(N):
        if w == u or w == v:
            continue
        if adj_bitmasks[u] & (1 << w):
            if bin(adj_bitmasks[v] & adj_bitmasks[w]).count('1') >= 2:
                return False
    
    return True

def _add_edge_bitmask(adj_bitmasks, u, v):
    """Add edge (u,v) to bitmask representation."""
    adj_bitmasks[u] |= (1 << v)
    adj_bitmasks[v] |= (1 << u)

def _remove_edge_bitmask(adj_bitmasks, u, v):
    """Remove edge (u,v) from bitmask representation."""
    adj_bitmasks[u] &= ~(1 << v)
    adj_bitmasks[v] &= ~(1 << u)

def _greedy_build_bitmask(seed):
    """Greedy construction using bitmask representation."""
    rng = random.Random(seed)
    adj_bitmasks = [0] * N
    
    edge_indices = list(range(TOTAL_EDGES))
    rng.shuffle(edge_indices)
    
    for idx in edge_indices:
        u, v = EDGE_LIST[idx]
        if _can_add_edge_bitmask(adj_bitmasks, u, v):
            _add_edge_bitmask(adj_bitmasks, u, v)
    
    return adj_bitmasks

def _simulated_annealing(adj_bitmasks, seed, time_limit):
    """Simulated annealing to maximize edges in K2,3-free graph."""
    rng = random.Random(seed)
    adj_bitmasks = [b for b in adj_bitmasks]  # Copy
    
    current_edges = _count_edges_bitmask(adj_bitmasks)
    best_edges = current_edges
    best_adj = [b for b in adj_bitmasks]
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.999
    
    T = T_start
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > T_end:
        # Pick a random edge to toggle
        idx = rng.randint(0, TOTAL_EDGES - 1)
        u, v = EDGE_LIST[idx]
        
        if adj_bitmasks[u] & (1 << v):
            # Remove edge
            _remove_edge_bitmask(adj_bitmasks, u, v)
            new_edges = current_edges - 1
            
            # Always accept removal (or accept with high probability)
            accept = True
        else:
            # Try to add edge
            if _can_add_edge_bitmask(adj_bitmasks, u, v):
                _add_edge_bitmask(adj_bitmasks, u, v)
                new_edges = current_edges + 1
                accept = True
            else:
                # Can't add due to constraint - reject
                T *= cooling_rate
                continue
        
        if accept:
            # Check if new state is valid (it should be, but verify)
            if not _is_valid_bitmask(adj_bitmasks):
                # Revert
                if adj_bitmasks[u] & (1 << v):
                    _remove_edge_bitmask(adj_bitmasks, u, v)
                else:
                    _add_edge_bitmask(adj_bitmasks, u, v)
                T *= cooling_rate
                continue
            
            current_edges = new_edges
            
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = [b for b in adj_bitmasks]
        
        T *= cooling_rate
    
    return best_adj, best_edges

def _local_search_bitmask(adj_bitmasks, seed, time_limit):
    """Local search: try to add edges, and if stuck, try swapping."""
    rng = random.Random(seed)
    adj_bitmasks = [b for b in adj_bitmasks]  # Copy
    
    current_edges = _count_edges_bitmask(adj_bitmasks)
    best_edges = current_edges
    best_adj = [b for b in adj_bitmasks]
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try to add random missing edges
        missing = [(u, v) for u, v in EDGE_LIST if not (adj_bitmasks[u] & (1 << v))]
        if missing:
            rng.shuffle(missing)
            for u, v in missing[:30]:
                if _can_add_edge_bitmask(adj_bitmasks, u, v):
                    _add_edge_bitmask(adj_bitmasks, u, v)
                    current_edges += 1
                    improved = True
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [b for b in adj_bitmasks]
                    break
        
        if not improved:
            # Try a swap: remove a random edge, then try to add edges
            present = [(u, v) for u, v in EDGE_LIST if (adj_bitmasks[u] & (1 << v))]
            if present:
                rng.shuffle(present)
                for u_rm, v_rm in present[:5]:
                    _remove_edge_bitmask(adj_bitmasks, u_rm, v_rm)
                    current_edges -= 1
                    
                    added = 0
                    missing_now = [(u, v) for u, v in EDGE_LIST if not (adj_bitmasks[u] & (1 << v))]
                    rng.shuffle(missing_now)
                    for u_add, v_add in missing_now[:20]:
                        if _can_add_edge_bitmask(adj_bitmasks, u_add, v_add):
                            _add_edge_bitmask(adj_bitmasks, u_add, v_add)
                            current_edges += 1
                            added += 1
                            if current_edges > best_edges:
                                best_edges = current_edges
                                best_adj = [b for b in adj_bitmasks]
                            if added >= 1:
                                break
                    
                    if added == 0:
                        # Revert
                        _add_edge_bitmask(adj_bitmasks, u_rm, v_rm)
                        current_edges += 1
    
    return best_adj, best_edges

def construct():
    global deadline
    deadline = time.time() + 150.0
    
    best_adj = None
    best_edges = 0
    
    # Phase 1: Multiple greedy builds
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111]
    
    for seed in seeds:
        if time.time() > deadline - DEADLINE_MARGIN:
            break
        adj = _greedy_build_bitmask(seed)
        edges = _count_edges_bitmask(adj)
        if edges > best_edges:
            best_edges = edges
            best_adj = adj
    
    if best_adj is None:
        # Fallback: 5 copies of K4
        adj_bitmasks = [0] * N
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    adj_bitmasks[u] |= (1 << v)
                    adj_bitmasks[v] |= (1 << u)
        best_adj = adj_bitmasks
        best_edges = _count_edges_bitmask(adj_bitmasks)
    
    # Phase 2: Local search on best
    remaining_time = deadline - DEADLINE_MARGIN - time.time()
    if remaining_time > 2.0:
        local_time = remaining_time * 0.4
        best_adj, best_edges = _local_search_bitmask(best_adj, 42, local_time)
    
    # Phase 3: Simulated annealing from best
    remaining_time = deadline - DEADLINE_MARGIN - time.time()
    if remaining_time > 2.0:
        sa_time = remaining_time * 0.6
        sa_adj, sa_edges = _simulated_annealing(best_adj, 123, sa_time)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj
    
    # Convert to edge list
    edges = _bitmask_to_edge_list(best_adj)
    
    return edges

# Pre-compute deadline for local search
deadline = 0
