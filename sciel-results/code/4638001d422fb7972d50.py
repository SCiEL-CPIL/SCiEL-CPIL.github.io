"""Q3-free graph on 16 vertices - maximum edge count via simulated annealing with fast Q3 detection."""
import random
import time

# Precompute Q3 structure
_Q3_EDGES = []
for _i in range(8):
    for _j in range(_i + 1, 8):
        if bin(_i ^ _j).count('1') == 1:
            _Q3_EDGES.append((_i, _j))

# Q3 adjacency for each Q3 vertex (0-7)
_Q3_ADJ = [set() for _ in range(8)]
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a].add(_b)
    _Q3_ADJ[_b].add(_a)

# Optimal vertex ordering for backtracking: start with high-degree constraints
# In Q3 all vertices have degree 3, so order doesn't matter much for degree.
# Use natural order 0-7.

def _find_q3(adj_bits, n=16):
    """
    Check if graph (adjacency as list of bitmasks) contains Q3 subgraph.
    adj_bits[i] is a bitmask of neighbors of vertex i.
    Returns True if Q3 found.
    """
    assignment = [-1] * 8
    used_mask = 0
    
    def backtrack(idx):
        nonlocal used_mask
        if idx == 8:
            return True
        qv = idx
        # Build candidate bitmask: unused vertices adjacent to all assigned Q3-neighbors of qv
        # Start with all unused vertices
        candidates = ((1 << n) - 1) & ~used_mask
        # Intersect with neighbors of each already-assigned Q3-neighbor of qv
        for qn in _Q3_ADJ[qv]:
            if assignment[qn] != -1:
                candidates &= adj_bits[assignment[qn]]
        if candidates == 0:
            return False
        
        while candidates:
            lb = candidates & (-candidates)
            gv = lb.bit_length() - 1
            candidates ^= lb
            
            assignment[qv] = gv
            old_used = used_mask
            used_mask |= lb
            
            if backtrack(idx + 1):
                return True
            
            used_mask = old_used
            assignment[qv] = -1
        
        return False
    
    for gv in range(n):
        assignment[0] = gv
        old_used = used_mask
        used_mask = (1 << gv)
        if backtrack(1):
            used_mask = old_used
            return True
        assignment[0] = -1
        used_mask = old_used
    
    return False

def _edges_to_bits(edges, n=16):
    """Convert edge list to list of adjacency bitmasks."""
    bits = [0] * n
    for u, v in edges:
        bits[u] |= (1 << v)
        bits[v] |= (1 << u)
    return bits

def _bits_to_edges(bits, n=16):
    """Convert adjacency bitmasks to sorted edge list."""
    edges = []
    for i in range(n):
        mask = bits[i]
        while mask:
            lb = mask & (-mask)
            j = lb.bit_length() - 1
            mask ^= lb
            if i < j:
                edges.append([i, j])
    return edges

def _count_edges(bits, n=16):
    total = 0
    for i in range(n):
        total += bin(bits[i]).count('1')
    return total // 2

def _add_edge_creates_q3(bits, u, v, n=16):
    """Check if adding edge (u,v) to graph with adjacency bits creates a Q3."""
    # Temporarily add edge
    bits[u] |= (1 << v)
    bits[v] |= (1 << u)
    result = _find_q3(bits, n)
    # Revert
    bits[u] &= ~(1 << v)
    bits[v] &= ~(1 << u)
    return result

def _remove_edge_breaks_q3(bits, u, v, n=16):
    """Check if removing edge (u,v) from graph with adjacency bits breaks all Q3s (i.e., graph becomes Q3-free)."""
    # Temporarily remove edge
    bits[u] &= ~(1 << v)
    bits[v] &= ~(1 << u)
    result = not _find_q3(bits, n)
    # Revert
    bits[u] |= (1 << v)
    bits[v] |= (1 << u)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_bits = None
    best_count = 0
    
    # Multiple random restarts with simulated annealing
    rng = random.Random(42)
    
    num_restarts = 20
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        restarts_done += 1
        
        # Initialize: start with a random dense graph
        # Start with all edges, then remove edges to make Q3-free
        bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.7:  # Start fairly dense
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
        
        # Remove edges to make Q3-free
        changed = True
        while changed:
            if time.time() > deadline - 10.0:
                break
            changed = False
            if _find_q3(bits, n):
                # Find a random edge to remove that helps
                # Try random edges
                for _ in range(50):
                    if time.time() > deadline - 10.0:
                        break
                    u, v = rng.choice(all_edges)
                    if bits[u] & (1 << v):
                        bits[u] &= ~(1 << v)
                        bits[v] &= ~(1 << u)
                        if not _find_q3(bits, n):
                            changed = True
                            break
                        else:
                            # Revert
                            bits[u] |= (1 << v)
                            bits[v] |= (1 << u)
                if not changed:
                    # Forced to remove more aggressively
                    for u, v in all_edges:
                        if bits[u] & (1 << v):
                            bits[u] &= ~(1 << v)
                            bits[v] &= ~(1 << u)
                            if not _find_q3(bits, n):
                                changed = True
                                break
                    if not changed:
                        break
        
        if _find_q3(bits, n):
            continue
        
        cur_count = _count_edges(bits, n)
        
        # Simulated annealing phase
        temp = 2.0
        cooling = 0.999
        min_temp = 0.01
        step_limit = 50000
        
        for step in range(step_limit):
            if time.time() > deadline - 10.0:
                break
            if temp < min_temp:
                break
            
            # Choose a random move: add or remove an edge
            if rng.random() < 0.6:  # Bias toward adding edges
                # Try to add a random missing edge
                u, v = rng.choice(all_edges)
                if not (bits[u] & (1 << v)):
                    # Check if adding creates Q3
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
                    has_q3 = _find_q3(bits, n)
                    if has_q3:
                        # Revert
                        bits[u] &= ~(1 << v)
                        bits[v] &= ~(1 << u)
                        # Don't accept
                    else:
                        cur_count += 1
                        if cur_count > best_count:
                            best_count = cur_count
                            best_bits = list(bits)
                        # Always accept valid additions
            else:
                # Try to remove a random existing edge
                u, v = rng.choice(all_edges)
                if bits[u] & (1 << v):
                    bits[u] &= ~(1 << v)
                    bits[v] &= ~(1 << u)
                    cur_count -= 1
                    # Check if this breaks a Q3 (i.e., makes graph Q3-free if it was Q3-containing)
                    # Actually we should maintain Q3-free invariant, so only remove if needed
                    # For SA, we allow temporary violations
                    has_q3 = _find_q3(bits, n)
                    delta = 0 if not has_q3 else -10  # Penalty for Q3
                    # Accept with probability
                    if has_q3 or rng.random() < 0.3:
                        # Keep the removal (even if it reduces edge count)
                        if not has_q3:
                            if cur_count > best_count:
                                best_count = cur_count
                                best_bits = list(bits)
                    else:
                        # Revert
                        bits[u] |= (1 << v)
                        bits[v] |= (1 << u)
                        cur_count += 1
            
            temp *= cooling
        
        # End of restart: update best
        if not _find_q3(bits, n):
            cur_count = _count_edges(bits, n)
            if cur_count > best_count:
                best_count = cur_count
                best_bits = list(bits)
    
    if best_bits is None:
        # Fallback: K3,13
        bits = [0] * n
        for a in range(3):
            for b in range(3, n):
                bits[a] |= (1 << b)
                bits[b] |= (1 << a)
        best_bits = bits
    
    return _bits_to_edges(best_bits, n)
