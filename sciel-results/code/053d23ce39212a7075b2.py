"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edge_list):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edge_list:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _find_ham_cycle_bt(adj, start, deadline, rng):
    """Fast backtracking with Warnsdorff heuristic for Hamiltonian cycle."""
    n = P['n']
    adj_list = [list(adj[i]) for i in range(n)]
    
    def backtrack(path, visited, current):
        if time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj[current]:
                return path[:]
            return None
        
        remaining = n - len(path)
        # Get unvisited neighbors
        neighbors = [v for v in adj_list[current] if v not in visited]
        
        if not neighbors:
            return None
        
        # Warnsdorff: sort by number of onward moves (ascending)
        # For the last step, check if it can connect back to start
        if remaining == 1:
            # Only one more vertex to place; it must connect back to start
            for v in neighbors:
                if start in adj[v]:
                    return path + [v]
            return None
        
        # Sort by onward degree (unvisited neighbors count)
        neighbors.sort(key=lambda v: sum(1 for x in adj_list[v] if x not in visited))
        
        # Pruning: check that no unvisited vertex is isolated
        # (cheap check: each unvisited vertex must have at least 1 unvisited neighbor or be 'current')
        for v in neighbors:
            visited.add(v)
            path.append(v)
            result = backtrack(path, visited, v)
            if result is not None:
                return result
            path.pop()
            visited.remove(v)
        
        return None
    
    visited = {start}
    path = [start]
    return backtrack(path, visited, start)

def _find_ham_cycle_sa(adj, start, deadline, rng):
    """Simulated annealing with incremental cost for Hamiltonian cycle."""
    n = P['n']
    vertices = list(range(n))
    adj_set = [set(adj[i]) for i in range(n)]
    
    def count_invalid_edges(p):
        count = 0
        for i in range(len(p)):
            u = p[i]
            v = p[(i + 1) % len(p)]
            if v not in adj_set[u]:
                count += 1
        return count
    
    # Initialize with a heuristic: greedily build path
    rest = [v for v in vertices if v != start]
    rng.shuffle(rest)
    current = [start] + rest
    current_cost = count_invalid_edges(current)
    best = current[:]
    best_cost = current_cost
    
    T = 4.0
    cooling = 0.9997
    max_iter = 200000
    stuck_count = 0
    STUCK_THRESHOLD = 3000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if current_cost == 0:
            break
        
        # Pick move type: 2-opt (60%) or swap (40%)
        r = rng.random()
        if r < 0.6:
            # 2-opt: reverse segment
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            # Compute cost change for reversing segment i..j
            # Edges affected: (i-1,i), (j,j+1) removed; (i-1,j), (i,j+1) added
            # Plus internal edges are reversed but still valid (undirected)
            old_e1 = 1 if current[i-1] in adj_set[current[i]] else 0
            old_e2 = 1 if current[j] in adj_set[current[(j+1) % n]] else 0
            new_e1 = 1 if current[i-1] in adj_set[current[j]] else 0
            new_e2 = 1 if current[i] in adj_set[current[(j+1) % n]] else 0
            delta = (new_e1 + new_e2) - (old_e1 + old_e2)
            # Note: negative delta means fewer invalid edges (good)
            # Actually we want to minimize invalid edges, so delta < 0 is improvement
            if delta >= 0 and rng.random() > pow(2.718281828, -delta / max(T, 1e-10)):
                T *= cooling
                continue
            # Apply 2-opt
            current[i:j+1] = current[i:j+1][::-1]
            current_cost += delta
        else:
            # Swap two positions
            i = rng.randint(1, n - 1)
            j = rng.randint(1, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            # Compute cost change
            # Affected edges: (i-1,i), (i,i+1), (j-1,j), (j,j+1)
            # But careful with adjacency
            a, b, c, d = current[i-1], current[i], current[j], current[(j+1) % n]
            old1 = 1 if a in adj_set[b] else 0
            old2 = 1 if b in adj_set[current[(i+1) % n]] else 0
            old3 = 1 if current[j-1] in adj_set[c] else 0
            old4 = 1 if c in adj_set[d] else 0
            new1 = 1 if a in adj_set[c] else 0
            new2 = 1 if c in adj_set[current[(i+1) % n]] else 0
            new3 = 1 if current[j-1] in adj_set[b] else 0
            new4 = 1 if b in adj_set[d] else 0
            delta = (new1 + new2 + new3 + new4) - (old1 + old2 + old3 + old4)
            if delta >= 0 and rng.random() > pow(2.718281828, -delta / max(T, 1e-10)):
                T *= cooling
                continue
            # Apply swap
            current[i], current[j] = current[j], current[i]
            current_cost += delta
        
        stuck_count = 0
        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost
        else:
            stuck_count += 1
        
        if stuck_count > STUCK_THRESHOLD:
            stuck_count = 0
            rest = [v for v in vertices if v != start]
            rng.shuffle(rest)
            current = [start] + rest
            current_cost = count_invalid_edges(current)
            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost
        
        T *= cooling
    
    if best_cost == 0:
        return best
    return None

def construct():
    deadline = time.time() + 55
    rng = random.Random(12345)
    
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add(_edge_key(u, v))
    
    best_cycles = []
    
    # Strategy: try to find up to 3 edge-disjoint Hamiltonian cycles
    # Use multiple restarts with different seeds and start vertices
    
    max_outer = 200
    for outer in range(max_outer):
        if time.time() > deadline:
            break
        if len(best_cycles) >= 3:
            break
        
        # For each attempt, try to build up to 3 cycles
        remaining = set(base_edge_set)
        cycles = []
        
        # Try different orderings of start vertices
        starts = list(range(n))
        rng.shuffle(starts)
        
        success = True
        for cycle_idx in range(3):
            if time.time() > deadline:
                success = False
                break
            
            edge_list = list(remaining)
            adj = _build_adj(edge_list)
            
            # Check if graph is connected and all degrees >= 2
            min_deg = min(len(adj[i]) for i in range(n))
            if min_deg < 2:
                success = False
                break
            
            # Try multiple start vertices
            cycle_found = None
            for start in starts:
                if time.time() > deadline:
                    break
                if len(adj[start]) < 2:
                    continue
                
                # Try backtracking first (fast with Warnsdorff)
                cycle_found = _find_ham_cycle_bt(adj, start, deadline, rng)
                if cycle_found is not None:
                    break
                
                # Try simulated annealing
                cycle_found = _find_ham_cycle_sa(adj, start, deadline, rng)
                if cycle_found is not None:
                    break
            
            if cycle_found is None:
                success = False
                break
            
            # Verify and extract edges
            edges_used = set()
            valid = True
            for i in range(len(cycle_found)):
                u = cycle_found[i]
                v = cycle_found[(i + 1) % len(cycle_found)]
                key = _edge_key(u, v)
                if key not in remaining:
                    valid = False
                    break
                edges_used.add(key)
            
            if not valid:
                success = False
                break
            
            cycles.append(cycle_found)
            remaining -= edges_used
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles[:]
            if len(best_cycles) >= 3:
                break
    
    # Fallback
    if not best_cycles:
        adj = _build_adj(base_edges)
        cycle = _find_ham_cycle_bt(adj, 0, deadline, rng)
        if cycle is None:
            cycle = _find_ham_cycle_sa(adj, 0, deadline, rng)
        if cycle is not None:
            best_cycles = [cycle]
        else:
            best_cycles = [list(range(n))]
    
    result = [[int(v) for v in cycle] for cycle in best_cycles]
    return result
# EVOLVE-BLOCK-END
