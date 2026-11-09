"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_2factor(adj, used_edges, n, deadline):
    """Find a 2-regular spanning subgraph (2-factor) avoiding used edges.
    Returns list of cycles (each a list of vertices) or None."""
    # Start with a random 2-factor: for each vertex, pick 2 neighbors
    # Use a greedy + repair approach
    for attempt in range(200):
        if time.time() > deadline:
            return None
        random.seed(attempt * 31 + 7)
        
        # Build available adjacency
        avail = {}
        for v in range(n):
            avail[v] = [nb for nb in adj[v] if _edge_key(v, nb) not in used_edges]
            if len(avail[v]) < 2:
                break
        else:
            # Try to construct a 2-factor via random assignment
            # Each vertex needs exactly 2 edges
            # Use a simple iterative approach
            degree = [0] * n
            selected = [set() for _ in range(n)]
            
            # Randomly order vertices and assign edges
            order = list(range(n))
            random.shuffle(order)
            
            success = True
            for v in order:
                # Pick 2 neighbors that still need connections
                candidates = []
                for nb in avail[v]:
                    if nb not in selected[v]:
                        candidates.append(nb)
                if len(candidates) < 2:
                    success = False
                    break
                # Prefer neighbors with higher degree deficit
                candidates.sort(key=lambda nb: (degree[nb], random.random()))
                chosen = candidates[:2]
                for nb in chosen:
                    selected[v].add(nb)
                    selected[nb].add(v)
                    degree[v] += 1
                    degree[nb] += 1
            
            if not success:
                continue
            
            # Check if it's a valid 2-factor
            valid = True
            for v in range(n):
                if degree[v] != 2:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check symmetry
            valid = True
            for v in range(n):
                for nb in selected[v]:
                    if v not in selected[nb]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Extract cycles
            visited = set()
            cycles = []
            for v in range(n):
                if v not in visited:
                    cycle = [v]
                    visited.add(v)
                    prev = v
                    cur = list(selected[v])[0]
                    while cur != v:
                        cycle.append(cur)
                        visited.add(cur)
                        prev, cur = cur, [x for x in selected[cur] if x != prev][0]
                    cycles.append(cycle)
            
            # Check if it's a single Hamiltonian cycle
            if len(cycles) == 1 and len(cycles[0]) == n:
                return cycles[0]
            # Otherwise, try to merge cycles via 2-opt
            # For now, only accept single cycles
            continue
    
    return None

def _merge_cycles_2opt(cycles, adj, used_edges, n, deadline):
    """Try to merge multiple cycles into one Hamiltonian cycle using 2-opt."""
    if len(cycles) <= 1:
        return cycles[0] if cycles else None
    
    # Get all edges in the 2-factor
    all_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            all_edges.add(_edge_key(u, v))
    
    # Try 2-opt: remove two edges from different cycles, add two new edges
    # to merge them
    best = cycles[:]
    
    for iteration in range(1000):
        if time.time() > deadline:
            break
        if len(best) == 1:
            break
        
        # Pick two cycles to merge
        if len(best) < 2:
            break
        i, j = random.sample(range(len(best)), 2)
        c1, c2 = best[i], best[j]
        
        # Try all pairs of edges (one from each cycle)
        merged = None
        for a in range(len(c1)):
            u1, v1 = c1[a], c1[(a + 1) % len(c1)]
            for b in range(len(c2)):
                u2, v2 = c2[b], c2[(b + 1) % len(c2)]
                # Try swapping: remove (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2)
                ek1 = _edge_key(u1, v1)
                ek2 = _edge_key(u2, v2)
                ek3 = _edge_key(u1, u2)
                ek4 = _edge_key(v1, v2)
                
                if ek3 not in adj[u1] and ek4 not in adj[v1]:
                    continue
                if ek3 in used_edges or ek4 in used_edges:
                    continue
                if ek3 == ek1 or ek3 == ek2 or ek4 == ek1 or ek4 == ek2:
                    continue
                
                # Check if this creates a single cycle
                # Build the merged cycle
                # c1: ... u1 - v1 ...  c2: ... u2 - v2 ...
                # New: ... u1 - u2 ... v2 - v1 ...
                new_c1 = c1[:a] + [v1] + c1[a+1:]  # This is getting complex
                # Simpler: just check if edges exist and not used
                if ek3 not in used_edges and ek4 not in used_edges:
                    # Verify the edges are in the original graph
                    if u2 in adj[u1] and v2 in adj[v1]:
                        # Try to form merged cycle
                        # c1 without edge (u1,v1): u1, ..., v1 (path)
                        # c2 without edge (u2,v2): u2, ..., v2 (path)
                        # Merge: u1 -> u2 (path in c2 reversed) -> v2 -> v1 (path in c1)
                        # Actually: u1 to u2, then follow c2 from u2 to v2, then v2 to v1, then follow c1 from v1 to u1
                        path1 = c1[a:] + c1[:a+1]  # u1, ..., v1
                        path2 = c2[b:] + c2[:b+1]  # u2, ..., v2
                        
                        # Merge: start at u1, go to u2, follow path2 from u2 to v2, go to v1, follow path1 from v1 to u1
                        # path1 starts at u1, ends at v1
                        # path2 starts at u2, ends at v2
                        # New cycle: u1, u2, ..., v2, v1, ..., u1
                        merged_candidate = [u1] + path2[1:] + [v1] + path1[1:-1]
                        
                        if len(merged_candidate) == len(c1) + len(c2):
                            # Check all edges valid
                            valid = True
                            for k in range(len(merged_candidate)):
                                ek = _edge_key(merged_candidate[k], merged_candidate[(k+1) % len(merged_candidate)])
                                if ek in used_edges:
                                    valid = False
                                    break
                            if valid:
                                merged = merged_candidate
                                break
            if merged:
                best = best[:i] + best[i+1:j] + best[j+1:] + [merged]
                # Re-sort for consistent indexing
                best.sort(key=len)
                break
        if merged is None:
            # Try random perturbation
            if random.random() < 0.3:
                continue
    
    if len(best) == 1:
        return best[0]
    return None

def _find_hamiltonian_sa(adj, used_edges, n, deadline):
    """Simulated annealing to find a Hamiltonian cycle."""
    best_cycle = None
    best_score = -1
    
    # Start with a random permutation
    perm = list(range(n))
    random.shuffle(perm)
    
    def score(p):
        """Count how many consecutive pairs are valid edges (not used)."""
        s = 0
        for i in range(n):
            u = p[i]
            v = p[(i + 1) % n]
            if _edge_key(u, v) not in used_edges and v in adj[u]:
                s += 1
        return s
    
    current_score = score(perm)
    T = 10.0
    cooling = 0.995
    
    steps = 0
    while T > 0.01 and time.time() < deadline:
        steps += 1
        # 2-opt move: reverse a segment
        i, j = sorted(random.sample(range(n), 2))
        new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
        new_score = score(new_perm)
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < pow(2.718281828, delta / T):
            perm = new_perm
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_cycle = list(perm)
                if best_score == n:
                    break
        T *= cooling
    
    if best_score == n:
        return best_cycle
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = P['n']
    adj = _build_adjacency(P['edges'])
    
    best_cycles = []
    
    for outer_attempt in range(50):
        if time.time() > deadline:
            break
        
        random.seed(outer_attempt * 104729 + 37)
        
        used_edges = set()
        cycles = []
        
        for cycle_idx in range(3):
            if time.time() > deadline:
                break
            
            # Try SA-based Hamiltonian cycle finding
            cycle = _find_hamiltonian_sa(adj, used_edges, n, deadline)
            
            if cycle is None:
                break
            
            # Verify
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if _edge_key(u, v) in used_edges or v not in adj[u]:
                    valid = False
                    break
            if not valid:
                break
            
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(_edge_key(u, v))
            
            cycles.append([int(v) for v in cycle])
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                break
    
    # Final validation
    seen_edges = set()
    final_cycles = []
    for cycle in best_cycles:
        edges_in_cycle = set()
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            ek = _edge_key(u, v)
            if ek in seen_edges or ek in edges_in_cycle or v not in adj[u]:
                valid = False
                break
            edges_in_cycle.add(ek)
        if valid:
            seen_edges.update(edges_in_cycle)
            final_cycles.append(cycle)
    
    return final_cycles
# EVOLVE-BLOCK-END
