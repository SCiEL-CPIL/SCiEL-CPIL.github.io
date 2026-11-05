import random
import time
import math

def construct():
    n = 16
    possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_edges_set = set(possible_edges)
    
    def is_3_colorable(adj):
        # Order vertices by degree for better pruning
        order = sorted(range(n), key=lambda x: bin(adj[x]).count('1'), reverse=True)
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            used = 0
            nb = adj[v]
            while nb:
                lsb = nb & -nb
                u = lsb.bit_length() - 1
                if colors[u] != -1:
                    used |= (1 << colors[u])
                nb ^= lsb
            
            for c in range(3):
                if not (used & (1 << c)):
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
        
        return backtrack(0)

    # Initialize with Clebsch Graph
    # Vertices 0..15, edges if XOR in {1, 2, 4, 8, 15}
    clebsch_edges = []
    adj = [0] * n
    for u in range(n):
        for d in [1, 2, 4, 8, 15]:
            v = u ^ d
            if u < v:
                clebsch_edges.append([u, v])
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    current_edges = set(tuple(e) for e in clebsch_edges)
    current_m = len(current_edges)
    current_colorable = is_3_colorable(adj)
    
    # Energy: -m if not colorable, -m + PENALTY if colorable
    PENALTY = 10.0
    
    def get_energy(m, colorable):
        if not colorable:
            return -m
        else:
            return -m + PENALTY

    current_energy = get_energy(current_m, current_colorable)
    
    best_edges = clebsch_edges[:]
    best_m = current_m
    best_colorable = current_colorable
    
    # If Clebsch is not 3-colorable, it's a valid start
    if not current_colorable:
        best_m = current_m
        best_edges = clebsch_edges[:]
    else:
        # Should not happen, but just in case
        best_m = -1
        best_edges = []

    start_time = time.time()
    rng = random.Random(42)
    
    T = 5.0
    cooling = 0.999
    
    while time.time() - start_time < 100:
        # Try a move
        if rng.random() < 0.5:
            # Add edge
            # Pick random non-edge
            # To speed up, pick random u, v and check
            # Or maintain list of non-edges? 
            # Simple: pick random u, v from possible_edges
            u, v = rng.choice(possible_edges)
            if (u, v) in current_edges:
                continue
            
            # Check triangle
            if adj[u] & adj[v]:
                continue
            
            # Tentative add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.add((u, v))
            new_m = current_m + 1
            new_colorable = is_3_colorable(adj)
            new_energy = get_energy(new_m, new_colorable)
            
            delta_e = new_energy - current_energy
            
            accept = False
            if delta_e < 0:
                accept = True
            else:
                if T > 1e-6:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                current_m = new_m
                current_colorable = new_colorable
                current_energy = new_energy
                
                if not current_colorable and current_m > best_m:
                    best_m = current_m
                    best_edges = [[u, v] for u, v in current_edges]
            else:
                # Reject: revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.discard((u, v))
                
        else:
            # Remove edge
            if not current_edges:
                continue
            u, v = rng.choice(list(current_edges))
            
            # Tentative remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.discard((u, v))
            new_m = current_m - 1
            new_colorable = is_3_colorable(adj)
            new_energy = get_energy(new_m, new_colorable)
            
            delta_e = new_energy - current_energy
            
            accept = False
            if delta_e < 0:
                accept = True
            else:
                if T > 1e-6:
                    prob = math.exp(-delta_e / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                current_m = new_m
                current_colorable = new_colorable
                current_energy = new_energy
            else:
                # Reject: revert
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
                
        T *= cooling

    # Final check: if we didn't find a better one, return best found
    # Ensure best_edges is a list of lists
    if best_m == -1:
        # Fallback to Clebsch
        return clebsch_edges
        
    return best_edges
