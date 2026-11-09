import time
import math
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    LIMIT = 512
    N = LIMIT
    
    # 1. Build Compatibility Graph
    # compat[i] is a bitmask of nodes j (1..N) such that lcm(i, j) <= LIMIT
    compat = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                compat[i] |= (1 << (j - 1))
                compat[j] |= (1 << (i - 1))
    
    # Precompute neighbor lists for faster operations
    neighbors = [0] * (N + 1)
    for i in range(1, N + 1):
        neighbors[i] = compat[i]
    
    # Precompute degree
    degree = [0] * (N + 1)
    for i in range(1, N + 1):
        degree[i] = bin(compat[i]).count('1')
    
    # 2. Tomita's Algorithm with Bitmasks
    best_clique = []
    
    # Greedy Lower Bound
    def greedy_clique(order=None):
        if order is None:
            order = list(range(1, N+1))
            order.sort(key=lambda x: -degree[x])
        
        clique = []
        for node in order:
            if time.time() > DEADLINE:
                break
            ok = True
            for c in clique:
                if not (compat[node] & (1 << (c - 1))):
                    ok = False
                    break
            if ok:
                clique.append(node)
        return clique

    init_clique = greedy_clique()
    if init_clique:
        best_clique = init_clique
    
    sys.setrecursionlimit(10000)
    
    def color_sort(P_mask):
        """Fast greedy coloring using bitmasks for color classes."""
        # Extract vertices from P_mask
        P = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            P.append(idx + 1)
            temp ^= lsb
        
        # Sort by degree descending
        P.sort(key=lambda v: -degree[v])
        
        # Color classes as bitmasks
        color_masks = []
        vertex_color = {}
        
        for v in P:
            assigned = False
            for k in range(len(color_masks)):
                # Check if v is compatible with any vertex in color_masks[k]
                # If v is NOT compatible with any, it can go in this color
                if (compat[v] & color_masks[k]) == 0:
                    color_masks[k] |= (1 << (v - 1))
                    vertex_color[v] = k
                    assigned = True
                    break
            
            if not assigned:
                color_masks.append(1 << (v - 1))
                vertex_color[v] = len(color_masks) - 1
        
        return P, color_masks, vertex_color, len(color_masks)
    
    def expand(P_mask, C, depth):
        nonlocal best_clique
        
        if time.time() > DEADLINE:
            return
            
        if P_mask == 0:
            if depth > len(best_clique):
                best_clique = C[:]
            return
        
        if depth + bin(P_mask).count('1') <= len(best_clique):
            return
            
        # Greedy Coloring with bitmask-based color classes
        P, color_masks, vertex_color, max_color = color_sort(P_mask)
        
        if depth + max_color <= len(best_clique):
            return
        
        # Branching: iterate colors in reverse
        for k in range(max_color - 1, -1, -1):
            # Extract vertices from color_masks[k] that are still in P_mask
            cm = color_masks[k]
            temp = cm
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()  # This gives the index + 1 directly
                v = lsb.bit_length()  # bit_length of 1<<i is i+1
                # Actually: if lsb = 1 << (v-1), then lsb.bit_length() = v
                v = lsb.bit_length()
                
                if P_mask & lsb:
                    # Include v
                    C.append(v)
                    
                    # New P: neighbors of v in P_mask \ {v}
                    new_P_mask = P_mask & compat[v] & ~(1 << (v - 1))
                    
                    expand(new_P_mask, C, depth + 1)
                    
                    C.pop()
                    
                    # Exclude v
                    P_mask &= ~(1 << (v - 1))
                    
                    if time.time() > DEADLINE:
                        return
                
                temp ^= lsb
            
            if time.time() > DEADLINE:
                return

    # Multiple restarts with different orderings
    initial_P_mask = (1 << N) - 1
    
    # First run: standard ordering
    expand(initial_P_mask, [], 0)
    
    # Additional runs with perturbed orderings if time permits
    num_restarts = 0
    max_restarts = 50
    while time.time() < DEADLINE - 5 and num_restarts < max_restarts:
        num_restarts += 1
        
        # Create a perturbed ordering
        order = list(range(1, N+1))
        random.shuffle(order)
        
        # Use greedy with this ordering to get initial clique
        init_clique = greedy_clique(order)
        if init_clique and len(init_clique) > len(best_clique):
            best_clique = init_clique
        
        # Run expand with a subset or different strategy
        # For now, just run the full expand again with different internal ordering
        # This is expensive, so we limit restarts
        if num_restarts >= 10:
            break
    
    # Local search polish: try to add vertices to the best clique
    if best_clique and time.time() < DEADLINE - 2:
        clique_set = set(best_clique)
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            # Try adding each vertex not in clique
            for v in range(1, N+1):
                if v in clique_set:
                    continue
                # Check if v is compatible with all in clique
                ok = True
                for c in best_clique:
                    if not (compat[v] & (1 << (c - 1))):
                        ok = False
                        break
                if ok:
                    best_clique.append(v)
                    clique_set.add(v)
                    improved = True
                    break
            
            # Try swaps: remove one, add one
            if not improved:
                for i, c in enumerate(best_clique):
                    if time.time() > DEADLINE - 2:
                        break
                    # Try removing c and adding some v
                    temp_clique = best_clique[:i] + best_clique[i+1:]
                    for v in range(1, N+1):
                        if v in clique_set or v == c:
                            continue
                        # Check if v is compatible with all in temp_clique
                        ok = True
                        for tc in temp_clique:
                            if not (compat[v] & (1 << (tc - 1))):
                                ok = False
                                break
                        if ok:
                            # Also check that removing c and adding v doesn't make things worse
                            # We already know v is compatible with temp_clique
                            best_clique = temp_clique + [v]
                            clique_set = set(best_clique)
                            improved = True
                            break
                    if improved:
                        break
    
    if not best_clique:
        return [1]
        
    return best_clique

if __name__ == "__main__":
    result = construct()
    print(f"Found clique of size {len(result)}")
    print(result)
