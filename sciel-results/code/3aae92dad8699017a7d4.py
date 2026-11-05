"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def generate_c6s(n):
    """Generate all 6-cycles in K_n and map edges to cycles."""
    cycles = []
    edge_to_cycles = [[] for _ in range(n * (n - 1) // 2)]
    
    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        return u * n - u * (u + 1) // 2 + (v - u - 1)

    # Generate all combinations of 6 vertices
    for verts in combinations(range(n), 6):
        # Generate all permutations of these 6 vertices that form a cycle
        # A cycle is defined by its vertices in order.
        # To avoid duplicates, we can fix the first vertex to be the smallest.
        # But simpler: just generate all permutations and check uniqueness? 
        # No, 6! = 720 permutations. 12376 * 720 is too many.
        
        # Better: Generate unique cycles.
        # Fix v0 = min(verts).
        # Permute the remaining 5 vertices.
        # (5-1)! / 2 = 12 unique cycles per set of 6 vertices?
        # Number of cycles on 6 labeled vertices is (6-1)!/2 = 60.
        # If we fix one vertex, we have 5! = 120 permutations.
        # Each cycle is counted twice (forward/backward).
        # So 120/2 = 60. Correct.
        
        v0 = min(verts)
        others = [v for v in verts if v != v0]
        
        # Generate permutations of others
        from itertools import permutations
        seen = set()
        for perm in permutations(others):
            # Cycle: v0 - perm[0] - perm[1] - perm[2] - perm[3] - perm[4] - v0
            # To avoid duplicates (reverse), we can enforce an ordering, e.g., perm[0] < perm[4]
            if perm[0] > perm[4]:
                continue
            
            cycle_verts = [v0] + list(perm)
            # Get edge indices
            edges = []
            for i in range(6):
                u = cycle_verts[i]
                v = cycle_verts[(i + 1) % 6]
                idx = get_edge_idx(u, v)
                edges.append(idx)
            
            # Check if this set of edges was already seen? 
            # The permutation logic with v0 fixed and perm[0]<perm[4] should generate unique cycles.
            # Let's assume it does.
            
            cycle_id = len(cycles)
            cycles.append(edges)
            
            for e_idx in edges:
                edge_to_cycles[e_idx].append(cycle_id)
                
    return cycles, edge_to_cycles

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_n = 1
    best_colors = [0]
    
    # Try to find valid colorings for n=17, 18, 19, 20
    for target_n in [17, 18, 19, 20]:
        if time.time() > deadline:
            break
        
        m = target_n * (target_n - 1) // 2
        
        # Precompute cycles
        cycles, edge_to_cycles = generate_c6s(target_n)
        
        # Initialize colors
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Initialize counts
        # count_c[c] is the number of monochromatic C6s of color c
        count_c = [0, 0, 0, 0]
        
        # Calculate initial counts
        for c_id, edges in enumerate(cycles):
            c0 = colors[edges[0]]
            mono = True
            for e in edges[1:]:
                if colors[e] != c0:
                    mono = False
                    break
            if mono:
                count_c[c0] += 1
        
        total_c6 = sum(count_c)
        
        if total_c6 == 0:
            best_n = target_n
            best_colors = colors
            continue # Try next larger n
        
        # Local search
        found = False
        # We try to reduce total_c6 to 0
        # Use simulated annealing or simple hill climbing with random restarts
        
        # To save time, we limit the number of steps per restart
        max_steps = 5000
        
        for restart in range(20):
            if time.time() > deadline:
                break
            
            if restart > 0:
                # Random restart
                colors = [random.randint(0, 3) for _ in range(m)]
                count_c = [0, 0, 0, 0]
                for c_id, edges in enumerate(cycles):
                    c0 = colors[edges[0]]
                    mono = True
                    for e in edges[1:]:
                        if colors[e] != c0:
                            mono = False
                            break
                    if mono:
                        count_c[c0] += 1
                total_c6 = sum(count_c)
                
                if total_c6 == 0:
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                
                if total_c6 == 0:
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
                
                # Pick a random edge
                e_idx = random.randint(0, m - 1)
                old_color = colors[e_idx]
                
                # Try to change to a different color
                new_color = (old_color + 1) % 4
                
                # Update counts
                # For each cycle containing e_idx:
                delta_c = [0, 0, 0, 0]
                
                for c_id in edge_to_cycles[e_idx]:
                    edges = cycles[c_id]
                    # Check colors of other 5 edges
                    # If all 5 are old_color, then this cycle was mono old_color.
                    # It will no longer be mono old_color.
                    # If all 5 are new_color, then this cycle will be mono new_color.
                    
                    all_old = True
                    all_new = True
                    for e in edges:
                        if e == e_idx:
                            continue
                        if colors[e] != old_color:
                            all_old = False
                        if colors[e] != new_color:
                            all_new = False
                        if not all_old and not all_new:
                            break
                    
                    if all_old:
                        delta_c[old_color] -= 1
                    if all_new:
                        delta_c[new_color] += 1
                
                # Apply delta
                new_total = total_c6
                for c in range(4):
                    if count_c[c] + delta_c[c] < 0:
                        # Should not happen if logic is correct
                        pass
                    new_total += delta_c[c]
                
                # Accept if it improves or is neutral (to escape local minima, maybe accept worse sometimes?)
                # For now, only accept if it improves (total_c6 decreases) or is equal.
                if new_total <= total_c6:
                    colors[e_idx] = new_color
                    for c in range(4):
                        count_c[c] += delta_c[c]
                    total_c6 = new_total
                else:
                    # Revert (do nothing)
                    pass
        
        if found:
            continue
        else:
            # If we failed for this n, break?
            # If we fail 17, we likely can't do 18.
            if target_n == 17:
                # We might have a valid smaller n? 
                # We haven't tried smaller n.
                # Let's just return what we have.
                pass
            break
    
    # Fallback: if best_n is 1, ensure colors is valid (it is).
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
