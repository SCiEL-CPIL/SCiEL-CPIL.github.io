import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Fallback valid coloring: n=5
    # Any coloring of K5 is valid because C6 requires 6 vertices.
    def get_fallback(n):
        colors = [0] * (n * (n - 1) // 2)
        return {'n': n, 'colors': colors}
    
    best_result = get_fallback(5)
    
    # Precomputation helper
    def precompute(n):
        """
        Precompute edge map and C6 cycles for a given n.
        Returns:
        edge_map: dict mapping (u,v) to index
        cycles: list of lists of edge indices
        edge_to_cycles: list of lists of cycle indices
        """
        edge_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map[(i, j)] = idx
                idx += 1
        
        def edge_index(u, v):
            if u > v:
                u, v = v, u
            return edge_map[(u, v)]
        
        vertices = list(range(n))
        cycles = []
        edge_to_cycles = [[] for _ in range(n * (n - 1) // 2)]
        
        # Generate all 6-combinations
        for combo in itertools.combinations(vertices, 6):
            v0 = combo[0]
            others = combo[1:]
            seen = set()
            for perm in itertools.permutations(others):
                # Cycle: v0, p0, p1, p2, p3, p4
                # To avoid reverse duplicates, enforce p0 < p4
                if perm[0] > perm[-1]:
                    continue
                
                # Canonicalize the cycle to avoid duplicates in 'seen' set?
                # The permutation logic above with p0 < p4 should handle most,
                # but let's be safe.
                path = [v0] + list(perm)
                
                # Check if this specific cyclic order was already added
                # We can represent the cycle by its edge set, but order matters for 
                # "is it a C6"? No, a set of 6 edges forming a C6 is unique.
                # Actually, for a fixed set of 6 vertices, there are 60 distinct C6s.
                # My permutation generation:
                # Fix v0. Permute others (5! = 120).
                # Divide by 2 for reversal -> 60.
                # The condition perm[0] < perm[-1] ensures we pick one from each reversal pair.
                # So we generate exactly 60 cycles per combination of 6 vertices.
                
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    e = edge_index(u, v)
                    cycle_edges.append(e)
                
                # Store
                cycle_id = len(cycles)
                cycles.append(cycle_edges)
                for e in cycle_edges:
                    edge_to_cycles[e].append(cycle_id)
        
        return edge_map, cycles, edge_to_cycles

    # Search function for a specific n
    def search_n(n, time_budget):
        if time.time() > deadline - 5:
            return None
            
        try:
            edge_map, cycles, edge_to_cycles = precompute(n)
        except Exception:
            return None
            
        m = n * (n - 1) // 2
        num_cycles = len(cycles)
        
        if num_cycles == 0:
            colors = [0] * m
            return {'n': n, 'colors': colors}
        
        # Data structures for incremental updates
        # cycle_color_counts[cycle_idx] = [count_c0, count_c1, count_c2, count_c3]
        cycle_color_counts = [[0, 0, 0, 0] for _ in range(num_cycles)]
        bad_count = 0
        
        # Greedy Initialization
        colors = [-1] * m
        
        for e in range(m):
            best_color = 0
            min_bad_increase = float('inf')
            
            for c in range(4):
                # Calculate increase in bad_count if we assign color c to edge e
                increase = 0
                for ci in edge_to_cycles[e]:
                    # Check current state of cycle ci assuming e is uncolored
                    # We need to check if making e color c makes the cycle monochromatic c
                    # This requires knowing the colors of the other 5 edges.
                    # Since we are filling sequentially, some might be -1.
                    # Greedy usually assumes previous are fixed.
                    
                    # If any other edge in the cycle is uncolored, this cycle cannot be "bad" yet?
                    # No, "bad" means monochromatic. If any edge is uncolored, it's not monochromatic.
                    # However, if we set e to c, and the other 5 are already c, then it becomes bad.
                    
                    is_potentially_bad = True
                    for ed in cycles[ci]:
                        if ed == e:
                            continue
                        if colors[ed] != c:
                            is_potentially_bad = False
                            break
                    
                    if is_potentially_bad:
                        # If the other 5 are c, and we set e to c, this cycle becomes bad (if it wasn't already).
                        # It wasn't bad because e was uncolored.
                        increase += 1
                
                if increase < min_bad_increase:
                    min_bad_increase = increase
                    best_color = c
            
            colors[e] = best_color
            # Update counts
            for ci in edge_to_cycles[e]:
                cycle_color_counts[ci][best_color] += 1
                # Check if it became bad
                if cycle_color_counts[ci][best_color] == 6:
                    bad_count += 1
        
        # If greedy already solved it
        if bad_count == 0:
            return {'n': n, 'colors': colors}
        
        # Simulated Annealing
        T = 10.0
        cool_rate = 0.9995
        max_iter = 200000
        start_iter_time = time.time()
        
        best_bad = bad_count
        best_colors = colors[:]
        
        for i in range(max_iter):
            if time.time() > start_iter_time + time_budget:
                break
            if time.time() > deadline - 5:
                break
                
            # Pick random edge
            e = random.randint(0, m - 1)
            c_old = colors[e]
            
            # Pick new color
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            # Calculate delta
            delta = 0
            for ci in edge_to_cycles[e]:
                # Was it bad?
                was_bad = (cycle_color_counts[ci][c_old] == 6)
                
                # Update counts temporarily
                cycle_color_counts[ci][c_old] -= 1
                cycle_color_counts[ci][c_new] += 1
                
                # Is it bad now?
                is_bad_now = (cycle_color_counts[ci][c_new] == 6)
                
                if was_bad and not is_bad_now:
                    delta -= 1
                elif not was_bad and is_bad_now:
                    delta += 1
            
            # Acceptance
            accept = False
            if delta < 0:
                accept = True
            else:
                if delta > 0:
                    if delta / T > 100:
                        accept = False
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            accept = True
                else: # delta == 0
                    accept = True
            
            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors[:]
                    if best_bad == 0:
                        return {'n': n, 'colors': best_colors}
            else:
                # Revert counts if rejected
                for ci in edge_to_cycles[e]:
                    cycle_color_counts[ci][c_old] += 1
                    cycle_color_counts[ci][c_new] -= 1
            
            T *= cool_rate
            if T < 0.1:
                T = 1.0 # Reheat
                
        # If we didn't find 0, but found a low bad count, we might still return it?
        # No, we need valid coloring.
        return None

    # Targets to try
    # We want to maximize n.
    # Prioritize 17, then 16, etc.
    
    targets = [17, 16, 15, 14, 13]
    
    # Time allocation
    # Total available: ~145s.
    # n=17 is the hardest. Give it the most time.
    # If n=17 fails, try n=16 with remaining time.
    
    time_alloc = {
        17: 60.0,
        16: 40.0,
        15: 20.0,
        14: 10.0,
        13: 5.0
    }
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        time_budget = time_alloc.get(n, 5.0)
        
        # Check if we have enough time
        if time.time() + time_budget > deadline - 5:
            time_budget = deadline - 5 - time.time()
            if time_budget < 1.0:
                continue
        
        result = search_n(n, time_budget)
        
        if result:
            # Found a valid coloring for n
            best_result = result
            # Since we iterate from high to low, the first valid one is the best for this run
            # unless we want to try higher n later? No, we start with highest.
            # So if we find n=17, we return.
            # If we find n=16, it implies n=17 failed or wasn't tried? 
            # We tried 17 first. If it failed, we try 16.
            # So returning immediately is correct.
            return best_result
        
    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
