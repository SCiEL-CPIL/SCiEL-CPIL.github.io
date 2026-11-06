import time
import random
import sys
import itertools
import math
import numpy as np

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use a safety margin of 10s.
    deadline = start_time + 140.0 

    def get_fallback(n):
        # A valid coloring for small n is easy.
        # For n=6, any coloring works if we avoid C6? 
        # Actually, K6 has 15 edges. C6 uses 6.
        # A simple fallback: return n=1 with empty colors or n=5.
        # The task requires 1 <= n <= 20.
        # Let's return a trivially valid small n to ensure non-zero score if all else fails.
        # n=5 has no C6 cycles at all.
        m = 5 * 4 // 2
        colors = [0] * m
        return {'n': 5, 'colors': colors}

    best_result = get_fallback(5)

    def edge_index(i, j, n):
        if i > j:
            i, j = j, i
        # i is 0..n-2, j is i+1..n-1
        # Number of edges before i is 0 + 1 + ... + (i-1) = i*(i-1)/2
        # Edges in row i: (i, i+1), ..., (i, j) -> j - i
        # Total index = i*(n-2) - i*(i-1)//2 + j - i - 1 
        # Let's verify for n=4.
        # Edges: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
        # Formula: i*(n-2) - i*(i-1)//2 + j - 1
        # (0,1): 0 - 0 + 0 = 0. Correct.
        # (0,2): 0 - 0 + 1 = 1. Correct.
        # (0,3): 0 - 0 + 2 = 2. Correct.
        # (1,2): 1*(2) - 0 + 1 = 3. Correct.
        # (1,3): 1*(2) - 0 + 2 = 4. Correct.
        # (2,3): 2*(2) - 1 + 2 = 5. Correct.
        return i * (n - 2) - i * (i - 1) // 2 + j - 1

    def build_structure(n):
        """
        Builds the list of C6 cycles and the map from edge to other edges in those cycles.
        Returns:
        cycles: list of lists of edge indices (length 6)
        edge_to_contexts: list of lists. edge_to_contexts[e] is a list of arrays, 
                          each array contains the 5 other edges in a C6 with e.
        """
        m = n * (n - 1) // 2
        if n < 6:
            return [], [[] for _ in range(m)]

        verts = list(range(n))
        cycles_list = []
        
        # Generate all 6-cycles
        # A 6-cycle is determined by 6 distinct vertices.
        # To avoid duplicates, we fix the smallest vertex as the start.
        # Then we fix the second vertex to be smaller than the last vertex to handle rotational symmetry?
        # Actually, standard way:
        # 1. Choose 6 vertices.
        # 2. Fix one vertex (e.g., the smallest) at position 0.
        # 3. Permute the other 5.
        # 4. To handle reversal symmetry (cycle (0,1,2,3,4,5) is same as (0,5,4,3,2,1)), 
        #    we can require that the second vertex < the last vertex.
        
        for combo in itertools.combinations(verts, 6):
            v0 = combo[0]
            others = combo[1:]
            for perm in itertools.permutations(others):
                # Check reversal symmetry: perm[0] < perm[-1]
                if perm[0] > perm[-1]:
                    continue
                
                path = [v0] + list(perm)
                cycle_edges = []
                for k in range(6):
                    u = path[k]
                    v = path[(k + 1) % 6]
                    cycle_edges.append(edge_index(u, v, n))
                cycles_list.append(cycle_edges)
        
        if not cycles_list:
            return [], [[] for _ in range(m)]

        # Precompute edge_to_contexts
        # For each edge e, we want to know the other 5 edges for every cycle containing e.
        edge_to_contexts = [[] for _ in range(m)]
        
        for cycle_edges in cycles_list:
            for e in cycle_edges:
                # Create a tuple of the other 5 edges
                other_edges = [x for x in cycle_edges if x != e]
                edge_to_contexts[e].append(tuple(other_edges))
                
        return cycles_list, edge_to_contexts

    def calculate_initial_bad(colors, cycles_list):
        """
        colors: np.array of shape (M,)
        cycles_list: list of lists of edge indices
        Returns: number of monochromatic cycles
        """
        bad_count = 0
        for cycle_edges in cycles_list:
            c0 = colors[cycle_edges[0]]
            if all(colors[e] == c0 for e in cycle_edges[1:]):
                bad_count += 1
        return bad_count

    def search_n(n, time_budget, seed_offset=0):
        if time.time() > deadline - 5:
            return None

        m = n * (n - 1) // 2
        random.seed(42 + seed_offset)
        np.random.seed(42 + seed_offset)

        cycles_list, edge_to_contexts = build_structure(n)
        N_cycles = len(cycles_list)

        if N_cycles == 0:
            # No cycles to violate, any coloring works
            colors = np.zeros(m, dtype=np.uint8)
            return {'n': n, 'colors': colors.tolist()}

        # Initial random coloring
        colors = np.random.randint(0, 4, size=m, dtype=np.uint8)
        bad_count = calculate_initial_bad(colors, cycles_list)
        
        if bad_count == 0:
            return {'n': n, 'colors': colors.tolist()}

        best_colors = colors.copy()
        best_bad = bad_count

        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.05
        start_iter_time = time.time()
        
        # Precompute a list of indices for random selection from bad cycles?
        # It's hard to maintain a list of bad cycles efficiently in Python without overhead.
        # Strategy: 
        # 1. Random edge selection.
        # 2. If we pick an edge, we calculate delta.
        # To bias towards bad cycles, we can periodically scan for bad cycles?
        # Or, we can just rely on SA. The delta calculation is the key.
        
        # Let's add a "Walk" component:
        # With probability 0.5, if there are bad cycles, try to pick an edge from a bad cycle.
        # But finding a bad cycle requires a scan.
        # Alternative: Just use SA with a good cooling schedule.
        
        while time.time() < start_iter_time + time_budget:
            if time.time() > deadline - 5:
                break

            # Pick random edge
            e = np.random.randint(0, m)
            c_old = colors[e]
            
            # Pick new color
            c_new = np.random.randint(0, 4)
            while c_new == c_old:
                c_new = np.random.randint(0, 4)

            # Calculate Delta
            contexts = edge_to_contexts[e]
            if not contexts:
                delta = 0
            else:
                delta = 0
                # For each context (5 other edges), check if they form a mono cycle with e
                for other_edges in contexts:
                    # Check if all other_edges have color c_old
                    # If yes, this context was a violation before.
                    # Check if all other_edges have color c_new
                    # If yes, this context will be a violation after.
                    
                    # Using numpy for speed if contexts are large?
                    # m is at most 190 for n=20. N_cycles is ~3000.
                    # Each edge is in ~50 cycles.
                    # Python loop over 50 tuples of 5 ints is fast enough.
                    
                    is_old_mono = True
                    is_new_mono = True
                    for oe in other_edges:
                        ce = colors[oe]
                        if ce != c_old:
                            is_old_mono = False
                        if ce != c_new:
                            is_new_mono = False
                        # Early exit
                        if not is_old_mono and not is_new_mono:
                            break
                    
                    if is_old_mono:
                        delta -= 1
                    if is_new_mono:
                        delta += 1

            # Acceptance
            accept = False
            if delta <= 0:
                accept = True
            else:
                # Temperature
                elapsed = time.time() - start_iter_time
                progress = min(1.0, elapsed / time_budget)
                T = T_start * (T_end / T_start) ** progress
                
                # Cap the exponent to avoid underflow/overflow issues in exp
                if delta / T > 50:
                    accept = False
                else:
                    prob = math.exp(-delta / T)
                    accept = random.random() < prob

            if accept:
                colors[e] = c_new
                bad_count += delta
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_colors = colors.copy()
                    if best_bad == 0:
                        return {'n': n, 'colors': colors.tolist()}

        if best_bad == 0:
            return {'n': n, 'colors': best_colors.tolist()}
        return None

    # Try targets from high to low
    # Prioritize n=18 (breakthrough) and n=17 (known best)
    targets = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
    
    # Time allocation strategy:
    # Spend significant time on 18 and 17.
    # If 18 fails, ensure 17 is found.
    
    # We have ~140s.
    # n=18: 60s
    # n=17: 50s
    # n=16: 10s
    # Rest: 10s
    
    time_alloc = {
        18: 60, 
        17: 50, 
        16: 10, 
        15: 5, 
        14: 3, 
        13: 2, 
        12: 1, 
        11: 1, 
        10: 1, 
        9: 1, 
        8: 1, 
        7: 1, 
        6: 1
    }

    for n in targets:
        if time.time() > deadline - 10:
            break

        tb = time_alloc.get(n, 2)
        
        # If we are trying n=18 or 17, we might want multiple restarts
        # but SA with a long run is often better than many short runs for CSPs.
        # However, random restarts help escape local minima.
        # Let's do 2-3 restarts for high n.
        
        num_restarts = 3 if n >= 17 else 2
        restart_budget = tb / num_restarts

        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            result = search_n(n, restart_budget, seed_offset=r * 1000 + n)
            if result:
                best_result = result
                return best_result

    return best_result

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
