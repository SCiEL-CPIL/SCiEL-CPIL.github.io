import time
import random

N = 40
DEADLINE = 85.0

def construct():
    start = time.time()
    
    # Helper to normalize edge
    def norm_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    # Base graph: The 40-cycle rim (40 edges)
    base_edges = [norm_edge(i, (i + 1) % N) for i in range(N)]
    
    # Initial state: A known valid solution (Fan-like or specific chord set)
    # We start with the 17-chord solution found in previous iterations (score 2.176)
    # Chords: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22
    # This covers all lengths 3..40.
    # Lengths covered by chord d: d+1 and N-d+1
    # d=2: 3, 39
    # d=3: 4, 38
    # ...
    # d=17: 18, 24
    # d=22: 23, 19 (Wait, 22+1=23, 40-22+1=19)
    # Missing: 20, 21, 22?
    # Let's re-verify the 17-chord set from the "cell findings": 
    # "chord at 22 to cover lengths 20, 21, 22 via differences with chords at 4, 3, 2"
    # This implies composite cycles.
    # For the search, we just need a valid starting point.
    # Let's use a safe 19-chord set to start, then try to remove edges.
    # Chords 2..20 covers:
    # d=2: 3, 39
    # d=3: 4, 38
    # ...
    # d=19: 20, 22
    # d=20: 21, 21 (Self-loop? No, 0-20 is a diameter. Cycle 0..20..0? No, simple cycle must be distinct vertices.
    # A chord (0, 20) creates cycles of length 21 (0..20) and 21 (0, 39, 38..20).
    # So d=20 covers 21.
    # d=19 covers 20 and 22.
    # So chords 2..20 (19 chords) cover 3..40.
    
    initial_chords = list(range(2, 21)) # 2 to 20 inclusive
    current_edges = base_edges + [norm_edge(0, d) for d in initial_chords]
    
    # Verify initial state validity
    def get_cycles_and_validity(edge_list):
        edge_set = set(edge_list)
        if len(edge_set) != len(edge_list):
            return None, False
        
        cycles = {}
        # 40-cycle is always the rim
        cycles[40] = list(range(N))
        
        # Check chords from 0
        for d in range(2, N):
            if norm_edge(0, d) in edge_set:
                # Cycle 1: 0, 1, ..., d (length d+1)
                if 3 <= d + 1 <= 40:
                    cycles[d + 1] = list(range(d + 1))
                # Cycle 2: 0, d, d+1, ..., 39 (length N - d + 1)
                if 3 <= N - d + 1 <= 40:
                    cycles[N - d + 1] = [0] + list(range(d, N))
        
        # Check if all 3..40 are covered by these simple cycles
        # Note: This simple check misses composite cycles. 
        # However, for the search heuristic, if we find a set of chords that covers all lengths via simple cycles, 
        # it's a valid solution. If not, we might still have a valid solution via composite cycles, 
        # but proving that is expensive. 
        # To be safe, we only accept states where simple cycles cover all lengths, 
        # OR we implement a quick BFS for missing lengths.
        
        missing = [k for k in range(3, 41) if k not in cycles]
        
        if missing:
            # Try to find cycles for missing lengths via BFS on the current graph
            # This is expensive, so we only do it if we are close to a solution or at the end.
            # For now, return the number of missing simple-cycle lengths as a penalty.
            return None, False
        
        # Verify all cycles are valid (edges exist)
        for k, cyc in cycles.items():
            if len(cyc) != k:
                return None, False
            if len(set(cyc)) != k:
                return None, False
            for i in range(k):
                u, v = cyc[i], cyc[(i + 1) % k]
                if norm_edge(u, v) not in edge_set:
                    return None, False
        
        return cycles, True

    current_cycles, current_valid = get_cycles_and_validity(current_edges)
    
    # If initial is not valid via simple chords (it should be), fallback to fan
    if not current_valid:
         # Fallback: Fan graph
         fan_edges = [norm_edge(0, i) for i in range(1, N)] + [norm_edge(i, i+1) for i in range(1, N-1)]
         fan_cycles = [list(range(k)) for k in range(3, 41)]
         # Verify fan
         es = set(fan_edges)
         for k, c in zip(range(3, 41), fan_cycles):
             for i in range(k):
                 if norm_edge(c[i], c[(i+1)%k]) not in es:
                     # Should not happen
                     pass
         return {'edges': fan_edges, 'cycles': fan_cycles}

    best_edges = current_edges
    best_cycles = current_cycles
    
    # Simulated Annealing / Local Search
    # Goal: Remove chords that are redundant.
    # A chord d is redundant if removing it still allows all cycles 3..40 to be formed.
    
    t = 0.5
    cooling = 0.999
    
    while time.time() < start + DEADLINE:
        # Try to remove a random chord
        if len(current_edges) <= 40:
            break # Can't go below rim
        
        # Identify removable candidates: chords (edges not in rim)
        rim_set = set(base_edges)
        chords = [e for e in current_edges if e not in rim_set]
        
        if not chords:
            break
            
        # Pick a random chord to remove
        idx = random.randint(0, len(chords) - 1)
        chord_to_remove = chords[idx]
        
        new_edges = [e for e in current_edges if e != chord_to_remove]
        
        # Check validity of new_edges
        # Optimization: Only check if the removed chord was critical for some length
        # But for simplicity and robustness, run the checker.
        new_cycles, new_valid = get_cycles_and_validity(new_edges)
        
        if new_valid:
            # Accept improvement
            current_edges = new_edges
            current_cycles = new_cycles
            best_edges = current_edges
            best_cycles = current_cycles
            t *= cooling
        else:
            # Reject (or accept with prob exp(-delta/T) if we had a cost function, but here cost is binary valid/invalid)
            # Since we want to minimize edges, and validity is a hard constraint, we only accept valid moves.
            # This is a greedy local search.
            # To escape local optima, we could allow temporary invalid states and try to fix them, 
            # but that's complex. Greedy removal is usually effective for sparse graphs.
            
            # If we get stuck, we might try adding a different chord? No, we want to minimize.
            # If greedy gets stuck, we restart with a different random seed or perturbation.
            # For now, just continue. If no improvement in many steps, we could break.
            
            # Simple stagnation check
            if t < 0.1:
                # Reset temperature or break
                break

    # Final verification
    final_cycles, final_valid = get_cycles_and_validity(best_edges)
    if not final_valid:
        # Should not happen, but fallback
        fan_edges = [norm_edge(0, i) for i in range(1, N)] + [norm_edge(i, i+1) for i in range(1, N-1)]
        fan_cycles = [list(range(k)) for k in range(3, 41)]
        return {'edges': fan_edges, 'cycles': fan_cycles}

    return {'edges': best_edges, 'cycles': [final_cycles[k] for k in range(3, 41)]}

