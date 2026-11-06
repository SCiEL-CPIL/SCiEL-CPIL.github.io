import time
import sys
import random
from itertools import combinations

N = 40

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0
    
    # Fallback: The 59-edge solution (19 chords)
    def get_59_edge_solution():
        base_edges = [(i, (i + 1) % N) for i in range(N)]
        base_edge_set = set((min(e), max(e)) for e in base_edges)
        
        # Chords to cover all lengths 3..39
        # We need 19 chords.
        # Distances 2..20 cover 3..21 and 41-2..41-20 -> 39..22.
        # So distances 2..20 cover 3..39.
        # We need to pick u for each d in 2..20.
        chords = []
        for d in range(2, 21):
            u = 0 # Simple choice
            v = (u + d) % N
            chords.append((u, v))
            
        all_edges = base_edges + chords
        edge_set = set((min(e), max(e)) for e in all_edges)
        
        cycles = []
        # Map length to chord and direction
        # d covers d+1 and N-d+1
        # For L in 3..21, use d=L-1, short cycle
        # For L in 22..39, use d=41-L, long cycle
        # For L=40, base cycle
        
        # Precompute cycles for each L
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
                continue
            
            if L <= 21:
                d = L - 1
                # Find chord with distance d
                # In this simple fallback, u=0 for all
                u = 0
                v = (u + d) % N
                # Short cycle: u..v
                cycle = list(range(u, v + 1)) if v > u else list(range(u, N)) + list(range(0, v + 1))
                # Wait, if u=0, v=d. range(0, d+1) is correct.
                # But if we used random u, we need to handle wrap.
                # For fallback, u=0 is safe.
                cycles.append(cycle)
            else:
                d = 41 - L
                u = 0
                v = (u + d) % N
                # Long cycle: u, u-1, ..., v
                # This is the complement of the short path
                short_path = list(range(u, v + 1)) if v > u else list(range(u, N)) + list(range(0, v + 1))
                # The long cycle is the rest of the vertices + u
                # Actually, easier to construct:
                # Start at u, go backwards to v
                cycle = [u]
                cur = (u - 1) % N
                while cur != v:
                    cycle.append(cur)
                    cur = (cur - 1) % N
                cycle.append(v)
                cycles.append(cycle)
                
        return {'edges': all_edges, 'cycles': cycles}

    # Try to find a 58-edge solution
    # Chords with distances 2..19
    # Missing length 21
    
    def check_cycle_21(edge_set):
        # Try to find a simple cycle of length 21
        # Randomized DFS
        # To make it fast, try a few random starts and orders
        for _ in range(50): # 50 attempts
            start = random.randint(0, N - 1)
            # Use a stack for DFS
            # State: (current_node, path)
            # We need to avoid revisiting nodes in the path
            # Since N=40, depth 21 is manageable
            
            # We can use a recursive function or iterative
            # Iterative is safer for stack limits, but recursion is cleaner
            # Let's use a simple backtracking with random neighbor order
            
            def dfs(node, depth, path, visited):
                if depth == 21:
                    # Check if edge (node, start) exists
                    if (min(node, start), max(node, start)) in edge_set:
                        return path
                    return None
                
                neighbors = []
                for i in range(N):
                    if (min(node, i), max(node, i)) in edge_set:
                        if i not in visited:
                            neighbors.append(i)
                
                random.shuffle(neighbors)
                
                for nb in neighbors:
                    if nb == start and depth < 20:
                        continue # Don't close cycle early
                    res = dfs(nb, depth + 1, path + [nb], visited | {nb})
                    if res:
                        return res
                return None
            
            res = dfs(start, 1, [start], {start})
            if res:
                return res
        
        return None

    # Search for 58-edge solution
    best_58 = None
    attempts = 0
    while time.time() - start_time < deadline - 5.0:
        attempts += 1
        # Assign random u to each distance 2..19
        chords = []
        for d in range(2, 20):
            u = random.randint(0, N - 1)
            v = (u + d) % N
            chords.append((u, v))
            
        base_edges = [(i, (i + 1) % N) for i in range(N)]
        all_edges = base_edges + chords
        edge_set = set((min(e), max(e)) for e in all_edges)
        
        # Check for cycle 21
        cycle_21 = check_cycle_21(edge_set)
        if cycle_21:
            # Construct full cycles list
            cycles = []
            # We need to map L to the specific chord used
            # For L in 3..20, use chord with d=L-1
            # For L in 22..39, use chord with d=41-L
            # We need to know which u was used for that d
            
            chord_by_d = {}
            for u, v in chords:
                # d is distance
                # We need to recover d from u, v?
                # d = (v - u) % N
                # But if v < u, d = (v - u + N) % N
                # In our generation, v = (u+d)%N, so d is the distance along the cycle
                d = (v - u) % N
                if d < 2 or d > 19: continue # Should not happen
                chord_by_d[d] = (u, v)
            
            for L in range(3, N + 1):
                if L == N:
                    cycles.append(list(range(N)))
                elif L == 21:
                    cycles.append(cycle_21)
                elif L <= 20:
                    d = L - 1
                    u, v = chord_by_d[d]
                    # Short cycle
                    if v > u:
                        cycle = list(range(u, v + 1))
                    else:
                        cycle = list(range(u, N)) + list(range(0, v + 1))
                    cycles.append(cycle)
                else: # 22 <= L <= 39
                    d = 41 - L
                    u, v = chord_by_d[d]
                    # Long cycle
                    cycle = [u]
                    cur = (u - 1) % N
                    while cur != v:
                        cycle.append(cur)
                        cur = (cur - 1) % N
                    cycle.append(v)
                    cycles.append(cycle)
            
            # Verify
            valid = True
            for c in cycles:
                if len(set(c)) != len(c):
                    valid = False
                    break
                for i in range(len(c)):
                    a = c[i]
                    b = c[(i+1)%len(c)]
                    if (min(a,b), max(a,b)) not in edge_set:
                        valid = False
                        break
                if not valid: break
            
            if valid:
                return {'edges': all_edges, 'cycles': cycles}
        
        if attempts % 100 == 0:
            # Check time
            if time.time() - start_time > deadline:
                break

    # Fallback to 59 edges
    return get_59_edge_solution()
# EVOLVE-BLOCK-END
