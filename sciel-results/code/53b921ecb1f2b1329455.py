<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    return list(range(1, 257, 3))
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 256
    # Build conflict graph: vertex i (0-indexed, value i+1) conflicts with j if (i+1)+(j+1) is a perfect square
    # Include self-conflicts: (i+1)+(i+1) is a perfect square
    squares = set()
    for s in range(1, int(math.isqrt(2*N)) + 2):
        squares.add(s * s)
    
    # conflicts[i] is a bitset of vertices conflicting with vertex i
    conflicts = [0] * N
    for i in range(N):
        val = i + 1
        for j in range(i, N):
            s = val + (j + 1)
            if s in squares:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Branch and bound for maximum independent set
    # Use bitsets for speed
    best_size = 0
    best_set = None
    all_vertices = (1 << N) - 1
    
    # Greedy initial solution for lower bound
    def greedy_is(available):
        """Greedy independent set from available vertices (bitset)."""
        chosen = 0
        avail = available
        while avail:
            # Pick vertex with minimum degree in remaining graph
            best_v = -1
            best_deg = float('inf')
            temp = avail
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = bin(conflicts[v] & avail).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lb
            if best_v < 0:
                break
            chosen |= (1 << best_v)
            avail &= ~(conflicts[best_v] | (1 << best_v))
        return chosen
    
    # Initial greedy solution
    init_sol = greedy_is(all_vertices)
    best_size = bin(init_sol).count('1')
    best_set = init_sol
    
    # Upper bound: greedy coloring / packing bound
    def upper_bound(available):
        """Upper bound on independent set size in available subgraph.
        Use greedy coloring of the conflict graph restricted to available."""
        if available == 0:
            return 0
        # Simple bound: count vertices / (min_degree + 1) is weak
        # Better: greedy coloring gives upper bound
        # Actually for MIS, we need upper bound. Use: |V| is trivial.
        # Use a simple bound: sum over connected components of (n_comp / (min_deg_comp + 1))
        # Simpler: just use |available| as upper bound (weak but safe)
        # Better: use a greedy matching bound
        # For now use |available|
        return bin(available).count('1')
    
    # Better upper bound using greedy coloring of complement (i.e., coloring the conflict graph)
    def coloring_bound(available):
        """Upper bound via greedy coloring of the conflict graph on available vertices.
        The chromatic number of the conflict graph >= alpha, so we need a different approach.
        Actually: if we color the conflict graph, vertices of same color form an independent set.
        So the number of colors is an UPPER BOUND on... no, it's a LOWER bound on chromatic number.
        For MIS: we want an upper bound on |IS|. 
        Use: partition available into cliques in the conflict graph. Each clique contributes at most 1.
        Greedy clique cover: repeatedly pick a vertex and its neighbors (a clique in conflict graph = independent set in original... no).
        In conflict graph, a clique is a set where every pair conflicts. We can pick at most 1 from each clique.
        So greedy: pick vertex v, add all neighbors of v (which form a clique with v in conflict graph), remove them.
        """
        if available == 0:
            return 0
        count = 0
        avail = available
        while avail:
            # Pick vertex with max degree in avail
            best_v = -1
            best_deg = -1
            temp = avail
            while temp:
                lb = temp & (-temp)
                v = lb.bit_length() - 1
                deg = bin(conflicts[v] & avail).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp ^= lb
            if best_v < 0:
                break
            # Remove best_v and all its neighbors in avail (they form a clique with best_v)
            clique = (1 << best_v) | (conflicts[best_v] & avail)
            avail &= ~clique
            count += 1
        return count
    
    # Branch and bound
    def branch_and_bound(available, current_size):
        nonlocal best_size, best_set
        if time.time() > deadline:
            return
        
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = 0  # We'd need to track the actual set
            return
        
        ub = coloring_bound(available)
        if current_size + ub <= best_size:
            return
        
        # Choose branching vertex: pick vertex with max degree in available
        best_v = -1
        best_deg = -1
        temp = available
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = bin(conflicts[v] & available).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lb
        
        if best_v < 0:
            return
        
        v_bit = 1 << best_v
        
        # Branch 1: include best_v
        new_avail = available & ~(conflicts[best_v] | v_bit)
        branch_and_bound(new_avail, current_size + 1)
        
        # Branch 2: exclude best_v
        branch_and_bound(available & ~v_bit, current_size)
    
    # We need to track the actual set, not just size. Let me restructure.
    # Use a different approach: track the set as a list
    
    # Simpler approach with time limit
    best_result = None
    best_count = 0
    
    # Start with greedy solution
    greedy_list = []
    avail = set(range(N))
    while avail:
        # Pick vertex with min conflicts in remaining
        best_v = min(avail, key=lambda v: len(conflicts_set(v) & avail))
        greedy_list.append(best_v)
        avail -= conflicts_set(best_v)
        avail.discard(best_v)
    
    best_count = len(greedy_list)
    best_result = set(greedy_list)
    
    # Try to improve with local search
    # ... (this is getting complex, let me simplify)
    
    # Actually, let me just return the known good solution for now
    # and add a simple improvement attempt
    result = list(range(1, 257, 3))  # 85 elements, known valid
    
    # Try to add elements
    for x in range(1, 257):
        if x in result:
            continue
        # Check if x can be added
        can_add = True
        for y in result:
            s = x + y
            r = int(math.isqrt(s))
            if r * r == s:
                can_add = False
                break
        if can_add:
            result.append(x)
            result.sort()
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I made an error - I referenced `conflicts_set` which doesn't exist. Let me rewrite this properly.

