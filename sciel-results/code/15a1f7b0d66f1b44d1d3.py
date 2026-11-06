"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # Self-conflict: 2a is a perfect square -> a is forbidden (diagonal)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}

    # Build conflict edges: a conflicts with b (a != b) if a + b is a perfect square
    # We need to find a maximum independent set in this graph.
    #
    # KEY INSIGHT: The conflict graph is bipartite!
    # Why? If a + b = k^2, then a and b have opposite parity when k is odd,
    # but same parity when k is even. Wait, let me reconsider.
    #
    # Actually, let's check: if a+b = square, the parity of a+b depends on the square.
    # Odd squares: 1, 9, 25, 49, 81, 121, 169, 225 -> odd
    # Even squares: 4, 16, 36, 64, 100, 144, 196, 256 -> even
    # So a+b can be odd or even. This means the graph is NOT necessarily bipartite
    # by parity. Let me verify with a concrete example.
    #
    # 1 + 3 = 4 (even square): both odd -> same parity edge
    # 1 + 8 = 9 (odd square): 1 odd, 8 even -> different parity edge
    # 2 + 7 = 9: 2 even, 7 odd -> different parity
    # 3 + 6 = 9: 3 odd, 6 even -> different parity
    # 5 + 4 = 9: 5 odd, 4 even -> different parity
    # 1 + 15 = 16: both odd -> same parity
    # 2 + 14 = 16: both even -> same parity
    #
    # So the graph has edges within same-parity groups too. It's NOT bipartite.
    #
    # Alternative: Use the fact that this is a maximum independent set problem.
    # For general graphs, MIS is NP-hard, but our graph has special structure.
    #
    # Let me think differently. The graph is defined by: edge (a,b) iff a+b is a perfect square.
    # Each vertex a has neighbors {sq - a : sq in squares, 1 <= sq-a <= N, sq-a != a, sq-a not forbidden}
    #
    # Since N=256 is small, let's use a different exact approach:
    # Maximum Independent Set via Maximum Matching on bipartite graph... no, not bipartite.
    #
    # Let me try: convert MIS to Minimum Vertex Cover on bipartite graphs only works if bipartite.
    #
    # New approach: Since the graph is small (M ~ 230 vertices), use a branch-and-bound
    # with a MUCH better upper bound. Or better yet, use the following observation:
    #
    # The conflict graph has bounded degree. Let me compute degrees.
    # Each vertex a conflicts with at most len(squares) - 1 vertices (one per square, minus self).
    # len(squares) for squares <= 512: 1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400,441,484,512? No, 22^2=484, 23^2=529>512.
    # So 22 squares. Each vertex has at most 21 neighbors (excluding self).
    #
    # With bounded degree ~21 and 230 vertices, we can use a clever approach.
    #
    # Actually, let me reconsider the bipartite structure more carefully.
    # Color by: color(a) = a mod 2? No, we showed same-parity edges exist.
    #
    # What about coloring by a mod something else?
    # For a+b = k^2:
    # If k is odd, k^2 ≡ 1 (mod 8) for k odd... no, k^2 mod 8:
    # 1^2=1, 3^2=1, 5^2=1, 7^2=1 (mod 8) -> odd squares ≡ 1 mod 8
    # 2^2=4, 4^2=0, 6^2=4, 8^2=0 (mod 8) -> even squares ≡ 0 or 4 mod 8
    #
    # This doesn't give a clean 2-coloring.
    #
    # Let me try a completely different structural approach:
    # Use integer linear programming with branch and bound, but with a strong LP relaxation
    # via the clique cover bound.
    #
    # Actually, for N=256 with ~230 valid vertices and max degree ~21,
    # a well-implemented branch-and-bound with greedy coloring upper bound
    # should be able to solve this exactly in reasonable time.
    #
    # But the seed already does this and gets 1.012. Let me try a different angle.
    #
    # NEW STRATEGY: Use the fact that the graph is "almost" bipartite.
    # Most edges go between odd and even (from odd squares).
    # Same-parity edges come from even squares.
    # 
    # Let me check: how many same-parity edges are there?
    # Even squares: 4, 16, 36, 64, 100, 144, 196, 256
    # For a+b = 4: (1,3) both odd
    # a+b = 16: (1,15),(2,14),(3,13),(4,12),(5,11),(6,10),(7,9) - mixed parity
    # Wait, 1+15=16: both odd. 2+14=16: both even. 3+13=16: both odd. 4+12=16: both even.
    # So for even squares, we get same-parity edges.
    #
    # This is getting complex. Let me just implement a very efficient B&B.
    #
    # Actually, the BEST structural change: use a SAT/ILP approach with
    # a different decomposition. Or better: use the following.
    #
    # INSIGHT: The problem is equivalent to finding a maximum set of numbers
    # such that no two sum to a square. This is a "sum-free" type problem
    # with respect to the set of squares.
    #
    # Let me try a different algorithmic family: use a local search with
    # simulated annealing, which is a fundamentally different optimization
    # paradigm from B&B.
    
    # Build adjacency
    adj = [0] * M
    for i, a in enumerate(valid_numbers):
        for sq in squares:
            b = sq - a
            if 1 <= b <= N and b != a:
                if b in num_to_idx:
                    j = num_to_idx[b]
                    adj[i] |= (1 << j)

    # Simulated annealing approach
    import random
    import math

    random.seed(12345)

    def is_valid_mask(mask):
        """Check if mask represents a valid independent set."""
        for i in range(M):
            if (mask >> i) & 1:
                if adj[i] & mask:
                    return False
        return True

    # Initial solution: greedy
    degrees = [bin(adj[i]).count('1') for i in range(M)]
    degree_order = sorted(range(M), key=lambda i: degrees[i])
    
    best_mask = 0
    for i in degree_order:
        if (adj[i] & best_mask) == 0:
            best_mask |= (1 << i)
    
    best_size = bin(best_mask).count('1')
    
    # Simulated annealing
    current_mask = best_mask
    current_size = best_size
    
    # Precompute for speed
    def get_selected(mask):
        return [i for i in range(M) if (mask >> i) & 1]
    
    def get_unselected(mask):
        return [i for i in range(M) if not (mask >> i) & 1]
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 200000
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        if T < T_min:
            break
        
        # Choose a random move
        move_type = random.random()
        selected = get_selected(current_mask)
        unselected = get_unselected(current_mask)
        
        if not selected or not unselected:
            break
        
        new_mask = current_mask
        delta = 0
        
        if move_type < 0.3:
            # Add a vertex
            i = random.choice(unselected)
            if (adj[i] & current_mask) == 0:
                new_mask = current_mask | (1 << i)
                delta = 1
            else:
                continue
        elif move_type < 0.6:
            # Remove a vertex
            i = random.choice(selected)
            new_mask = current_mask & ~(1 << i)
            delta = -1
        else:
            # Swap: remove one selected, add one unselected
            i = random.choice(selected)
            j = random.choice(unselected)
            # Check if adding j conflicts with current set minus i
            temp_mask = current_mask & ~(1 << i)
            if (adj[j] & temp_mask) == 0:
                new_mask = temp_mask | (1 << j)
                delta = 0  # size unchanged, but structure changes
            else:
                continue
        
        new_size = current_size + delta
        
        if new_size >= best_size:
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
            current_mask = new_mask
            current_size = new_size
        else:
            # Accept with probability exp(delta / T)
            # delta is negative here
            prob = math.exp(delta / T)
            if random.random() < prob:
                current_mask = new_mask
                current_size = new_size
        
        T *= cooling
    
    # Final greedy improvement
    improved = True
    while improved:
        improved = False
        for i in degree_order:
            if not (best_mask >> i) & 1:
                if (adj[i] & best_mask) == 0:
                    best_mask |= (1 << i)
                    best_size += 1
                    improved = True
    
    # Convert back to actual numbers
    result = [valid_numbers[i] for i in range(M) if (best_mask >> i) & 1]
    
    if not result:
        result = [x for x in range(1, N + 1, 3) if x not in forbidden]
    
    return sorted(result)
# EVOLVE-BLOCK-END
