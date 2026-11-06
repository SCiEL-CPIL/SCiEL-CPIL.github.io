"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # First, filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    # Build conflict graph: which valid numbers conflict with which
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # For each valid number, find all other valid numbers it conflicts with
    # a and b conflict if a + b is a perfect square (a != b)
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])
    
    # Build adjacency bitsets for fast operations
    adj_bits = []
    for i in range(num_valid):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        adj_bits.append(bits)
    
    # Greedy upper bound: greedy coloring of the induced subgraph on candidates
    def greedy_color_bound(candidates_bits):
        """Upper bound on independent set size using greedy coloring."""
        if candidates_bits == 0:
            return 0
        
        colors = []
        remaining = candidates_bits
        while remaining:
            # Greedy: pick vertex with max degree in remaining
            # For speed, just pick the lowest set bit
            v = (remaining & -remaining).bit_length() - 1
            remaining &= ~(1 << v)
            colors.append(v)
            # Remove neighbors from remaining
            remaining &= ~adj_bits[v]
        
        # The number of colors used is an upper bound on independence number
        # But this greedy approach gives a lower bound on chromatic number
        # Actually, for upper bound on independent set, we need a different approach
        
        # Better: use greedy set cover / packing as upper bound
        # The size of a maximal independent set found greedily is a lower bound
        # For upper bound, use: |V| / (1 + min_degree) or similar
        # Simpler upper bound: just count vertices (weak)
        
        # Let's use a proper greedy upper bound:
        # Process vertices in order, assign each to first available "color class"
        # where it has no conflicts with existing members
        # This gives an upper bound = number of color classes needed
        # Wait, that's for coloring, not independence
        
        # Actually, the standard upper bound for BB is:
        # Greedily find a maximal independent set, then the rest form a clique cover
        # Upper bound = size of greedy MIS + number of cliques needed to cover rest
        # Simplified: just return the number of vertices (very weak)
        
        # Let's use a better approach:
        # The independence number <= number of vertices in the graph
        # But we want a tighter bound.
        
        # Use: alpha(G) <= |V| - minimum vertex cover
        # And minimum vertex cover >= maximum matching (Kőnig's theorem for bipartite, but this isn't bipartite)
        
        # For simplicity, let's use a heuristic upper bound:
        # Process vertices in descending order of degree
        # Count how many we "need" by greedy packing
        pass
        
        # Simpler: return the count of candidates as a weak upper bound
        return bin(candidates_bits).count('1')
    
    # Better upper bound: greedy coloring
    def coloring_upper_bound(candidates_bits):
        """Upper bound on independence number via greedy coloring."""
        if candidates_bits == 0:
            return 0
        
        # Extract candidate vertices
        candidates = []
        temp = candidates_bits
        while temp:
            v = (temp & -temp).bit_length() - 1
            candidates.append(v)
            temp &= ~(1 << v)
        
        # Order by degree in the induced subgraph (descending)
        degs = []
        for v in candidates:
            deg = bin(adj_bits[v] & candidates_bits).count('1')
            degs.append((deg, v))
        degs.sort(reverse=True)
        
        # Greedy coloring
        color_of = {}
        num_colors = 0
        for deg, v in degs:
            # Find a color class where v has no conflicts
            placed = False
            for c in range(num_colors):
                # Check if v conflicts with any vertex in color class c
                # We need to track which vertices are in each color class
                # This is getting complex. Let's use a simpler bound.
                pass
        
        # Simpler upper bound: just use the number of vertices
        return len(candidates)
    
    # Even simpler: use a precomputed upper bound based on the structure
    # For now, let's use the trivial upper bound and rely on the search
    
    # Branch and bound search
    best_solution = []
    best_size = 0
    
    def bb_search(remaining_bits, current_set, current_size):
        nonlocal best_solution, best_size
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_size + number of remaining vertices
        ub = current_size + bin(remaining_bits).count('1')
        if ub <= best_size:
            return
        
        if remaining_bits == 0:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_set)
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in remaining
        v = (remaining_bits & -remaining_bits).bit_length() - 1
        # Find vertex with max degree
        temp = remaining_bits
        best_v = -1
        best_deg = -1
        while temp:
            u = (temp & -temp).bit_length() - 1
            deg = bin(adj_bits[u] & remaining_bits).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = u
            temp &= ~(1 << u)
        v = best_v
        
        # Branch 1: include v
        new_remaining = remaining_bits & ~(1 << v) & ~adj_bits[v]
        current_set.add(v)
        bb_search(new_remaining, current_set, current_size + 1)
        current_set.discard(v)
        
        # Branch 2: exclude v
        new_remaining = remaining_bits & ~(1 << v)
        bb_search(new_remaining, current_set, current_size)
    
    # Initial greedy solution to set a lower bound
    def greedy_initial():
        remaining = set(range(num_valid))
        selected = []
        while remaining:
            min_deg = float('inf')
            best_idx = -1
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)
            for neighbor in conflict[best_idx]:
                remaining.discard(neighbor)
        return [valid_numbers[i] for i in selected]
    
    initial = greedy_initial()
    best_size = len(initial)
    best_solution = initial
    
    # Convert initial solution to bitset for reference
    initial_set = set(initial)
    
    # Run branch and bound with time limit
    all_bits = (1 << num_valid) - 1
    bb_search(all_bits, set(), 0)
    
    # Return the best solution found
    if best_solution:
        return sorted(best_solution)
    else:
        return initial
# EVOLVE-BLOCK-END
