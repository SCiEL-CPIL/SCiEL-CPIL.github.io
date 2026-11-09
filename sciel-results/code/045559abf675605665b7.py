import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 256

    # Build conflict graph using the gcd characterization:
    # a,b conflict iff gcd(x,y)=1 and (x+y)|g where a=g*x, b=g*y
    # conflict[a] is a bitset (Python int) with bit (b-1) set if a and b conflict.
    conflict = [0] * (N + 1)  # 1-indexed

    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if math.gcd(x, y) != 1:
                continue
            s = x + y
            if s > N:
                continue
            # g must be a multiple of s, and g*x <= N, g*y <= N
            # max g = N // max(x, y) = N // y (since y > x)
            max_g = N // y
            for g in range(s, max_g + 1, s):
                a = g * x
                b = g * y
                if a <= N and b <= N:
                    conflict[a] |= (1 << (b - 1))
                    conflict[b] |= (1 << (a - 1))

    # Compute degrees
    degrees = [0] * (N + 1)
    for v in range(1, N + 1):
        degrees[v] = conflict[v].bit_count()

    # Greedy independent set for initial bound
    # Try multiple orderings: by degree ascending, random, etc.
    all_nodes = list(range(1, N + 1))

    def greedy_is(order):
        chosen_bits = 0
        chosen = []
        for v in order:
            if (conflict[v] & chosen_bits) == 0:
                chosen_bits |= (1 << (v - 1))
                chosen.append(v)
        return chosen_bits, chosen, len(chosen)

    # Order by degree ascending (low degree first tends to give larger IS)
    order_deg_asc = sorted(all_nodes, key=lambda v: degrees[v])
    bits, lst, sz = greedy_is(order_deg_asc)
    best_size = sz
    best_set = set(lst)

    # Try a few random orderings for better initial bound
    import random
    rng = random.Random(42)
    for _ in range(20):
        order = all_nodes[:]
        rng.shuffle(order)
        _, _, sz2 = greedy_is(order)
        if sz2 > best_size:
            best_size = sz2
            _, lst2, _ = greedy_is(order)
            best_set = set(lst2)

    # Branch and bound for maximum independent set
    # State: current independent set (list), available candidates (bitset)
    # A candidate v can be added if (conflict[v] & current_bits) == 0
    # Upper bound: len(current) + popcount(available)
    # Branching: pick vertex with max degree within available
    
    # Precompute: for each vertex, its neighbors as a bitset (already in conflict[v])
    
    # We'll use a recursive B&B with bitsets
    # To speed up, precompute the order of vertices by degree for branching
    
    # Sort vertices by degree descending for branching priority
    vertices_by_deg_desc = sorted(all_nodes, key=lambda v: -degrees[v])
    
    # For branching, we want to pick the vertex in 'available' with highest degree
    # We can iterate vertices_by_deg_desc and find the first one in available
    
    best_size_global = best_size
    best_set_global = set(best_set)
    
    # Convert best_set to bitset
    best_bits_global = 0
    for v in best_set_global:
        best_bits_global |= (1 << (v - 1))
    
    # Local Search to improve the greedy solution
    # Strategy: Try to swap one vertex out and fill the space with greedy.
    
    all_available = 0
    for v in range(1, N + 1):
        all_available |= (1 << (v - 1))
        
    current_set = list(best_set_global)
    current_size = len(current_set)
    current_bits = best_bits_global
    
    # Helper to get available bits given a set of bits
    def get_available_bits(bits):
        blocked = 0
        for v in range(1, N + 1):
            if (bits >> (v - 1)) & 1:
                blocked |= conflict[v] | (1 << (v - 1))
        return all_available & ~blocked

    # Helper to greedily fill available bits
    def greedy_fill(avail_bits):
        chosen = []
        fill_bits = 0
        for v in order_deg_asc:
            if (avail_bits >> (v - 1)) & 1:
                if (conflict[v] & fill_bits) == 0:
                    fill_bits |= (1 << (v - 1))
                    chosen.append(v)
        return chosen, fill_bits

    # Local search loop
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try removing each vertex in the current set
        for v in current_set:
            if time.time() > deadline - 2:
                break
            
            # Remove v
            temp_bits = current_bits & ~(1 << (v - 1))
            temp_set = [x for x in current_set if x != v]
            
            # Get available vertices
            avail_bits = get_available_bits(temp_bits)
            
            # Greedy fill
            added, fill_bits = greedy_fill(avail_bits)
            
            new_size = len(temp_set) + len(added)
            
            if new_size > current_size:
                current_set = temp_set + added
                current_size = new_size
                current_bits = temp_bits | fill_bits
                improved = True
                break # Restart with the improved set
    
    # Update global best
    if current_size > best_size_global:
        best_size_global = current_size
        best_set_global = set(current_set)
        best_bits_global = current_bits
    
    # Return the best set found
    result = list(best_set_global)
    if not result:
        # Fallback: return all odd numbers (should be a valid independent set)
        result = [v for v in range(1, N + 1, 2)]
    
    return result
