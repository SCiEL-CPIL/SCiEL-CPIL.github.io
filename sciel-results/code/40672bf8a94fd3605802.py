import time
import sys

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # 1. Build Conflict Graph
    # adj_mask[i] is a bitmask of vertices that conflict with i
    adj_mask = [0] * (N + 1)
    for a in vertices:
        for b in vertices:
            if a == b: continue
            if (a * b) % (a + b) == 0:
                adj_mask[a] |= (1 << (b - 1))
    
    # 2. Greedy Heuristic for Initial Lower Bound
    # We want a good independent set.
    # Sort by degree ascending (low degree first) is often good for greedy independent set?
    # Or high degree? 
    # Let's try a few greedy strategies.
    
    def greedy_mis(order):
        U = (1 << N) - 2  # bits 1 to N set (0-indexed: 0 to N-1)
        # Actually, let's use 0-based indexing for bits.
        # Vertex i (1..N) corresponds to bit i-1.
        # So U = sum(1 << (v-1) for v in vertices) = (1<<N) - 2 is wrong because bit 0 is v=1.
        # (1<<N) - 1 has bits 0..N-1 set.
        # We want vertices 1..N. So bits 0..N-1 are all valid.
        # So initial U = (1 << N) - 1.
        
        selected = 0
        U = (1 << N) - 1
        # Iterate through order
        for v in order:
            bit = 1 << (v - 1)
            if U & bit:
                # Select v
                selected |= bit
                # Remove v and its neighbors from U
                U &= ~(adj_mask[v] | bit)
        return selected

    # Strategy 1: Sort by degree ascending
    degs = [bin(adj_mask[v]).count('1') for v in vertices]
    order1 = sorted(vertices, key=lambda v: degs[v-1])
    set1 = greedy_mis(order1)
    size1 = bin(set1).count('1')
    
    # Strategy 2: Sort by degree descending
    order2 = sorted(vertices, key=lambda v: -degs[v-1])
    set2 = greedy_mis(order2)
    size2 = bin(set2).count('1')
    
    # Strategy 3: Random
    import random
    rng = random.Random(42)
    best_greedy_set = set1 if size1 > size2 else set2
    best_greedy_size = max(size1, size2)
    
    for _ in range(10):
        order3 = vertices[:]
        rng.shuffle(order3)
        set3 = greedy_mis(order3)
        size3 = bin(set3).count('1')
        if size3 > best_greedy_size:
            best_greedy_size = size3
            best_greedy_set = set3
            
    best_size = best_greedy_size
    best_set_mask = best_greedy_set

    # 3. Tomita Branch and Bound
    # We need to track time.
    def count_bits(x):
        return bin(x).count('1')
        
    # Greedy Coloring Bound for a set U (bitmask)
    # Returns upper bound on independence number of U
    def greedy_bound(U):
        if U == 0:
            return 0
        color = 0
        u = U
        while u:
            color += 1
            # Pick a vertex from u. 
            # To make it deterministic and fast, pick the lowest set bit?
            # Or pick a vertex with high degree in u?
            # For simplicity and speed in Python, picking lowest set bit is O(1) to find.
            # However, the quality of the bound depends on the order.
            # A better bound is obtained if we color using a specific order.
            # Standard Tomita uses a specific ordering.
            # Let's just pick the lowest index vertex available.
            v_idx = (u & -u).bit_length() - 1 # 0-based index
            v = v_idx + 1
            # Remove v and its neighbors from u
            u &= ~(adj_mask[v] | (1 << v_idx))
        return color

    # Search
    # U: bitmask of candidates
    # current_size: size of current independent set
    # current_mask: bitmask of current independent set
    # best_size, best_set_mask are non-local variables (using closure or class)
    
    # To allow early termination, we use a mutable list or nonlocal
    # We'll use a class-like structure or just nested functions with nonlocal.
    
    # Optimization: Precompute bit positions?
    # We can iterate bits using:
    # while U:
    #   v = (U & -U).bit_length() - 1
    #   U &= U - 1
    
    # Let's implement the search.
    # Recursion depth up to 256.
    
    def search(U, current_size, current_mask):
        nonlocal best_size, best_set_mask
        if time.time() > DEADLINE:
            raise TimeoutError()
            
        if U == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Bound
        bound = greedy_bound(U)
        if current_size + bound <= best_size:
            return
            
        # Pick vertex v from U
        # Heuristic: Pick vertex with highest degree in U to reduce U quickly?
        # Or just pick the first one?
        # Tomita's original algorithm picks vertices based on a specific ordering (Coloring order).
        # Let's stick to picking the lowest index for simplicity, but maybe sorting U by degree helps.
        # Given Python speed, iterating all bits to find max degree might be slow.
        # Let's try picking the vertex with the highest degree in the *original* graph first?
        # No, dynamic degree is better.
        
        # Let's try a simple heuristic: Pick the vertex v in U that has the most neighbors in U.
        # If U has k bits, this takes O(k * cost_of_bit_op).
        # For k=256, this is fine.
        
        # Find max degree vertex in U
        u_temp = U
        best_v_idx = -1
        max_deg = -1
        while u_temp:
            v_idx = (u_temp & -u_temp).bit_length() - 1
            u_temp &= u_temp - 1
            v = v_idx + 1
            # Degree of v in U is count of bits in (U & adj_mask[v])
            deg = count_bits(U & adj_mask[v])
            if deg > max_deg:
                max_deg = deg
                best_v_idx = v_idx
                
        v = best_v_idx + 1
        bit = 1 << best_v_idx
        
        # Branch 1: Include v
        # New U is U without v and its neighbors
        new_U_include = U & ~(adj_mask[v] | bit)
        search(new_U_include, current_size + 1, current_mask | bit)
        
        # Branch 2: Exclude v
        # New U is U without v
        new_U_exclude = U & ~bit
        search(new_U_exclude, current_size, current_mask)

    # Start search with all vertices
    # We can order the initial U? The search handles the set U.
    # The order of branching is determined by the vertex selection heuristic.
    
    try:
        search((1 << N) - 1, 0, 0)
    except TimeoutError:
        pass
    except RecursionError:
        # Fallback if recursion fails, though 256 is safe
        pass

    # Convert best_set_mask to list of integers
    result = []
    for i in range(N):
        if best_set_mask & (1 << i):
            result.append(i + 1)
            
    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Invalid, fallback to greedy
                return [i+1 for i in range(N) if best_greedy_set & (1<<i)]
                
    return result
