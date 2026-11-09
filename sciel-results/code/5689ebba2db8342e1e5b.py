import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256
    # conflict[a] is a bitset (integer) where bit b is 1 if a and b conflict
    # We use 0-indexed for bitsets internally, so index i in list corresponds to value i+1? 
    # No, let's map value v to index v-1.
    # conflict[i] has bit j set if (i+1) and (j+1) conflict.
    
    conflict = [0] * (N + 1) # 1-indexed for clarity in logic, but we can just use 0..N-1
    
    # Let's stick to 1-indexed values 1..N, but store conflict in a list of length N+1
    # conflict[a] is an integer. Bit (b-1) is set if a and b conflict.
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find divisors of a_sq
        # d | a_sq => d is a divisor.
        # b = d - a.
        # We need 1 <= b <= N and b != a.
        # So a < d <= a + N.
        # Also d | a_sq.
        
        # Efficient divisor finding:
        # Iterate k from 1 to sqrt(a_sq) = a
        # If a_sq % k == 0:
        #   d1 = k
        #   d2 = a_sq // k
        #   Check if d1 is in (a, a+N]
        #   Check if d2 is in (a, a+N]
        
        limit = int(math.isqrt(a_sq))
        for k in range(1, limit + 1):
            if a_sq % k == 0:
                d1 = k
                d2 = a_sq // k
                
                for d in (d1, d2):
                    if a < d <= a + N:
                        b = d - a
                        if 1 <= b <= N and b != a:
                            # Set bit for b in conflict[a]
                            # Bit index is b-1
                            conflict[a] |= (1 << (b - 1))
                            # Set bit for a in conflict[b]
                            conflict[b] |= (1 << (a - 1))
    
    # Helper: check if v can be added to a set represented by a bitset `current_bits`
    # current_bits has bit i set if value i+1 is in the set.
    # We need to check if any neighbor of v is in the set.
    # Neighbors of v are given by conflict[v].
    # If (conflict[v] & current_bits) == 0, then v can be added.
    
    def can_add(v, current_bits):
        return (conflict[v] & current_bits) == 0
    
    def greedy_independent_set(order):
        chosen_bits = 0
        chosen_list = []
        for v in order:
            if can_add(v, chosen_bits):
                chosen_bits |= (1 << (v - 1))
                chosen_list.append(v)
        return chosen_bits, chosen_list
    
    all_nodes = list(range(1, N + 1))
    
    # Precompute degrees for sorting
    # degree of v is popcount(conflict[v])
    # Python 3.8+ has int.bit_count()
    degrees = [v.bit_count() for v in conflict] # index 0 is unused, conflict[0] is 0
    
    best_bits = 0
    best_size = 0
    best_list = []
    
    # Strategy 1: Greedy by degree (ascending)
    order_by_degree = sorted(all_nodes, key=lambda x: degrees[x])
    bits, lst = greedy_independent_set(order_by_degree)
    if len(lst) > best_size:
        best_size = len(lst)
        best_bits = bits
        best_list = lst
    
    # Strategy 2: Random restarts with local search
    random.seed(42)
    
    def local_search(bits, size, max_iters=20):
        # Try to improve by removing one vertex and adding others
        # We represent the set as a list for easy removal, and bits for fast checks
        current_list = [v for v in range(1, N+1) if (bits >> (v-1)) & 1]
        current_bits = bits
        current_size = size
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters and time.time() < deadline - 5:
            improved = False
            iter_count += 1
            # Try removing each vertex in current_list
            for i, v in enumerate(current_list):
                if time.time() > deadline - 5:
                    break
                # Remove v
                temp_bits = current_bits & ~(1 << (v - 1))
                temp_list = current_list[:i] + current_list[i+1:]
                
                # Try to add any vertex from the complement
                # To speed this up, we can iterate through all nodes not in temp_bits
                # But N=256 is small, so O(N) per removal is fine.
                # Total O(N^2) per pass.
                
                added_any = False
                # We want to find a vertex w not in temp_bits that can be added.
                # Actually, we want to find a sequence of additions that increases size.
                # Simple heuristic: Try adding the vertex with the lowest degree among available ones.
                # Or just try all available vertices.
                
                # Let's try all vertices w in 1..N
                for w in all_nodes:
                    if time.time() > deadline - 5:
                        break
                    if (temp_bits >> (w - 1)) & 1:
                        continue
                    if can_add(w, temp_bits):
                        # We found one. But can we find a BETTER one?
                        # Let's just take the first one and see if we can add more.
                        # Actually, to maximize gain, we should try to add as many as possible.
                        # Let's do a greedy fill on the temp_set.
                        new_bits = temp_bits
                        new_list = temp_list[:]
                        # Greedy fill
                        # Sort available candidates by degree?
                        # Just iterate all_nodes
                        for u in all_nodes:
                            if time.time() > deadline - 5:
                                break
                            if (new_bits >> (u - 1)) & 1:
                                continue
                            if can_add(u, new_bits):
                                new_bits |= (1 << (u - 1))
                                new_list.append(u)
                        
                        if len(new_list) > current_size:
                            current_bits = new_bits
                            current_list = new_list
                            current_size = len(new_list)
                            improved = True
                            break # Restart the outer loop with new set
                if improved:
                    break
                    
        return current_bits, current_list, current_size

    # Run multiple random greedy initializations followed by local search
    # Limit restarts to fit in time
    max_restarts = 100
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        random_order = all_nodes[:]
        random.shuffle(random_order)
        
        # Weighted random? No, just shuffle.
        bits, lst = greedy_independent_set(random_order)
        
        # Local search
        new_bits, new_list, new_size = local_search(bits, len(lst))
        
        if new_size > best_size:
            best_size = new_size
            best_bits = new_bits
            best_list = new_list
            
    # One more pass with degree-sorted order + local search
    bits, lst = greedy_independent_set(order_by_degree)
    new_bits, new_list, new_size = local_search(bits, len(lst))
    if new_size > best_size:
        best_size = new_size
        best_bits = new_bits
        best_list = new_list

    return best_list
