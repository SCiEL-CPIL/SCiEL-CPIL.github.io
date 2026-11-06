import math

def construct():
    n_max = 256
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    # 1. Identify self-conflicting nodes (2*x is a square)
    # 2. Build conflict masks for valid nodes
    # 3. Solve Max Independent Set using Bitmask Heuristic + Local Search
    
    # Precompute self-conflicts: nodes that cannot be in the set because 2*node is a square
    self_conflict = [False] * (n_max + 1)
    for i in range(1, n_max + 1):
        if is_perfect_square(2 * i):
            self_conflict[i] = True

    # Precompute conflict masks
    # conflict_mask[i] is an integer where bit j is set if i and j conflict (i+j is square)
    # We only care about j > i for building, but for checking, we check against current set S.
    conflict_mask = [0] * (n_max + 1)
    for i in range(1, n_max + 1):
        if self_conflict[i]:
            continue
        mask = 0
        for j in range(1, n_max + 1):
            if i != j and is_perfect_square(i + j):
                mask |= (1 << j)
        conflict_mask[i] = mask

    # Valid nodes are those not self-conflicting
    valid_nodes = [i for i in range(1, n_max + 1) if not self_conflict[i]]
    
    # Heuristic: Greedy on Bitmask (Min-Degree)
    def greedy_mis_bitmask():
        remaining = set(valid_nodes)
        selected_mask = 0
        while remaining:
            # Find node in remaining with minimum conflicts with other nodes in remaining
            best_node = None
            min_deg = float('inf')
            
            # Calculate bitmask of remaining nodes
            rem_mask = 0
            for v in remaining:
                rem_mask |= (1 << v)
                
            for v in remaining:
                # Degree is count of conflicts with other nodes in 'remaining'
                neigh = conflict_mask[v] & rem_mask
                deg = bin(neigh).count('1')
                if deg < min_deg:
                    min_deg = deg
                    best_node = v
            
            if best_node is None:
                break
            selected_mask |= (1 << best_node)
            # Remove best_node and its neighbors from remaining
            neighbors = conflict_mask[best_node] | (1 << best_node)
            new_remaining = set()
            for v in remaining:
                if not (neighbors & (1 << v)):
                    new_remaining.add(v)
            remaining = new_remaining
            
        return selected_mask

    initial_mask = greedy_mis_bitmask()
    
    # Local Search
    def local_search(start_mask):
        current_mask = start_mask
        current_size = bin(current_mask).count('1')
        best_mask = current_mask
        best_size = current_size
        
        for _ in range(50000): # Iterations
            improved = False
            
            # 1. Try to add a node
            # A node v can be added if:
            # a) v is valid (not self-conflict)
            # b) v is not in current_mask
            # c) (conflict_mask[v] & current_mask) == 0
            for v in valid_nodes:
                if (current_mask & (1 << v)) == 0:
                    if (conflict_mask[v] & current_mask) == 0:
                        current_mask |= (1 << v)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        improved = True
                        break 
                        
            if improved:
                continue
                
            # 2. Try 1-swap: Remove u, add v
            # Condition: v not in S, (conflict_mask[v] & S) is exactly {u}
            found_swap = False
            for v in valid_nodes:
                if (current_mask & (1 << v)) == 0:
                    conflicts_with_S = conflict_mask[v] & current_mask
                    if conflicts_with_S == 0:
                        continue # Should have been added
                    # Check if conflicts_with_S is a power of 2 (single bit)
                    if conflicts_with_S & (conflicts_with_S - 1) == 0:
                        # Single conflict u
                        u_bit = conflicts_with_S
                        # Swap u with v
                        current_mask = (current_mask ^ u_bit) | (1 << v)
                        found_swap = True
                        break
                        
            if found_swap:
                continue

            # 3. Kick: Remove a node, then greedily add as many as possible
            # This helps escape local minima
            bits = []
            temp = current_mask
            while temp:
                lsb = temp & -temp
                bits.append(lsb)
                temp ^= lsb
                
            if bits:
                # Pick a random bit to remove (deterministic for reproducibility in seed, but random is better for search)
                # Using middle bit for simplicity
                remove_bit = bits[len(bits)//2]
                new_mask = current_mask ^ remove_bit
                # Greedily add
                for v in valid_nodes:
                    if (new_mask & (1 << v)) == 0:
                        if (conflict_mask[v] & new_mask) == 0:
                            new_mask |= (1 << v)
                new_size = bin(new_mask).count('1')
                if new_size > best_size:
                    best_size = new_size
                    best_mask = new_mask
                # Adopt if size is same or better to continue search
                if new_size >= current_size:
                    current_mask = new_mask
                    current_size = new_size

        return best_mask

    best_mask = local_search(initial_mask)
    
    # Convert bitmask to list
    result = []
    for i in range(1, n_max + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    # Final verification
    def is_valid(s):
        for i in range(len(s)):
            for j in range(i, len(s)):
                if is_perfect_square(s[i] + s[j]):
                    return False
        return True

    if not is_valid(result):
        # Fallback to simple greedy
        result = []
        for i in range(1, n_max + 1):
            if self_conflict[i]: continue
            ok = True
            for j in result:
                if is_perfect_square(i + j):
                    ok = False
                    break
            if ok:
                result.append(i)
                
    return sorted(result)
