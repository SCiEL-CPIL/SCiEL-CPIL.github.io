import itertools
import time
import random

def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # For n > 1, we use the local search on permutation order
    # to maximize overlaps.
    
    perms = list(itertools.permutations(range(1, n + 1)))
    num_perms = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlaps
    # overlap[u][v] is the largest k such that perms[u][-k:] == perms[v][:k]
    # k can be 0 to n-1.
    # Note: In the superpermutation construction, we append P [n] P.
    # The overlap is between the suffix of the previous block and the prefix of the current block.
    # Block u ends with P_u. Block v starts with P_v.
    # So we need overlap between P_u and P_v.
    
    overlaps = [[0] * num_perms for _ in range(num_perms)]
    
    # Optimization: Use tuples for fast comparison
    # Precompute prefixes and suffixes?
    # For n=7, n-1=6.
    # We can just compare slices.
    
    for u in range(num_perms):
        pu = perms[u]
        for v in range(num_perms):
            pv = perms[v]
            # Max overlap is n-1
            for k in range(n - 1, -1, -1):
                if k == 0:
                    overlaps[u][v] = 0
                    break
                if pu[-k:] == pv[:k]:
                    overlaps[u][v] = k
                    break

    # Initial order: Recursive construction order
    # We need to generate the order used by the recursive construction.
    # The recursive construction order is the order in which permutations appear in S_{n-1}.
    # For n=7, we can generate S_6 recursively and extract the order.
    
    def get_recursive_order(m):
        if m == 1:
            return [(1,)]
        
        S = [1]
        for mm in range(2, m + 1):
            seen = set()
            perms_order = []
            for i in range(len(S) - (mm - 1) + 1):
                window = tuple(S[i:i + mm - 1])
                if window not in seen:
                    seen.add(window)
                    perms_order.append(window)
                    
            new_S = []
            for P in perms_order:
                block = list(P) + [mm] + list(P)
                max_possible_overlap = min(len(new_S), len(block) - 1)
                overlap = 0
                for k in range(max_possible_overlap, -1, -1):
                    if k == 0:
                        overlap = 0
                        break
                    if new_S[-k:] == block[:k]:
                        overlap = k
                        break
                new_S.extend(block[overlap:])
            S = new_S
        
        # Extract order of permutations of length m from S
        seen = set()
        final_order = []
        for i in range(len(S) - m + 1):
            window = tuple(S[i:i + m])
            if window not in seen:
                seen.add(window)
                final_order.append(window)
        return final_order

    initial_order = get_recursive_order(n)
    # Convert to indices
    order = [perm_to_idx[p] for p in initial_order]
    
    # Calculate initial total overlap
    total_overlap = 0
    for i in range(len(order) - 1):
        total_overlap += overlaps[order[i]][order[i+1]]
    
    best_order = order[:]
    best_overlap = total_overlap
    
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s for output generation
    
    # Local Search
    # Try adjacent swaps
    # To speed up, we can randomize the starting point or use a simulated annealing-like approach
    # But simple hill climbing with random restarts or multiple passes might work.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle the order of checks to avoid bias
        indices = list(range(len(order) - 1))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try swapping order[i] and order[i+1]
            u = order[i]
            v = order[i+1]
            
            # Current overlap contribution involving u, v
            # Prev: order[i-1] -> u
            # Mid: u -> v
            # Next: v -> order[i+2]
            
            old_mid = overlaps[u][v]
            
            # New arrangement: ... order[i-1], v, u, order[i+2] ...
            
            new_mid = overlaps[v][u]
            
            delta = new_mid - old_mid
            
            # Check boundary conditions
            if i > 0:
                prev = order[i-1]
                old_prev = overlaps[prev][u]
                new_prev = overlaps[prev][v]
                delta += new_prev - old_prev
            else:
                # No previous node
                pass
                
            if i + 2 < len(order):
                next_node = order[i+2]
                old_next = overlaps[v][next_node]
                new_next = overlaps[u][next_node]
                delta += new_next - old_next
            else:
                pass
                
            if delta > 0:
                # Perform swap
                order[i], order[i+1] = v, u
                total_overlap += delta
                improved = True
                
                if total_overlap > best_overlap:
                    best_overlap = total_overlap
                    best_order = order[:]
                    
    # Reconstruct the string using best_order
    S = []
    for idx in best_order:
        P = perms[idx]
        block = list(P) + [n] + list(P)
        
        # Find overlap with end of S
        # We know the overlaps from the matrix, but we need the actual string overlap
        # The matrix gives the max possible overlap based on perms.
        # The actual overlap is min(matrix_overlap, len(S))
        # But since we are building the string, we can just check.
        # However, using the matrix is faster if we trust it.
        # The matrix overlap is between P_prev and P_curr.
        # The block starts with P_curr.
        # So the overlap is exactly overlaps[prev_idx][curr_idx] if prev_idx exists.
        
        if not S:
            S.extend(block)
        else:
            prev_idx = best_order[best_order.index(idx) - 1] if idx != best_order[0] else None
            # This index lookup is O(N). We should track it.
            # Let's just do the string check, it's O(n) and n=7.
            max_possible_overlap = min(len(S), len(block) - 1)
            overlap = 0
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if S[-k:] == block[:k]:
                    overlap = k
                    break
            S.extend(block[overlap:])
            
    return "".join(map(str, S))
