import math
import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    # Step 1: Build the classic recursive superpermutation
    def build_classic(n):
        if n == 1:
            return [1]
        prev = build_classic(n - 1)
        # The standard construction S_n is formed by taking S_{n-1} 
        # and for each permutation pi in S_{n-1}, we have a block.
        # Actually, the recursive definition is:
        # S_n = S_{n-1} with n inserted?
        # Let's use the explicit construction from the seed.
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next((o for o in range(min(len(new), len(block)), -1, -1)
                               if o == 0 or new[len(new) - o:] == block[:o]), 0)
                new += block[overlap:]
            word = new
        return word

    def all_perms_present(word, n):
        target = set(permutations(range(1, n + 1)))
        found = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i + n])
            found.add(w)
            if len(found) == len(target):
                return True
        return False

    if n <= 4:
        # For small n, standard is optimal or close, just return it
        word = build_classic(n)
        return "".join(map(str, word))

    # For n >= 5, try to improve
    word = build_classic(n)
    
    # Extract permutation sequence
    perms_in_order = []
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i + n])
        if len(set(w)) == n and w not in seen:
            seen.add(w)
            perms_in_order.append(w)
            
    # If we didn't find all, return standard
    if len(perms_in_order) < math.factorial(n):
        return "".join(map(str, word))

    # Precompute distances? 
    # Let's use a heuristic distance function for the search
    # Distance = number of positions where pi[i] != sigma[i]? No, that's not shift distance.
    # Let's use a cached BFS distance.
    
    dist_cache = {}
    
    def get_dist(pi, sigma):
        key = (pi, sigma)
        if key in dist_cache:
            return dist_cache[key]
        
        # BFS from pi to sigma
        # Limit depth to 10 to keep it fast
        max_depth = 10
        queue = [(pi, 0)]
        visited = {pi}
        while queue:
            node, depth = queue.pop(0)
            if depth > max_depth:
                continue
            if node == sigma:
                dist_cache[key] = depth
                return depth
            # Generate neighbors
            # node is a tuple
            # neighbor = node[1:] + (x,)
            for x in range(1, n + 1):
                neighbor = node[1:] + (x,)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        dist_cache[key] = max_depth + 1 # Penalty
        return max_depth + 1

    # Local Search: Simulated Annealing
    # State: list of permutations
    # Cost: sum of distances between consecutive permutations
    
    current_perms = perms_in_order[:]
    current_cost = sum(get_dist(current_perms[i], current_perms[i+1]) for i in range(len(current_perms)-1))
    
    best_perms = current_perms[:]
    best_cost = current_cost
    
    T = 100.0
    cooling = 0.999
    
    # Number of iterations
    # We have ~140 seconds.
    # Each iteration involves checking a swap.
    # Let's run as many as possible.
    
    iterations = 0
    while time.time() - start_time < DEADLINE - 10.0:
        iterations += 1
        if iterations % 10000 == 0 and T < 1.0:
            break
            
        # Pick a random swap
        i = random.randint(0, len(current_perms) - 2)
        j = random.randint(i + 1, len(current_perms) - 1)
        
        # Calculate delta cost
        # Remove i, j and reinsert
        # This is complex. Let's just do simple swaps of adjacent elements first?
        # Or swap two random elements.
        
        # Let's try swapping two random elements i and j
        pi_i = current_perms[i]
        pi_j = current_perms[j]
        
        # Calculate cost change
        # Old cost involves:
        # ... -> A -> pi_i -> B ...
        # ... -> C -> pi_j -> D ...
        # New cost:
        # ... -> A -> pi_j -> B ...
        # ... -> C -> pi_i -> D ...
        
        # We need to get neighbors
        def get_neighbors(idx):
            prev = current_perms[idx-1] if idx > 0 else None
            next_ = current_perms[idx+1] if idx < len(current_perms)-1 else None
            return prev, next_
        
        prev_i, next_i = get_neighbors(i)
        prev_j, next_j = get_neighbors(j)
        
        old_cost = 0
        new_cost = 0
        
        if prev_i is not None:
            old_cost += get_dist(prev_i, pi_i)
            new_cost += get_dist(prev_i, pi_j)
        if next_i is not None:
            old_cost += get_dist(pi_i, next_i)
            new_cost += get_dist(pi_j, next_i)
            
        if prev_j is not None:
            old_cost += get_dist(prev_j, pi_j)
            new_cost += get_dist(prev_j, pi_i)
        if next_j is not None:
            old_cost += get_dist(pi_j, next_j)
            new_cost += get_dist(pi_i, next_j)
            
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_perms[i], current_perms[j] = current_perms[j], current_perms[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_perms = current_perms[:]
        
        T *= cooling

    # Reconstruct word from best_perms
    def reconstruct_word(perms, n):
        if not perms:
            return []
        word = list(perms[0])
        for i in range(1, len(perms)):
            pi = perms[i-1]
            sigma = perms[i]
            # Find shortest path from pi to sigma
            # Use BFS again
            queue = [(pi, [pi])]
            visited = {pi}
            found_path = None
            while queue:
                node, path = queue.pop(0)
                if node == sigma:
                    found_path = path
                    break
                for x in range(1, n + 1):
                    neighbor = node[1:] + (x,)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            if found_path:
                # Append the last elements of the path
                # The path is a sequence of permutations (or states)
                # We only need to append the last symbol of each transition
                # Actually, the word is formed by the first permutation, then the appended symbols.
                # If path is P0, P1, ..., Pk
                # P0 is already in word.
                # P1 = P0[1:] + [x1] -> append x1
                # P2 = P1[1:] + [x2] -> append x2
                for k in range(1, len(found_path)):
                    # The last symbol of found_path[k] is the appended symbol
                    word.append(found_path[k][-1])
            else:
                # Fallback: just append sigma[1:]? No, that's not a valid transition.
                # If no path found (should not happen with limit 10), just append sigma
                word.extend(sigma)
                
        return word

    improved_word = reconstruct_word(best_perms, n)
    
    if all_perms_present(improved_word, n) and len(improved_word) < len(word):
        return "".join(map(str, improved_word))
    
    return "".join(map(str, word))

