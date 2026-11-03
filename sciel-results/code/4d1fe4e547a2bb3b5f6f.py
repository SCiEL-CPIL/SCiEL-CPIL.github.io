import time
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312134213241234" # Naive is 33, optimal is 33.
        # Actually, let's just run the generic algorithm for n>=4 if time permits, 
        # but for n=4 naive is optimal.
    
    # Generate all permutations
    symbols = list(range(1, n + 1))
    perms = list(permutations(symbols))
    N = len(perms)
    
    # Map permutations to indices for fast lookup? 
    # We just need the list of tuples.
    
    # Precompute overlaps
    # O[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # We store this as a list of lists or a 2D array.
    # N=5040, N^2 = 25M. 
    # Storing as int8 or int16 is fine.
    # But Python lists of lists are slow.
    # We can use a flat list or numpy array.
    # Let's use a list of bytearrays or similar if possible, or just standard list.
    # Given N=5040, a list of 5040 lists of 5040 ints is ~25M ints.
    # In Python, an int is 28 bytes. 25M * 28 = 700MB. Too much?
    # Maybe.
    # We can store only the cost matrix? No, we need overlaps.
    # We can store overlaps in a NumPy array of type uint8.
    try:
        import numpy as np
        overlap_matrix = np.zeros((N, N), dtype=np.uint8)
        use_numpy = True
    except ImportError:
        overlap_matrix = [[0]*N for _ in range(N)]
        use_numpy = False
    
    # Build prefix maps for each length k
    # prefix_map[k][prefix_tuple] = list of indices
    prefix_maps = [{} for _ in range(n)] # index 0 unused, 1..n-1
    
    for k in range(1, n):
        pm = prefix_maps[k]
        for j, p in enumerate(perms):
            prefix = tuple(p[:k])
            if prefix in pm:
                pm[prefix].append(j)
            else:
                pm[prefix] = [j]
    
    # Fill overlap matrix
    for i, p in enumerate(perms):
        if use_numpy:
            row = overlap_matrix[i]
            for k in range(n - 1, 0, -1):
                suffix = tuple(p[-k:])
                if suffix in prefix_maps[k]:
                    for j in prefix_maps[k][suffix]:
                        if row[j] < k:
                            row[j] = k
        else:
            row = overlap_matrix[i]
            for k in range(n - 1, 0, -1):
                suffix = tuple(p[-k:])
                if suffix in prefix_maps[k]:
                    for j in prefix_maps[k][suffix]:
                        if row[j] < k:
                            row[j] = k
    
    # Generate naive path
    # The naive path is generated recursively.
    # We can simulate the naive construction to get the order of permutations.
    # Or we can just start with a random permutation of the indices?
    # No, starting with the naive one is a good heuristic.
    
    # Let's generate the naive order.
    # W_1 = [1]
    # Order for n=2: [12, 21]? 
    # The naive construction inserts the new element.
    # Let's just use the order they appear in the naive string.
    
    def get_naive_order(n):
        if n == 1:
            return [tuple([1])]
        prev_order = get_naive_order(n - 1)
        # The naive construction:
        # For each permutation P in prev_order:
        #   Add P + [n] + P
        # The permutations of 1..n that appear are:
        #   The windows of size n in the constructed string.
        #   This is complex to simulate directly.
        #   
        # Alternative: Just start with a random order?
        # Random order length is approx N * (n - 1) = 5040 * 6 = 30240.
        # Naive is 5913.
        # Random is much worse.
        # 
        # We need a good initial guess.
        # Let's try to construct the naive order explicitly.
        
        # The naive superpermutation for n is formed by:
        # S_n = S_{n-1} with n inserted.
        # Specifically, if S_{n-1} contains perms p1, p2, ... in order,
        # S_n is formed by blocks p1 n p1, p2 n p2, etc?
        # No, it's overlapping.
        
        # Let's just use a simple heuristic: 
        # Start with the identity permutation and greedily pick the next permutation 
        # that maximizes overlap with the current end.
        
        used = [False] * N
        path = [0] # Start with perms[0]
        used[0] = True
        current_end = perms[0]
        
        for _ in range(N - 1):
            best_j = -1
            best_overlap = 0
            # To make this fast, we can't check all N every time.
            # But N=5040. 5040*5040 = 25M checks.
            # Each check is O(1) if we have the matrix.
            # 25M checks in Python might take 5-10 seconds.
            # We have 110s. This is feasible.
            
            for j in range(N):
                if not used[j]:
                    o = overlap_matrix[0][j] if use_numpy else overlap_matrix[0][j]
                    # Wait, we need overlap with the LAST element in path.
                    # The last element is path[-1].
                    pass
            
            # This greedy approach is O(N^2) total.
            # Let's implement it properly.
            
        # Actually, let's just use the recursive naive construction to get the order.
        # It's easier to just generate the string and extract the order.
        
        word = [1]
        for m in range(2, n + 1):
            perms_prev, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_prev.append(window)
            
            new = []
            for p in perms_prev:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        
        # Extract order of permutations from word
        naive_order = []
        seen_perms = set()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if p not in seen_perms:
                seen_perms.add(p)
                # Find index of p in perms
                # We need a map from tuple to index
                pass
        
        # We need a map from tuple to index.
        perm_to_idx = {p: i for i, p in enumerate(perms)}
        
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if p in perm_to_idx and p not in seen_perms:
                seen_perms.add(p)
                naive_order.append(perm_to_idx[p])
        
        return naive_order

    naive_path = get_naive_order(n)
    
    # Calculate initial cost
    def calc_cost(path):
        cost = n
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            o = overlap_matrix[u][v] if use_numpy else overlap_matrix[u][v]
            cost += (n - o)
        return cost

    current_path = naive_path
    current_cost = calc_cost(current_path)
    best_path = current_path[:]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick a random move
        # 1. Swap adjacent
        # 2. Move one element to another position
        
        if random.random() < 0.5:
            if len(current_path) < 2: break
            i = random.randint(0, len(current_path) - 2)
            j = i + 1
            # Swap i and j
            current_path[i], current_path[j] = current_path[j], current_path[i]
            
            # Calculate delta cost
            # Affected edges: (i-1, i), (i, j), (j, j+1)
            # Original: cost(i-1, i) + cost(i, j) + cost(j, j+1)
            # New: cost(i-1, j) + cost(j, i) + cost(i, j+1)
            
            def edge_cost(u, v):
                if u is None or v is None: return 0
                o = overlap_matrix[u][v] if use_numpy else overlap_matrix[u][v]
                return n - o
            
            u1 = current_path[i-1] if i > 0 else None
            u2 = current_path[i]
            u3 = current_path[j]
            u4 = current_path[j+1] if j+1 < len(current_path) else None
            
            # Note: current_path is already swapped. 
            # So current_path[i] is u3 (old j), current_path[j] is u2 (old i).
            # We need to calculate cost based on NEW arrangement.
            
            # New edges: (u1, u3), (u3, u2), (u2, u4)
            new_cost_part = edge_cost(u1, u3) + edge_cost(u3, u2) + edge_cost(u2, u4)
            
            # Old edges: (u1, u2), (u2, u3), (u3, u4)
            # We can store old costs? Or just recalculate?
            # Recalculating is safer.
            # But we need the OLD cost of these specific edges.
            # Let's just recalculate the whole cost? No, too slow.
            
            # Let's store the previous path state?
            # Simpler: Just calculate the change.
            
            # Let's revert swap, calculate old, swap, calculate new.
            current_path[i], current_path[j] = current_path[j], current_path[i] # Revert
            
            u1 = current_path[i-1] if i > 0 else None
            u2 = current_path[i]
            u3 = current_path[j]
            u4 = current_path[j+1] if j+1 < len(current_path) else None
            
            old_cost_part = edge_cost(u1, u2) + edge_cost(u2, u3) + edge_cost(u3, u4)
            
            current_path[i], current_path[j] = current_path[j], current_path[i] # Swap back
            
            u1_new = current_path[i-1] if i > 0 else None
            u2_new = current_path[j] # This is old u3
            u3_new = current_path[i] # This is old u2
            u4_new = current_path[j+1] if j+1 < len(current_path) else None
            
            # Wait, indices i and j are swapped.
            # Path is ... u1, u3, u2, u4 ...
            new_cost_part = edge_cost(u1, u3) + edge_cost(u3, u2) + edge_cost(u2, u4)
            
            delta = new_cost_part - old_cost_part
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_path = current_path[:]
            else:
                current_path[i], current_path[j] = current_path[j], current_path[i] # Revert swap
            
        else:
            # Move element
            if len(current_path) < 2: break
            i = random.randint(0, len(current_path) - 1)
            j = random.randint(0, len(current_path) - 1)
            if i == j: continue
            
            # Move element at i to j
            # If i < j, shift i+1..j left, put at j
            # If i > j, shift j..i-1 right, put at j
            
            # This is more complex to calculate delta.
            # Let's just do full cost calculation for move?
            # No, let's stick to swaps for simplicity and speed.
            # Or implement 3-opt carefully.
            
            pass
        
        T *= cooling
    
    # Construct final string
    word = list(perms[best_path[0]])
    for idx in range(1, len(best_path)):
        u = best_path[idx-1]
        v = best_path[idx]
        o = overlap_matrix[u][v] if use_numpy else overlap_matrix[u][v]
        word.extend(perms[v][o:])
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
