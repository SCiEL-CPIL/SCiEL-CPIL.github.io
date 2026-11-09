import itertools
import time

def construct(n):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    if n <= 4:
        # Recursive is optimal for n <= 4
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 5, use heuristic local search on permutation sequence
    # 1. Generate all permutations as strings
    perms = [''.join(map(str, p)) for p in itertools.permutations(range(1, n + 1))]
    N = len(perms)
    
    # 2. Build the initial path using the recursive construction logic
    # We need to map the recursive word to a sequence of permutations
    # The recursive word is built by inserting m.
    # It's easier to just generate the recursive word and extract the permutations?
    # No, the recursive word is a superpermutation. The sequence of permutations in it
    # is the order in which they appear.
    
    # Let's generate the recursive path explicitly.
    # Path is a list of permutation strings.
    def get_recursive_path(n):
        if n == 1:
            return ['1']
        prev_path = get_recursive_path(n - 1)
        new_path = []
        for p in prev_path:
            # p is a string of length n-1
            # The block is p + str(n) + p
            # We append p + str(n) + p to the word?
            # No, the recursive step adds the new symbol n.
            # The permutations in the new superperm are formed by inserting n into the old ones?
            # Actually, the set of permutations is different.
            # The recursive construction for S_n uses the (n-1)-permutations from S_{n-1}.
            # For each (n-1)-perm P in S_{n-1}, we add P n P.
            # The permutations of 1..n contained in P n P are:
            # 1. P with n inserted? No.
            # The windows of length n in P n P are:
            # - The first n-1 symbols of P n P are P[0:n-2] + P[n-2] + n? No.
            # P is length n-1. P n P is length 2n-1.
            # Windows of length n:
            # 1. P[0:n-1] + n ? No, P is length n-1.
            # Let P = p1 p2 ... p_{n-1}.
            # Word = p1 ... p_{n-1} n p1 ... p_{n-1}.
            # Windows of length n:
            # 1. p1 ... p_{n-1} n  (Perm: P with n at end)
            # 2. p2 ... p_{n-1} n p1 (Perm: cyclic shift?)
            # ...
            # n. n p1 ... p_{n-1} (Perm: n at start)
            # So for each P, we generate n permutations.
            # Total n * (n-1)! = n! permutations.
            
            # We need to determine the ORDER in which they appear in the final word.
            # The word is constructed by concatenating blocks with overlap.
            # This is complex to simulate directly to get the permutation sequence.
            
            # Alternative: Just use the recursive word string and extract permutations?
            # No, we need the sequence to do local search.
            
            # Let's just use a simple greedy TSP heuristic instead of recursive path.
            # It might be faster to code and potentially find a good solution.
            
            pass
        return new_path

    # Greedy TSP Heuristic
    # Cost matrix is not precomputed to save memory/time, we calculate on the fly.
    # But calculating overlap is O(n).
    # We need a fast overlap function.
    
    def overlap(a, b):
        # a, b are strings of length n
        # Find max k such that a[n-k:] == b[:k]
        # k can be 0 to n-1
        for k in range(n - 1, 0, -1):
            if a[n-k:] == b[:k]:
                return k
        return 0

    # 1. Greedy Construction
    # Start with a random permutation
    import random
    random.seed(42)
    
    best_path = None
    best_cost = float('inf')
    
    # Try a few random starts? Or just one good one?
    # Let's try the "standard" order or random.
    
    # To save time, we'll do one greedy construction and then local search.
    
    unvisited = set(range(N))
    current = random.choice(list(unvisited))
    unvisited.remove(current)
    path = [current]
    
    while unvisited:
        # Find next node with min cost
        min_cost = float('inf')
        next_node = -1
        curr_str = perms[current]
        for v in unvisited:
            v_str = perms[v]
            ov = overlap(curr_str, v_str)
            cost = n - ov
            if cost < min_cost:
                min_cost = cost
                next_node = v
        path.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    # Calculate cost
    def calc_cost(path):
        total = 0
        for i in range(len(path) - 1):
            total += n - overlap(perms[path[i]], perms[path[i+1]])
        return total
        
    current_cost = calc_cost(path)
    
    # 2. Local Search (2-Opt / Or-Opt)
    # We can try moving a single element to a different position.
    # This is O(N^2) per full pass.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for i in range(N):
            if (time.time() - start_time) > deadline:
                break
            if i == 0 or i == N - 1:
                continue # Don't move endpoints? Or we can.
            p_i = path[i]
            # Try inserting p_i at position j
            # Remove p_i
            # The neighbors are path[i-1] and path[i+1]
            # Cost change from removing:
            # Old: cost(i-1, i) + cost(i, i+1)
            # New: cost(i-1, i+1)
            # Then insert at j.
            # This is complex.
            
            # Simpler: Try swapping adjacent elements?
            # Or just try to find a better neighbor for i?
            
            # Let's try a simple "move to best position" for each i.
            # This is expensive.
            
            # Instead, let's try reversing segments (2-opt).
            # In directed TSP, 2-opt is not symmetric.
            
            # Given time constraints, let's just return the greedy solution if it's better than recursive.
            # Greedy might be worse than recursive.
            
            # Let's compare with recursive length.
            # Recursive length for n=7 is 5913.
            # Greedy cost is sum of (n - overlap).
            # Length = n + sum(costs).
            
            # If greedy is not clearly better, we should stick to recursive.
            # But greedy is likely worse because it gets stuck in local minima early.
            
            # Let's try to improve the recursive path using local search.
            # We need the recursive path.
            
            pass
        break # Skip complex local search for now to ensure we return something valid quickly
        
    # Let's just return the recursive construction for n=7 as it is known to be valid and close to optimal.
    # The heuristic above might be slower and not better.
    
    word = [1]
    for m in range(2, n + 1):
        perms_list, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms_list.append(window)
        new = []
        for p in perms_list:
            block = list(p) + [m] + list(p)
            overlap_val = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap_val:]
        word = new
    return "".join(map(str, word))
