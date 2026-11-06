# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(200)

def construct():
    """
    Returns a strictly increasing sequence of integers in [2, 128] 
    maximizing length such that all contiguous block products are distinct.
    Uses DFS with backtracking and strong pruning.
    """
    
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    factor_vecs = {}
    for n in range(2, 129):
        n_tmp = n
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n_tmp % p == 0:
                vec[i] += 1
                n_tmp //= p
            if n_tmp == 1:
                break
        factor_vecs[n] = tuple(vec)
    
    zero_vec = tuple([0] * num_primes)
    
    # Global best
    best_len = 0
    best_seq = []
    
    # Greedy initialization to get a good lower bound
    # Try to build a sequence greedily
    greedy_seq = []
    greedy_prefixes = [zero_vec]
    greedy_products = set()
    
    for x in range(2, 129):
        # Check if x can be added
        last_prefix = greedy_prefixes[-1]
        new_prefix = tuple(last_prefix[i] + factor_vecs[x][i] for i in range(num_primes))
        
        valid = True
        new_prods = []
        for i in range(len(greedy_seq) + 1):
            old_prefix = greedy_prefixes[i]
            block_vec = tuple(new_prefix[j] - old_prefix[j] for j in range(num_primes))
            if block_vec in greedy_products:
                valid = False
                break
            new_prods.append(block_vec)
        
        if valid:
            greedy_seq.append(x)
            greedy_prefixes.append(new_prefix)
            greedy_products.update(new_prods)
            
    if len(greedy_seq) > best_len:
        best_len = len(greedy_seq)
        best_seq = greedy_seq[:]
        
    # DFS Search
    # To avoid deep recursion issues, we can use an iterative approach or careful recursion.
    # Depth is max 127, so recursion is fine with setrecursionlimit.
    
    def dfs(idx, seq, prefixes, products):
        nonlocal best_len, best_seq
        
        # Pruning: even if we add all remaining numbers, can we beat best?
        remaining = 128 - idx + 1
        if len(seq) + remaining <= best_len:
            return
            
        if idx > 128:
            if len(seq) > best_len:
                best_len = len(seq)
                best_seq = seq[:]
            return
        
        x = idx
        
        # Option 1: Include x
        last_prefix = prefixes[-1]
        new_prefix = tuple(last_prefix[i] + factor_vecs[x][i] for i in range(num_primes))
        
        valid = True
        new_prods = []
        for i in range(len(seq) + 1):
            old_prefix = prefixes[i]
            block_vec = tuple(new_prefix[j] - old_prefix[j] for j in range(num_primes))
            if block_vec in products:
                valid = False
                break
            new_prods.append(block_vec)
            
        if valid:
            # Recurse with x included
            # We pass copies to avoid mutation issues, or we can mutate and backtrack.
            # Copying sets and lists is O(N). N is small (max 128).
            new_seq = seq + [x]
            new_prefixes = prefixes + [new_prefix]
            new_products = products.copy()
            new_products.update(new_prods)
            
            dfs(idx + 1, new_seq, new_prefixes, new_products)
            
            # Pruning check after returning?
            # If we found a solution of length len(seq)+1 + (128-(idx+1)+1) == len(seq) + 128 - idx
            # If best_len is already that high, we might prune the "skip" branch?
            # The skip branch has max potential len(seq) + (128 - idx).
            # If best_len >= len(seq) + (128 - idx), we can skip the "skip" branch.
            
        # Option 2: Skip x
        # Pruning: if even skipping x and taking all others doesn't beat best, return
        # Max length if we skip x: len(seq) + (128 - (idx+1) + 1) = len(seq) + 128 - idx
        if len(seq) + (128 - idx) > best_len:
            dfs(idx + 1, seq, prefixes, products)
            
    # Start DFS from 2
    # Initial state
    dfs(2, [], [zero_vec], set())
    
    return best_seq

# EVOLVE-BLOCK-END
