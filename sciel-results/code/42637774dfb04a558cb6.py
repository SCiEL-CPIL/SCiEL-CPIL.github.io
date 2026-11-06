import time

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # 2s safety margin for 90s limit
    
    # Global variables for DFS
    best_seq = []
    best_len = 0
    nodes_visited = 0

    def fast_greedy():
        values = []
        products = set()
        for x in range(2, 129):
            # Calculate new suffix products if x is appended
            # Suffixes are x, x*last, x*last*second_last, etc.
            new_sufs = [x]
            prod = x
            valid = True
            for a in reversed(values):
                prod *= a
                if prod in products:
                    valid = False
                    break
                new_sufs.append(prod)
            
            if not valid:
                continue
                
            # Check if any new suffix collides with existing products
            # Note: new_sufs are distinct from each other for strictly increasing seq >= 2
            if not products.intersection(new_sufs):
                values.append(x)
                products.update(new_sufs)
        
        return values

    # Initialize with greedy result
    init_seq = fast_greedy()
    best_len = len(init_seq)
    best_seq = init_seq
    
    if best_len == 127: # Max possible
        return best_seq

    # Precompute primes? Not strictly necessary for the set logic, but might help heuristics.
    # We will stick to the set-based collision check.

    def dfs(seq, prods, sufs, next_val):
        global best_seq, best_len, nodes_visited, deadline, start_time
        
        nodes_visited += 1
        if nodes_visited % 10000 == 0:
            if time.time() > deadline:
                return

        # Update best
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)
            
        # Pruning: Can we possibly beat best_len?
        # Max additional elements is (128 - next_val + 1)
        if len(seq) + (128 - next_val + 1) <= best_len:
            return

        if next_val > 128:
            return

        # Time check
        if time.time() > deadline:
            return

        x = next_val
        
        # Option 1: Include x
        # Calculate new suffixes
        new_sufs = [x]
        prod = x
        valid = True
        for a in reversed(seq):
            prod *= a
            if prod in prods:
                valid = False
                break
            new_sufs.append(prod)
            
        if valid and not prods.intersection(new_sufs):
            new_prods = prods.union(new_sufs)
            # To avoid modifying the original list, we pass a copy or use a stack
            # Since recursion depth is max 127, list copying is O(N) which is fine.
            new_seq = seq + [x]
            dfs(new_seq, new_prods, new_sufs, x + 1)
            
        # Option 2: Exclude x
        dfs(seq, prods, sufs, x + 1)

    # Start DFS
    # We start from an empty sequence. 
    # However, starting from empty might explore too many branches.
    # We can seed the DFS with the greedy sequence's prefix to focus search?
    # No, that biases the search.
    # Let's just run DFS from empty. The "Include First" heuristic will quickly build up a long sequence,
    # setting a high best_len, which then prunes the "Exclude" branches effectively.
    
    # Optimization: Start DFS from the greedy sequence?
    # If we start from greedy, we only search for extensions or modifications?
    # The problem is we might need to swap elements.
    # Standard DFS from scratch is safer.
    
    # To speed up initial exploration, we can force the first few elements to match greedy?
    # Let's just run the standard DFS.
    
    # Initial state
    initial_seq = []
    initial_prods = set()
    initial_sufs = []
    
    # One potential issue: DFS might be too slow if it explores too many "exclude" branches.
    # But the pruning `len(seq) + remaining <= best_len` is very strong once best_len is high.
    # The greedy gives a high best_len immediately.
    
    dfs(initial_seq, initial_prods, initial_sufs, 2)
    
    return best_seq

if __name__ == "__main__":
    res = construct()
    print(len(res))
