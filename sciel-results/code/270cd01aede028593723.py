import itertools

def construct(n):
    if n <= 1:
        return '1' * n
    
    symbols = list(range(1, n + 1))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute prefixes for fast overlap check?
    # Or just use the suffix of the current word.
    
    uncovered = set(range(num_perms))
    word = []
    
    # To speed up overlap finding, we can maintain a set of suffixes of uncovered perms?
    # But the current suffix changes.
    # The seed's approach of scanning k is fine.
    
    # Optimization: Group uncovered perms by their first k elements?
    # This is dynamic.
    
    # Let's stick to the simple scan, but optimized.
    
    def get_current_suffix(k):
        if k > len(word):
            return None
        return tuple(word[-k:])
        
    while uncovered:
        best_perm_idx = -1
        best_overlap = 0
        best_new_covers = 0 # For tie breaking
        
        # Find max overlap
        for k in range(n - 1, 0, -1):
            if len(word) < k:
                continue
            suffix = tuple(word[-k:])
            # Find all uncovered perms starting with this suffix
            # We need a map from prefix to list of uncovered perm indices.
            # This map changes as we remove perms.
            # Maintaining this map dynamically is O(N) per removal.
            # Total O(N^2). 5040^2 = 25M. Feasible.
            
            # Let's maintain a dict: prefix_to_uncovered
            # prefix_to_uncovered[tuple] = set of indices
            
            # Wait, the seed didn't do this.
            # Let's try to implement dynamic prefix map.
            
            pass # Placeholder for logic
        
        # This dynamic map approach is better.
        
        # Let's restart with the dynamic map idea.
        
    return ''.join(map(str, word))
