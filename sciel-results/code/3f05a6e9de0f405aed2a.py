def construct():
    n = 256
    numbers = list(range(n, 0, -1))
    
    def is_compatible(x, S):
        """
        Check if x can be added to S (where all elements in S are > x).
        x becomes the new minimum.
        Violation if exists b, c in S such that x divides (b+c).
        """
        if not S:
            return True
        residues = set()
        for c in S:
            residues.add(c % x)
        for b in S:
            if (-b) % x in residues:
                return False
        return True

    # Greedy initialization to set a good lower bound
    greedy_set = []
    for x in numbers:
        if is_compatible(x, greedy_set):
            greedy_set.append(x)
    
    best_set = greedy_set.copy()
    
    def search(S, idx):
        nonlocal best_set
        # Pruning
        remaining = n - idx
        if len(S) + remaining <= len(best_set):
            return
            
        if idx == n + 1:
            if len(S) > len(best_set):
                best_set = S.copy()
            return
        
        x = numbers[idx - 1]
        
        # Option 1: Include x
        if is_compatible(x, S):
            S.append(x)
            search(S, idx + 1)
            S.pop()
            
        # Option 2: Exclude x
        search(S, idx + 1)
        
    search([], 1)
    return list(best_set)
