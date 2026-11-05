import time
import itertools

def _greedy_set_cover(n):
    """
    Greedy set cover on de Bruijn graph of order n-1.
    Nodes: permutations of n-1 distinct symbols from 1..n
    Edges: transitions that create valid n-permutations
    Goal: cover all n! permutations of 1..n
    """
    if n <= 1:
        return list(range(1, n + 1))
    
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    # All permutations of 1..n
    all_perms = list(itertools.permutations(range(1, n + 1)))
    total_perms = len(all_perms)
    
    # We need to find a path in the de Bruijn graph that covers all permutations
    # Nodes are (n-1)-tuples, edges represent adding a symbol
    
    # For n=7, we have 7! = 5040 permutations to cover
    # The de Bruijn graph has nodes that are (n-1)-permutations of 1..n
    
    # Strategy: Build a word by greedily selecting next symbols
    # that maximize the number of new permutations covered
    
    # Initialize with the first permutation
    current = list(all_perms[0])
    covered = set()
    covered.add(tuple(all_perms[0]))
    
    # Track which permutations we can still cover
    remaining = set(range(total_perms))
    remaining.discard(0)
    
    # For each position, try each possible next symbol and pick the one
    # that covers the most new permutations
    
    word = current[:]
    
    # We need to extend the word until all permutations are covered
    # Each new symbol creates one new n-window
    
    while len(covered) < total_perms:
        if time.time() > deadline:
            break
        
        best_symbol = None
        best_new_covered = 0
        best_cost = float('inf')
        
        # Try each possible next symbol
        for sym in range(1, n + 1):
            # Check if this symbol can be appended (must not create duplicate in window)
            if sym in word[-(n-1):]:
                continue
            
            # Simulate appending this symbol
            new_window = tuple(word[-(n-1):] + [sym])
            
            # Check if this window is a valid permutation
            if len(set(new_window)) != n:
                continue
            
            # Find which permutation this is
            # We need to map this tuple to an index in all_perms
            # For efficiency, we could precompute a dictionary
            
            # Count how many new permutations this would cover
            # Actually, each new symbol creates exactly one new n-window
            # So we just need to check if this specific permutation is new
            
            # Use a set for O(1) lookup
            if new_window not in covered:
                new_covered_count = 1
            else:
                new_covered_count = 0
            
            # We want to maximize new coverage, minimize cost
            # Cost here is just 1 (one symbol added)
            cost = 1
            
            # Prefer symbols that cover new permutations
            if new_covered_count > best_new_covered or (new_covered_count == best_new_covered and cost < best_cost):
                best_new_covered = new_covered_count
                best_cost = cost
                best_symbol = sym
        
        if best_symbol is None:
            # No valid symbol found - this shouldn't happen for valid superpermutations
            # But as a fallback, just append 1
            best_symbol = 1
        
        word.append(best_symbol)
        new_window = tuple(word[-n:])
        covered.add(new_window)
        
        # Remove from remaining
        try:
            remaining.discard(all_perms.index(new_window))
        except ValueError:
            pass
    
    return word

def _classic_construct(n):
    """Classic recursive construction for superpermutations."""
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    word = [1]
    
    for m in range(2, n + 1):
        # Extract all unique (m-1)-permutations from the current word in order of appearance
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        # Construct the new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            
            new_word.extend(block[overlap:])
        
        word = new_word
    
    return "".join(map(str, word))

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses greedy set cover heuristic for n >= 6.
    """
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin before 120s limit
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # For n <= 5, use classic construction (optimal)
    if n <= 5:
        return _classic_construct(n)
    
    # For n >= 6, try greedy set cover
    try:
        word = _greedy_set_cover(n)
        result = "".join(map(str, word))
        
        # Verify the result is valid
        if time.time() < deadline:
            # Check if all permutations are covered
            all_perms = set(itertools.permutations(range(1, n + 1)))
            covered = set()
            for i in range(len(word) - n + 1):
                window = tuple(word[i:i + n])
                if len(set(window)) == n:
                    covered.add(window)
            
            if covered == all_perms:
                return result
    except Exception:
        pass
    
    # Fallback to classic construction
    return _classic_construct(n)
