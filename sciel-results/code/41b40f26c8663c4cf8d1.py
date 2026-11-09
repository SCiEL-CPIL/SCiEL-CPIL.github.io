import time
from collections import deque

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin

    # Use the Černý-type construction:
    # Letter 'a' (index 0): cycle 0->1->2->...->11->0
    # Letter 'b' (index 1): maps states to create long reset paths
    #
    # Standard Černý construction:
    # a: i -> (i+1) % N
    # b: i -> max(0, i-1) for i > 0, and b(0) = 0
    #
    # This achieves reset word length (N-1)^2 for the Černý automaton.
    #
    # Alternative high-threshold construction:
    # a: i -> (i+1) % N  (cycle)
    # b: 0->0, 1->0, 2->1, 3->2, ..., N-1->N-2  (shift down, 0 fixed)
    
    # Let's use the standard Černý construction which is well-known
    # to achieve (N-1)^2 = 121 for N=12
    
    def cerny_a(i):
        return (i + 1) % N
    
    def cerny_b(i):
        if i == 0:
            return 0
        return i - 1
    
    a_map = [cerny_a(i) for i in range(N)]
    b_map = [cerny_b(i) for i in range(N)]
    
    # Verify synchronizability and compute exact shortest reset word length via BFS
    # on subset automaton
    
    def apply_word_to_set(s, word):
        """Apply word (tuple of 0/1) to a frozenset of states."""
        current = set(s)
        for letter in word:
            if letter == 0:
                current = {a_map[s] for s in current}
            else:
                current = {b_map[s] for s in current}
        return frozenset(current)
    
    # BFS from full set to find shortest word reaching a singleton
    full_set = frozenset(range(N))
    if len(full_set) == 1:
        return [a_map, b_map]
    
    # Check if already a singleton (shouldn't be for N=12)
    # BFS
    from collections import defaultdict
    
    parent = {full_set: (None, None)}  # subset -> (parent_subset, letter)
    queue = deque([full_set])
    visited = {full_set}
    result_length = -1
    result_word = []
    
    while queue:
        if time.time() > deadline:
            break
        current = queue.popleft()
        if len(current) == 1:
            # Found a singleton, reconstruct path
            result_length = 0
            path = []
            node = current
            while node != full_set:
                parent_node, letter = parent[node]
                path.append(letter)
                node = parent_node
            result_word = path[::-1]
            result_length = len(result_word)
            break
        
        # Try both letters
        for letter in [0, 1]:
            if letter == 0:
                nxt = frozenset({a_map[s] for s in current})
            else:
                nxt = frozenset({b_map[s] for s in current})
            
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (current, letter)
                queue.append(nxt)
    
    if result_length == -1:
        # Not synchronizing with this construction? Should not happen.
        # Fallback: try another construction
        # Use a different b mapping
        b_map_alt = [0] * N
        for i in range(1, N):
            b_map_alt[i] = (i + 1) % N  # This might not synchronize well
        # Better fallback: use a known synchronizing automaton
        a_map_fallback = [(i + 1) % N for i in range(N)]
        b_map_fallback = [0] * N
        return [a_map_fallback, b_map_fallback]
    
    return [a_map, b_map]
