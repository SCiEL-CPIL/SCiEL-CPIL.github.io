import itertools
from collections import deque

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
        # Fallback to standard construction for larger n
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
    
    # For n=4,5,6,7: Use constraint-based backtracking
    # We build the word one symbol at a time.
    # State: current suffix (last n-1 symbols), set of covered permutations
    # Pruning: if a suffix of length k cannot be extended to cover any new permutation, prune.
    
    # Precompute all permutations
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_set = set(all_perms)
    total_perms = len(all_perms)
    
    # For n=7, total_perms = 5040
    # We need to find a word of length ~5905 that contains all 5040 permutations.
    
    # The key insight: when we append a symbol x to the current word,
    # the new window (last n symbols) is a permutation if it has all distinct symbols.
    # We track which permutations have been seen.
    
    # For efficient pruning, we need to know: given a suffix of length k (k < n),
    # can it be extended to form at least one uncovered permutation?
    
    # Strategy: Use a greedy approach with lookahead.
    # At each step, try each possible next symbol (1..n) that doesn't create a duplicate
    # in the last n-1 positions (to maintain the property that windows are permutations).
    # Actually, we don't need to maintain that property strictly - we just need to ensure
    # that every window of length n that IS a permutation gets counted.
    
    # Better approach: We're building a de Bruijn-like sequence for permutations.
    # The state is the last n-1 symbols. From state s = (s_1, ..., s_{n-1}),
    # we can append x to get state (s_2, ..., s_{n-1}, x).
    # The permutation covered is (s_1, ..., s_{n-1}, x) if it's a valid permutation.
    
    # We want to visit all 5040 permutation states (as windows) in a path.
    # This is equivalent to finding a path in the directed graph where:
    # - Nodes are permutations of 1..n (5040 nodes for n=7)
    # - Edge from perm P to perm Q exists if Q[i] = P[i+1] for i=1..n-1
    #   (i.e., Q is P shifted left by 1, with a new last symbol)
    # - We want a trail that visits every node (permutation) at least once.
    
    # This is the "permutation graph" or "superpermutation graph".
    # Finding a Hamiltonian path in this graph gives the shortest superpermutation.
    
    # For n=7, the graph has 5040 nodes. We can use a greedy heuristic with backtracking.
    
    # Represent permutations as tuples for hashing.
    # State: current permutation (last n symbols), set of visited permutations.
    
    # Start with permutation (1, 2, ..., n)
    start_perm = tuple(range(1, n + 1))
    
    # Build adjacency: from perm P, we can go to perm (P[1], P[2], ..., P[n-1], x)
    # where x is any symbol not in P[1:n-1] (to keep it a permutation).
    # Actually, (P[1], ..., P[n-1], x) is a permutation iff x is not in P[1:n-1].
    # There are exactly n-1 possible next symbols (all except P[0]).
    
    # Precompute neighbors for each permutation
    neighbors = {}
    for perm in all_perms:
        prefix = perm[1:]  # First n-1 elements of the "next" state
        # Next state is (prefix[0], ..., prefix[n-2], x) for x not in prefix
        used = set(prefix)
        available = [x for x in range(1, n + 1) if x not in used]
        nbrs = []
        for x in available:
            next_perm = prefix + (x,)
            nbrs.append(next_perm)
        neighbors[perm] = nbrs
    
    # Now we need to find a path that visits all 5040 nodes.
    # Use a greedy heuristic: at each step, go to the unvisited neighbor with the
    # fewest unvisited neighbors (Warnsdorff's rule).
    
    def solve_greedy():
        visited = set()
        path = [start_perm]
        visited.add(start_perm)
        current = start_perm
        
        for step in range(total_perms - 1):
            nbrs = neighbors[current]
            # Find unvisited neighbors
            unvisited_nbrs = [nb for nb in nbrs if nb not in visited]
            
            if not unvisited_nbrs:
                # Stuck - this shouldn't happen if we use good heuristic
                # But let's handle it: try to find any unvisited node reachable
                return None
            
            # Warnsdorff: pick the one with fewest unvisited neighbors
            def count_unvisited(nb):
                return sum(1 for x in neighbors[nb] if x not in visited)
            
            unvisited_nbrs.sort(key=count_unvisited)
            current = unvisited_nbrs[0]
            visited.add(current)
            path.append(current)
        
        # Build the word from the path
        # The word is: path[0] + path[1][n-1:] + path[2][n-1:] + ...
        word = list(path[0])
        for i in range(1, len(path)):
            word.append(path[i][n - 1])
        return word
    
    word = solve_greedy()
    
    if word is None:
        # Fallback to block-based approach if greedy fails
        prev_perms = list(itertools.permutations(range(1, n)))
        blocks = []
        for p in prev_perms:
            b = list(p) + [n] + list(p)
            blocks.append(b)
        
        num_blocks = len(blocks)
        overlaps = [[0]*num_blocks for _ in range(num_blocks)]
        for i in range(num_blocks):
            bi = blocks[i]
            for j in range(num_blocks):
                bj = blocks[j]
                max_k = min(len(bi), len(bj))
                found = 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        found = 0
                        break
                    if bi[-k:] == bj[:k]:
                        found = k
                        break
                overlaps[i][j] = found
        
        current = 0
        unvisited = set(range(1, num_blocks))
        path = [0]
        while unvisited:
            best_next = -1
            best_overlap = -1
            for j in unvisited:
                if overlaps[current][j] > best_overlap:
                    best_overlap = overlaps[current][j]
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        final_word = list(blocks[path[0]])
        for k in range(1, len(path)):
            ov = overlaps[path[k-1]][path[k]]
            final_word += blocks[path[k]][ov:]
        return "".join(map(str, final_word))
    
    return "".join(map(str, word))
