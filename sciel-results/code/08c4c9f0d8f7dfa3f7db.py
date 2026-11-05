import itertools
import random
import time

def construct(n):
    if n < 7:
        return standard_construct(n)
    
    # For n=7, use TSP on permutations of 1..7
    symbols = list(range(1, 8))
    perms = list(itertools.permutations(symbols))
    num_perms = len(perms)
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # We initialize with 0. We iterate k from 7 down to 1.
    # If suffix_k(i) == prefix_k(j), and overlap[i][j] is still 0, set to k.
    # Since we go from high k to low, the first match is the maximum.
    
    overlap = [[0] * num_perms for _ in range(num_perms)]
    
    # To speed up, we can use a list of lists for prefixes?
    # Or just use the dictionary method for each k.
    
    for k in range(7, 0, -1):
        # Map prefix of length k to list of indices
        prefix_map = {}
        for j, p in enumerate(perms):
            pref = p[:k]
            if pref not in prefix_map:
                prefix_map[pref] = []
            prefix_map[pref].append(j)
        
        for i, p in enumerate(perms):
            suff = p[-k:]
            if suff in prefix_map:
                for j in prefix_map[suff]:
                    if overlap[i][j] == 0:
                        overlap[i][j] = k
    
    # Now we have overlap matrix.
    # Cost of edge i->j is 7 - overlap[i][j].
    # We want to minimize total cost.
    
    # Greedy Initial Solution
    def greedy_path():
        visited = [False] * num_perms
        path = []
        start = random.randint(0, num_perms - 1)
        visited[start] = True
        path.append(start)
        current = start
        for _ in range(num_perms - 1):
            best_next = -1
            best_cost = 8 # max cost is 7
            for j in range(num_perms):
                if not visited[j]:
                    cost = 7 - overlap[current][j]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = j
            visited[best_next] = True
            path.append(best_next)
            current = best_next
        return path

    # 2-opt Improvement
    def path_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            cost += 7 - overlap[path[i]][path[i+1]]
        return cost

    def two_opt(path):
        current_cost = path_cost(path)
        improved = True
        # Time limit check inside loop? 
        # Let's run a fixed number of passes or until no improvement.
        # 5040 nodes, 2-opt is O(N^2) per pass. 25M ops per pass.
        # Python 25M ops is ~1-2 seconds.
        # We can do a few passes.
        
        for _ in range(50): # Max 50 passes
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path) - 1):
                    # Reverse path[i:j+1]
                    # Check if it reduces cost
                    # Edges affected: (i-1, i), (i, i+1) ... (j-1, j), (j, j+1)
                    # New edges: (i-1, j), (i, j+1) and the internal reversed edges.
                    # Wait, 2-opt reverses the segment.
                    # The internal edges are reversed, so their cost remains the same?
                    # No, overlap is not symmetric. overlap[A][B] != overlap[B][A].
                    # So we must recalculate the cost of the reversed segment.
                    
                    # This makes 2-opt expensive: O(N) per pair (i,j).
                    # Total O(N^3). 5040^3 is 10^11. Too slow.
                    
                    # Alternative: Use "Or-opt" or just simple swaps?
                    # Or just stick to Greedy with many restarts but smarter?
                    
                    # Let's abandon full 2-opt due to asymmetry and time.
                    # Instead, use a "First Improvement" local search with swaps of adjacent nodes?
                    # Swap i and i+1.
                    # Cost change:
                    # Old: cost(i-1, i) + cost(i, i+1) + cost(i+1, i+2)
                    # New: cost(i-1, i+1) + cost(i+1, i) + cost(i, i+2)
                    # This is O(1) per swap.
                    # Pass through all i, swap if better. Repeat.
                    
                    pass # This logic is getting complex to implement correctly in the search block.
            
            # Let's try a simpler local search: Random restarts + Greedy + 1-swap local search.
            
    # Given the complexity and time limits, let's stick to a robust Greedy with many restarts 
    # but optimized overlap lookup, which is effectively what the current code does 
    # but the current code's overlap precomputation is O(N^2 * 7) which is slow in Python.
    # The dictionary method above is O(N * 7 * log N) roughly, much faster.
    
    best_path = None
    best_cost = float('inf')
    
    start_time = time.time()
    num_restarts = 200 # Adjust based on time
    
    for _ in range(num_restarts):
        if time.time() - start_time > 100:
            break
            
        path = greedy_path()
        cost = path_cost(path)
        
        # Simple local search: try swapping adjacent elements
        improved = True
        while improved:
            improved = False
            for i in range(1, len(path) - 1):
                # Swap path[i] and path[i+1]
                a = path[i-1]
                b = path[i]
                c = path[i+1]
                d = path[i+2] if i+2 < len(path) else None
                
                old_cost = (7 - overlap[a][b]) + (7 - overlap[b][c])
                if d is not None:
                    old_cost += (7 - overlap[c][d])
                    new_cost = (7 - overlap[a][c]) + (7 - overlap[c][b]) + (7 - overlap[b][d])
                else:
                    new_cost = (7 - overlap[a][c]) + (7 - overlap[c][b])
                
                if new_cost < old_cost:
                    path[i], path[i+1] = path[i+1], path[i]
                    cost += new_cost - old_cost
                    improved = True
                    # Restart scan from beginning? Or continue?
                    # Continue is faster, but might miss improvements.
                    # Let's break and restart the while loop for simplicity/robustness
                    break
        
        if cost < best_cost:
            best_cost = cost
            best_path = path[:]
            
    if best_path is None:
        return standard_construct(n)
        
    # Construct string
    word = list(perms[best_path[0]])
    for idx in range(1, len(best_path)):
        ov = overlap[best_path[idx-1]][best_path[idx]]
        word.extend(perms[best_path[idx]][ov:])
        
    return "".join(map(str, word))

def standard_construct(n):
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
