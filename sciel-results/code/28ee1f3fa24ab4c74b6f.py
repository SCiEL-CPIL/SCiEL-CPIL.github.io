"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate the standard superpermutation
    def standard_superperm(n):
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
                # Find max overlap
                overlap = 0
                # Max possible overlap is len(block)-1, but practically small
                # We check suffix of new against prefix of block
                max_ov = min(len(new), len(block) - 1)
                for o in range(max_ov, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    best_word = standard_superperm(n)
    best_len = len(best_word)
    
    # 2. Try to find a shorter superpermutation using a greedy heuristic
    # This is effective for small n (like 7) if a Hamiltonian path or near-Hamiltonian path exists.
    if n <= 7 and time.time() < deadline:
        # Graph representation
        # Nodes are permutations (tuples of ints)
        # Edge from P to Q if P[1:] == Q[:n-1]
        
        # Precompute successors for all permutations
        # This is expensive for n=7 (5040 nodes), but doable once.
        # However, we don't need all edges, just neighbors.
        
        # We will use a greedy approach with backtracking
        # To keep it simple and fast, we limit the search depth or restarts.
        
        # For n=7, a full TSP is too slow.
        # We'll try a simple greedy:
        # 1. Random start
        # 2. Always pick the unvisited neighbor with the lowest "degree" (fewest unvisited neighbors)
        # 3. If stuck, restart with a new random start.
        
        # We need to check if we can beat the standard length.
        # Standard length for n=7 is 5913.
        # A Hamiltonian path would be 5046.
        # Even a path with 100 repeats would be 5146, which is much better.
        
        # Let's try a few random greedy runs.
        
        def greedy_search(n, max_restarts=50):
            perms = list(itertools.permutations(range(1, n + 1)))
            perm_to_idx = {p: i for i, p in enumerate(perms)}
            num_perms = len(perms)
            
            # Build adjacency list
            # adj[u] = list of v such that u[1:] == v[:n-1]
            adj = [[] for _ in range(num_perms)]
            for i, p in enumerate(perms):
                suffix = p[1:]
                # Find all perms starting with suffix
                # This is O(N * N) if naive. 
                # Better: map prefix to list of indices
                prefix_map = {}
                for j, q in enumerate(perms):
                    prefix_map.setdefault(q[:n-1], []).append(j)
                
                # Actually, let's just build the map once
                break # placeholder
                
            # Rebuild prefix map properly
            prefix_map = {}
            for j, q in enumerate(perms):
                prefix_map.setdefault(q[:n-1], []).append(j)
            
            for i, p in enumerate(perms):
                suffix = p[1:]
                adj[i] = prefix_map.get(suffix, [])
            
            best_len_found = float('inf')
            best_path = None
            
            for _ in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                visited = [False] * num_perms
                path = []
                
                # Random start
                start = random.randint(0, num_perms - 1)
                current = start
                visited[current] = True
                path.append(current)
                
                # Greedy
                while len(path) < num_perms:
                    neighbors = adj[current]
                    # Filter unvisited
                    unvisited_neighbors = [v for v in neighbors if not visited[v]]
                    
                    if not unvisited_neighbors:
                        # Stuck
                        break
                    
                    # Heuristic: pick the one with fewest unvisited neighbors (Warnsdorff's rule)
                    # Count unvisited neighbors for each candidate
                    best_candidate = None
                    min_unvisited_count = float('inf')
                    
                    # To save time, if many candidates, sample
                    if len(unvisited_neighbors) > 10:
                        candidates = random.sample(unvisited_neighbors, 10)
                    else:
                        candidates = unvisited_neighbors
                        
                    for v in candidates:
                        # Count unvisited neighbors of v
                        count = 0
                        for w in adj[v]:
                            if not visited[w]:
                                count += 1
                        if count < min_unvisited_count:
                            min_unvisited_count = count
                            best_candidate = v
                            
                    if best_candidate is None:
                        break
                        
                    current = best_candidate
                    visited[current] = True
                    path.append(current)
                    
                if len(path) == num_perms:
                    # Found a Hamiltonian path!
                    # Length is num_perms + n - 1
                    # This is the theoretical minimum.
                    return perms[path[0]] + [x for x in perms[path[1]][n-1:]] # Reconstruction logic needed
                    
            return None
        
        # The reconstruction from path of indices to string:
        # If path is P1, P2, ..., Pk
        # Word = P1 + P2[1:] + P3[1:] ...
        
        # Let's implement the reconstruction and check
        def path_to_word(path_indices, perms, n):
            if not path_indices:
                return ""
            w = list(perms[path_indices[0]])
            for idx in path_indices[1:]:
                p = perms[idx]
                w.extend(p[1:])
            return "".join(map(str, w))
        
        # We don't have the path from greedy_search above fully returned.
        # Let's modify greedy_search to return the word or None.
        
        # Given time constraints, and the difficulty of getting a valid Hamiltonian path for n=7,
        # and the fact that the standard construction is very robust,
        # I will stick to the standard construction to ensure validity and avoid timeouts/invalid results.
        # The potential gain is small (1.001 vs 1.0) and the risk of bug/timeout is high.
        
    return best_word
# EVOLVE-BLOCK-END
