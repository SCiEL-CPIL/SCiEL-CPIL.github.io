"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 80
    P = 101
    N = 10
    # Precompute inverses
    inv = [pow(x, -1, P) for x in range(1, N + 1)]
    
    # Precompute all non-empty subsets
    # subset_info[i] = (residue, mask, size)
    subset_info = []
    for mask in range(1, 1 << N):
        size = 0
        res = 0
        for i in range(N):
            if mask & (1 << i):
                size += 1
                res = (res + inv[i]) % P
        subset_info.append((res, mask, size))
        
    num_subsets = len(subset_info)
    
    # We want to find min L such that there is a matching from residues 0..P-1 to subsets
    # where subset size <= L and subset residue == row residue.
    
    def solve_matching(L):
        # Adjacency list for residues
        # adj[r] = list of subset indices j where subset_info[j][0] == r and subset_info[j][2] <= L
        adj = [[] for _ in range(P)]
        for j in range(num_subsets):
            res, mask, size = subset_info[j]
            if size <= L:
                adj[res].append(j)
        
        # Bipartite matching: Left = residues (0..P-1), Right = subsets (0..num_subsets-1)
        # matchR[j] = i means subset j is matched to residue i
        matchR = [-1] * num_subsets
        
        def dfs(u, seen):
            for v in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                if matchR[v] == -1 or dfs(matchR[v], seen):
                    matchR[v] = u
                    return True
            return False
        
        match_count = 0
        for u in range(P):
            seen = [False] * num_subsets
            if dfs(u, seen):
                match_count += 1
            if match_count == P:
                break
                
        if match_count < P:
            return None
        
        # Reconstruct assignment
        # assign[r] = subset index j
        assign = [None] * P
        for j in range(num_subsets):
            if matchR[j] != -1:
                assign[matchR[j]] = j
        return assign

    # Try L from 1 to N
    for L in range(1, N + 1):
        if time.time() > deadline:
            # Fallback: just use the first valid subset for each residue (might have large max length)
            # This should not happen with L=10
            break
        result = solve_matching(L)
        if result is not None:
            # Construct the final lists
            final_lists = []
            for r in range(P):
                j = result[r]
                mask = subset_info[j][1]
                terms = [x + 1 for x in range(N) if mask & (1 << x)]
                final_lists.append(terms)
            return final_lists
    
    # If L=10 failed (shouldn't happen), return something valid
    # Actually, L=10 always works because there are 1023 subsets and 101 residues.
    # Pigeonhole principle ensures some residue has at least one subset, but we need distinct subsets.
    # Wait, is it guaranteed that a perfect matching exists for L=10?
    # Yes, because the bipartite graph is very dense.
    # However, if it fails, we can just greedily pick.
    
    # Greedy fallback
    used = set()
    final_lists = []
    for r in range(P):
        found = False
        for j in range(num_subsets):
            res, mask, size = subset_info[j]
            if res == r and j not in used:
                used.add(j)
                terms = [x + 1 for x in range(N) if mask & (1 << x)]
                final_lists.append(terms)
                found = True
                break
        if not found:
            # This should not happen
            final_lists.append([1])
    return final_lists

# EVOLVE-BLOCK-END
