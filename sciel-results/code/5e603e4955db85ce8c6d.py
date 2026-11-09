import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s

    N = 12
    P = 27720  # LCM(1..12)
    
    best_L = 0
    best_m = 0

    # Function to check if a perfect matching exists for a given residue r and length L
    # Left nodes: 1..N
    # Right nodes: 1..L (representing offsets from m, so value is m+j)
    def has_perfect_matching(r, L):
        # adjacency list for left nodes
        # adj[i] contains list of j in 1..L such that (r+j) % i == 0
        adj = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            # Find j in 1..L such that (r+j) % i == 0
            # j % i == (-r) % i
            target = (-r) % i
            # j can be target, target+i, target+2i, ... <= L
            # Note: if target == 0, j can be i, 2i, ...
            # Actually, j is in 1..L.
            # So j = k*i - r mod i? No.
            # (r + j) % i == 0 => j % i == (-r) % i.
            # Let rem = (-r) % i.
            # Possible j: rem, rem+i, rem+2i, ...
            # But j must be >= 1.
            # If rem == 0, smallest positive j is i.
            # If rem > 0, smallest positive j is rem.
            
            rem = (-r) % i
            if rem == 0:
                j_start = i
            else:
                j_start = rem
                
            j = j_start
            while j <= L:
                adj[i].append(j)
                j += i

        # Bipartite matching using DFS (Kuhn's algorithm)
        # match[j] = i means right node j is matched to left node i
        # match[i] = j means left node i is matched to right node j
        match_r = [0] * (L + 1)  # 1-indexed for right nodes
        match_l = [0] * (N + 1)  # 1-indexed for left nodes

        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_r[v] == 0 or dfs(match_r[v], visited):
                    match_r[v] = u
                    match_l[u] = v
                    return True
            return False

        count = 0
        for u in range(1, N + 1):
            visited = [False] * (L + 1)
            if dfs(u, visited):
                count += 1
            if count < N:
                # Early exit: if we can't match all, and we've processed some, 
                # we might still match all if we reorder? 
                # Kuhn's algorithm works regardless of order if we retry, 
                # but standard implementation iterates u=1..N once.
                # Actually, standard Kuhn's is: for each u, try to find augmenting path.
                # If it fails for u, it might succeed later if graph structure allows?
                # No, the standard algorithm guarantees max matching if run for all u.
                # So we must run for all u.
                pass

        return count == N

    # Iterate over all residues
    for r in range(P):
        # Check time
        if time.time() > deadline:
            break
            
        # Find smallest L such that has_perfect_matching(r, L) is True
        # L must be at least 1.
        # Upper bound? The problem states max L is 144. Let's use 200 to be safe.
        found = False
        for L in range(1, 200):
            if has_perfect_matching(r, L):
                if L > best_L:
                    best_L = L
                    best_m = r
                found = True
                break

    # Return the best m found
    # Ensure it's an int
    return int(best_m)
