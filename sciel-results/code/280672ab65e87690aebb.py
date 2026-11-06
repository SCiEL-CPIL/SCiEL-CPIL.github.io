import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    # 1. Precompute inverses and all subsets
    P = 101
    D = 10
    invs = [0] * (D + 1)
    for i in range(1, D + 1):
        invs[i] = pow(i, -1, P)

    # Generate all non-empty subsets
    # Each subset is represented by a bitmask and a tuple of elements
    # We store: (sum_mod, size, bitmask, elements_tuple)
    subsets = []
    for mask in range(1, 1 << D):
        s = 0
        size = 0
        elems = []
        for i in range(1, D + 1):
            if mask & (1 << (i - 1)):
                s = (s + invs[i]) % P
                size += 1
                elems.append(i)
        subsets.append((s, size, mask, tuple(elems)))

    # Group subsets by their sum
    # sum_to_subsets[r] is a list of indices into 'subsets' that sum to r
    sum_to_subsets = [[] for _ in range(P)]
    for idx, (s, size, mask, elems) in enumerate(subsets):
        sum_to_subsets[s].append(idx)

    # 2. Define Hopcroft-Karp algorithm
    # Left nodes: 0..100 (residues)
    # Right nodes: 0..len(subsets)-1 (subset indices)
    # Edge exists if subset j sums to residue i
    
    def hopcroft_karp(max_size_K):
        # Build adjacency list for left nodes
        # adj[i] = list of subset indices j such that subsets[j] has size <= K and sum == i
        adj = [[] for _ in range(P)]
        for i in range(P):
            for j in sum_to_subsets[i]:
                if subsets[j][1] <= max_size_K:
                    adj[i].append(j)
        
        n_left = P
        n_right = len(subsets)
        
        # pairU[i] = subset index matched to residue i, or -1
        # pairV[j] = residue index matched to subset j, or -1
        pairU = [-1] * n_left
        pairV = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if pairU[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    u_prime = pairV[v]
                    if u_prime == -1:
                        found = True
                    elif dist[u_prime] == float('inf'):
                        dist[u_prime] = dist[u] + 1
                        queue.append(u_prime)
            return found

        def dfs(u):
            for v in adj[u]:
                u_prime = pairV[v]
                if u_prime == -1 or (dist[u_prime] == dist[u] + 1 and dfs(u_prime)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = float('inf')
            return False

        matching = 0
        while bfs():
            for u in range(n_left):
                if pairU[u] == -1:
                    if dfs(u):
                        matching += 1
                        if matching == n_left:
                            return pairU
        
        return None

    # 3. Binary search for minimum K
    min_K = 1
    max_K = D
    best_K = None
    best_matching = None
    
    # We know K=10 works (all subsets available)
    # We want the smallest K
    
    # Optimization: Check if K=1 is possible? Unlikely.
    # Let's just binary search.
    
    lo = 1
    hi = D
    while lo <= hi:
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) // 2
        match = hopcroft_karp(mid)
        if match is not None:
            best_K = mid
            best_matching = match
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_matching is None:
        # Fallback: Should not happen, but if it does, return empty or raise?
        # The problem implies feasibility. If we run out of time, we might not have found it.
        # However, with D=10, the search space is small.
        # If we didn't find a match, we have to return something. 
        # Let's assume we found it. If not, we can try a greedy fallback or just return empty lists (invalid).
        # Given the "cell findings", it works.
        return [[] for _ in range(P)]

    # 4. Construct result
    result = [None] * P
    for r in range(P):
        subset_idx = best_matching[r]
        # subsets[subset_idx] is (s, size, mask, elems)
        result[r] = list(subsets[subset_idx][3])
        
    return result

# Helper to allow module execution if needed, though construct is the entry point
if __name__ == "__main__":
    res = construct()
    # Verify
    valid = True
    max_len = 0
    for r in range(101):
        s = 0
        for d in res[r]:
            s = (s + pow(d, -1, 101)) % 101
        if s != r:
            valid = False
            break
        if len(res[r]) > max_len:
            max_len = len(res[r])
    print(f"Valid: {valid}, Max Length: {max_len}")
