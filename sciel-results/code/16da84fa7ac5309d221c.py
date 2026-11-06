"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    deadline = time.time() + 85  # safety margin within 90s limit
    
    # Precompute modular inverses of 1..10 mod 101
    inv = [pow(x, -1, 101) for x in range(1, 11)]
    
    # Precompute all non-empty subsets: (size, residue, mask)
    all_subsets = []
    for mask in range(1, 1 << 10):
        size = bin(mask).count('1')
        residue = 0
        m = mask
        for x in range(1, 11):
            if m & 1:
                residue += inv[x - 1]
            m >>= 1
        residue %= 101
        all_subsets.append((size, residue, mask))
    
    # Group subsets by residue
    subsets_by_residue = [[] for _ in range(101)]
    for size, residue, mask in all_subsets:
        subsets_by_residue[residue].append((size, mask))
    
    # For each residue, sort subsets by size
    for r in range(101):
        subsets_by_residue[r].sort(key=lambda x: x[0])
    
    # Find the minimum size needed for each residue
    min_sizes = [0] * 101
    for r in range(101):
        if subsets_by_residue[r]:
            min_sizes[r] = subsets_by_residue[r][0][0]
    
    # The maximum of minimum sizes gives a lower bound on the answer
    lower_bound = max(min_sizes)
    
    # Binary search on K (maximum allowed subset size)
    # For a given K, check if there's a perfect matching where each residue
    # gets a subset of size <= K
    
    def can_match(K):
        # Build bipartite graph: left = residues 0..100, right = subsets
        # Edge exists if subset size <= K and subset residue matches
        # We need to assign each residue a distinct subset
        # Use Hopcroft-Karp for maximum bipartite matching
        
        # Build adjacency list for left nodes (residues)
        # Each residue r can match to any subset with residue r and size <= K
        adj = [[] for _ in range(101)]
        for r in range(101):
            for size, mask in subsets_by_residue[r]:
                if size <= K:
                    adj[r].append(mask)
                else:
                    break  # sorted by size, so all subsequent are larger
        
        # If any residue has no available subsets, matching is impossible
        for r in range(101):
            if not adj[r]:
                return False
        
        # Hopcroft-Karp algorithm
        # matchL[r] = mask assigned to residue r, or -1
        # matchR[mask] = residue assigned to mask, or -1
        matchL = [-1] * 101
        matchR = {}  # mask -> residue
        dist = [0] * 101
        
        def bfs():
            queue = deque()
            found = False
            for r in range(101):
                if matchL[r] == -1:
                    dist[r] = 0
                    queue.append(r)
                else:
                    dist[r] = -1
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    matched_r = matchR.get(mask, -1)
                    if matched_r == -1:
                        found = True
                    elif dist[matched_r] == -1:
                        dist[matched_r] = dist[r] + 1
                        queue.append(matched_r)
            return found
        
        def dfs(r):
            for mask in adj[r]:
                matched_r = matchR.get(mask, -1)
                if matched_r == -1 or (dist[matched_r] == dist[r] + 1 and dfs(matched_r)):
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
            dist[r] = -1
            return False
        
        matching = 0
        while bfs():
            for r in range(101):
                if matchL[r] == -1:
                    if dfs(r):
                        matching += 1
                    if time.time() > deadline:
                        return False
        
        return matching == 101
    
    # Binary search for the minimum K
    lo = lower_bound
    hi = 10  # max possible size
    
    best_K = hi
    best_assignment = None
    
    # Try K values from lower bound up
    for K in range(lo, hi + 1):
        if time.time() > deadline:
            break
        if can_match(K):
            best_K = K
            break
    
    # Now find the actual assignment for best_K
    # Rebuild and do the matching to get the assignment
    def find_assignment(K):
        adj = [[] for _ in range(101)]
        for r in range(101):
            for size, mask in subsets_by_residue[r]:
                if size <= K:
                    adj[r].append(mask)
                else:
                    break
        
        matchL = [-1] * 101
        matchR = {}
        dist = [0] * 101
        
        def bfs():
            queue = deque()
            found = False
            for r in range(101):
                if matchL[r] == -1:
                    dist[r] = 0
                    queue.append(r)
                else:
                    dist[r] = -1
            while queue:
                r = queue.popleft()
                for mask in adj[r]:
                    matched_r = matchR.get(mask, -1)
                    if matched_r == -1:
                        found = True
                    elif dist[matched_r] == -1:
                        dist[matched_r] = dist[r] + 1
                        queue.append(matched_r)
            return found
        
        def dfs(r):
            for mask in adj[r]:
                matched_r = matchR.get(mask, -1)
                if matched_r == -1 or (dist[matched_r] == dist[r] + 1 and dfs(matched_r)):
                    matchL[r] = mask
                    matchR[mask] = r
                    return True
            dist[r] = -1
            return False
        
        while bfs():
            for r in range(101):
                if matchL[r] == -1:
                    if dfs(r):
                        pass
        
        # Convert masks back to term lists
        result = []
        for r in range(101):
            mask = matchL[r]
            if mask == -1:
                # Fallback: use any subset for this residue
                if subsets_by_residue[r]:
                    mask = subsets_by_residue[r][0][1]
                else:
                    mask = 1
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            result.append(terms)
        return result
    
    if best_K <= hi:
        result = find_assignment(best_K)
        return result
    else:
        # Fallback: use minimum-size subsets
        result = []
        for r in range(101):
            if subsets_by_residue[r]:
                mask = subsets_by_residue[r][0][1]
            else:
                mask = 1
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            result.append(terms)
        return result

# EVOLVE-BLOCK-END
