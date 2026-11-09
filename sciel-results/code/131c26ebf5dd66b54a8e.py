"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import itertools

def construct(n):
    if n <= 6:
        # Use standard recursive construction for small n
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
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        overlap = 0
                        break
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n=7, attempt to optimize the order of permutations of 1..6
    # to minimize the length of the superpermutation.
    
    # 1. Generate permutations of 1..6 in the order they appear in the standard S_6
    #    This is equivalent to generating all permutations of 1..6.
    #    The specific order in S_6 is not critical for the optimization, as long as
    #    we consider all 720 permutations.
    perms_6 = list(itertools.permutations(range(1, 7)))
    N = len(perms_6) # 720
    
    # 2. Precompute overlaps between all pairs of permutations.
    #    overlap[i][j] = max k such that suffix of perms_6[i] of length k
    #    equals prefix of perms_6[j] of length k.
    #    Max k is 5 (since length is 6, and we need distinct elements, so max match is 5?
    #    Actually, if perms are distinct, max match is 5. If match is 6, they are same perm,
    #    but we visit each once. So max k=5).
    
    #    However, we can just compute it on the fly or precompute a matrix.
    #    720x720 matrix is 518k entries. Fast enough.
    
    overlaps = [[0]*N for _ in range(N)]
    for i in range(N):
        pi = perms_6[i]
        for j in range(N):
            pj = perms_6[j]
            max_k = 5
            k = 0
            # Check suffix of pi vs prefix of pj
            # pi is tuple of 6 ints.
            # We want largest k such that pi[6-k:] == pj[:k]
            for k in range(5, 0, -1):
                if pi[6-k:] == pj[:k]:
                    overlaps[i][j] = k
                    break
    
    # 3. Find a path visiting all nodes with minimum cost.
    #    Cost of edge i->j is (13 - overlaps[i][j]).
    #    We want to minimize sum of costs.
    #    This is TSP path.
    #    Use 2-Opt heuristic.
    
    #    Initialize order with a greedy path (Nearest Neighbor)
    #    Start with node 0 (arbitrary).
    order = [0]
    visited = [False]*N
    visited[0] = True
    
    for _ in range(N-1):
        last = order[-1]
        best_next = -1
        best_cost = float('inf')
        for j in range(N):
            if not visited[j]:
                cost = 13 - overlaps[last][j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
        if best_next == -1:
            break
        order.append(best_next)
        visited[best_next] = True
        
    #    Apply 2-Opt improvements
    #    Limit iterations to fit in time budget.
    deadline = time.time() + 140.0
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(1, N-1):
            # Try reversing segment order[i..j]
            # Standard 2-Opt for path: reverse segment i..j
            # Check if reversing improves cost.
            # Cost before: cost(order[i-1], order[i]) + cost(order[j], order[j+1])
            # Cost after: cost(order[i-1], order[j]) + cost(order[i], order[j+1])
            # Note: indices in order list.
            # Let a = order[i-1], b = order[i], c = order[j], d = order[j+1]
            # We reverse b..c. New sequence: a, c, ..., b, d.
            # Edges changed: (a,b) and (c,d) become (a,c) and (b,d).
            # Wait, if we reverse order[i..j], the element at i becomes order[j] and at j becomes order[i].
            # So edge (i-1, i) changes from (a, b) to (a, order[j]).
            # Edge (j, j+1) changes from (c, d) to (order[i], d).
            
            a = order[i-1]
            b = order[i]
            c = order[j]
            d = order[j+1] if j+1 < N else -1
            
            if d == -1:
                # j is last element.
                # Reverse i..N-1.
                # Edge (i-1, i) changes from (a, b) to (a, order[N-1]).
                # No edge after j.
                old_cost = 13 - overlaps[a][b]
                new_cost = 13 - overlaps[a][order[N-1]]
                if new_cost < old_cost:
                    order[i:] = order[i:][::-1]
                    improved = True
                    continue
                
            old_cost = (13 - overlaps[a][b]) + (13 - overlaps[c][d])
            new_cost = (13 - overlaps[a][c]) + (13 - overlaps[b][d])
            
            if new_cost < old_cost:
                order[i:j+1] = order[i:j+1][::-1]
                improved = True
                
    # 4. Construct the superpermutation using the optimized order.
    word = []
    for idx, p_idx in enumerate(order):
        p = perms_6[p_idx]
        block = list(p) + [7] + list(p)
        
        if idx == 0:
            word.extend(block)
        else:
            # Find overlap with current word
            # We know the overlap is overlaps[order[idx-1]][p_idx]
            # But we should verify or just use it.
            # The overlap k means the last k chars of word match the first k chars of block.
            k = overlaps[order[idx-1]][p_idx]
            word.extend(block[k:])
            
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
