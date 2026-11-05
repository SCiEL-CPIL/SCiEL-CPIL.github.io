import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Get blocks for n-1
    # We need the list of all permutations of 1..n-1 in the order they appear in the superpermutation of n-1?
    # No, we just need the *set* of all permutations of 1..n-1.
    # The "standard" superpermutation for n-1 contains all of them.
    # We can just generate all permutations of 1..n-1.
    from itertools import permutations
    
    symbols = list(range(1, n))
    blocks = [p for p in permutations(symbols)]
    N = len(blocks)
    
    # Precompute overlap matrix
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Note: k can be 0 to n-1.
    M = [[0] * N for _ in range(N)]
    for i in range(N):
        bi = blocks[i]
        for j in range(N):
            bj = blocks[j]
            # Max possible overlap is n-1
            max_k = n - 1
            # Check k from max_k down to 0
            # Since n is small (<=7), this is fast
            for k in range(max_k, -1, -1):
                if k == 0:
                    M[i][j] = 0
                    break
                if bi[-k:] == bj[:k]:
                    M[i][j] = k
                    break

    # Initial solution: Greedy Nearest Neighbor
    # Start with random block
    visited = [False] * N
    order = [random.randint(0, N-1)]
    visited[order[0]] = True
    current = order[0]
    
    for _ in range(N - 1):
        best_next = -1
        best_overlap = -1
        for nxt in range(N):
            if not visited[nxt]:
                ov = M[current][nxt]
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = nxt
        if best_next == -1:
            # Should not happen
            break
        order.append(best_next)
        visited[best_next] = True
        current = best_next
        
    # Calculate initial score
    def calc_score(ord_list):
        score = 0
        for i in range(len(ord_list) - 1):
            score += M[ord_list[i]][ord_list[i+1]]
        return score

    best_score = calc_score(order)
    best_order = order[:]
    
    start_time = time.time()
    time_limit = 110
    
    # VNS
    # Neighborhoods:
    # 1. Swap(i, j)
    # 2. 2-opt(i, j) (reverse i..j)
    # 3. Or-opt(i, j) (move i to j)
    
    # To make it faster, we can limit the number of iterations or use a time check.
    # N=720. O(N^2) per pass is ~500k ops.
    # We can do multiple passes.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 1. 2-opt
        for i in range(N - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, N):
                # Reverse segment i..j
                # Affected edges:
                # Old: (i-1, i), (j, j+1) and internal (i, i+1)...(j-1, j)
                # New: (i-1, j), (i, j+1) and internal reversed
                
                # Calculate delta for external edges
                delta = 0
                if i > 0:
                    delta -= M[best_order[i-1]][best_order[i]]
                    delta += M[best_order[i-1]][best_order[j]]
                if j < N - 1:
                    delta -= M[best_order[j]][best_order[j+1]]
                    delta += M[best_order[i]][best_order[j+1]]
                
                # Calculate delta for internal edges
                # Old internal sum: sum(M[best_order[k]][best_order[k+1]] for k in i..j-1)
                # New internal sum: sum(M[best_order[k+1]][best_order[k]] for k in i..j-1)
                # We need to compute this sum.
                # To avoid O(N) per 2-opt, we can precompute prefix sums?
                # But the direction changes.
                # Given N=720, O(N) internal sum is 720 ops.
                # Total 2-opt is O(N^3) ~ 3.7 * 10^8. Too slow in Python.
                
                # Optimization: Only try small reversals? Or use a different move?
                # Let's try Swap first, it's O(1) delta.
                pass
        
        # Let's implement Swap (1-opt) which is O(1) delta.
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, N):
                # Swap i and j
                # Affected edges:
                # (i-1, i), (i, i+1), (j-1, j), (j, j+1)
                # Note: if j = i+1, edges overlap.
                
                delta = 0
                a, b, c, d = best_order[i-1] if i > 0 else None, best_order[i], best_order[i+1] if i < N-1 else None, best_order[j-1] if j > 0 else None
                # This is getting complicated with boundaries.
                
                # Let's write a helper to get score of a specific edge
                def get_edge(u, v):
                    if u is None or v is None:
                        return 0
                    return M[u][v]
                
                # Old edges
                old_score = 0
                if i > 0: old_score += get_edge(best_order[i-1], best_order[i])
                if i < N - 1: old_score += get_edge(best_order[i], best_order[i+1])
                if j > 0: old_score += get_edge(best_order[j-1], best_order[j])
                if j < N - 1: old_score += get_edge(best_order[j], best_order[j+1])
                
                # New edges (after swap)
                # best_order[i] <-> best_order[j]
                new_score = 0
                if i > 0: new_score += get_edge(best_order[i-1], best_order[j])
                if i < N - 1: new_score += get_edge(best_order[j], best_order[i+1])
                if j > 0: new_score += get_edge(best_order[j-1], best_order[i])
                if j < N - 1: new_score += get_edge(best_order[i], best_order[j+1])
                
                # Handle case where i and j are adjacent
                if j == i + 1:
                    # Edges involved: (i-1, i), (i, i+1), (i+1, i+2)
                    # Old: (i-1, i), (i, i+1), (i+1, i+2)
                    # New: (i-1, i+1), (i+1, i), (i, i+2)
                    # My generic formula above double counts or misses?
                    # If j = i+1:
                    # Old: E(i-1, i) + E(i, i+1) + E(i+1, i+2) [if i+1 < N-1]
                    # My formula:
                    # i>0: E(i-1, i)
                    # i<N-1: E(i, i+1)
                    # j>0: E(j-1, j) -> E(i, i+1) (Duplicate!)
                    # j<N-1: E(j, j+1) -> E(i+1, i+2)
                    # So if adjacent, we double count E(i, i+1).
                    
                    # Let's just recalculate the local neighborhood score properly.
                    # Simpler: Just swap and recalculate the 4 edges involved, being careful with indices.
                    
                    # Let's just do a simple full recalculation of the score if N is small?
                    # No, N=720.
                    
                    # Let's fix the delta calculation for Swap.
                    # Nodes involved: i-1, i, i+1, j-1, j, j+1.
                    # Since i < j, the sets are {i-1, i, i+1} and {j-1, j, j+1}.
                    # If j > i+1, they are disjoint.
                    # If j == i+1, they are {i-1, i, i+1, i+2}.
                    
                    if j > i + 1:
                        delta = (get_edge(best_order[i-1], best_order[j]) + get_edge(best_order[j], best_order[i+1]) + get_edge(best_order[j-1], best_order[i]) + get_edge(best_order[i], best_order[j+1])) \
                                - (get_edge(best_order[i-1], best_order[i]) + get_edge(best_order[i], best_order[i+1]) + get_edge(best_order[j-1], best_order[j]) + get_edge(best_order[j], best_order[j+1]))
                    else: # j == i+1
                        # Old: (i-1, i), (i, i+1), (i+1, i+2)
                        # New: (i-1, i+1), (i+1, i), (i, i+2)
                        old_s = get_edge(best_order[i-1], best_order[i]) + get_edge(best_order[i], best_order[i+1]) + get_edge(best_order[i+1], best_order[i+2])
                        new_s = get_edge(best_order[i-1], best_order[i+1]) + get_edge(best_order[i+1], best_order[i]) + get_edge(best_order[i], best_order[i+2])
                        delta = new_s - old_s
                        # Handle boundaries
                        if i == 0:
                            old_s = get_edge(best_order[0], best_order[1]) + get_edge(best_order[1], best_order[2])
                            new_s = get_edge(best_order[1], best_order[0]) + get_edge(best_order[0], best_order[2])
                            delta = new_s - old_s
                        if i == N - 2:
                            old_s = get_edge(best_order[i-1], best_order[i]) + get_edge(best_order[i], best_order[i+1])
                            new_s = get_edge(best_order[i-1], best_order[i+1]) + get_edge(best_order[i+1], best_order[i])
                            delta = new_s - old_s
                        if i == 0 and N == 2: # Unlikely for n>=3
                            delta = 0

                if delta > 0:
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    best_score += delta
                    improved = True
        
        # 2. Or-opt (Move)
        # Move block at i to position j (j > i)
        # This is O(N) to shift, but O(1) to calculate delta if we are careful?
        # Actually, shifting changes many edges.
        # Let's skip Or-opt for now if Swap is not enough, or implement it carefully.
        # Given time, Swap might be sufficient to find a good local optimum.
        
        # Let's add a simple "Insert" move: Remove i, insert at j.
        # This is expensive to update score (O(N)).
        # We can do it less frequently.
        
        if time.time() - start_time > time_limit:
            break
            
        # Random restart if stuck?
        # For now, just one pass of Swap.
        
    # Construct final word
    word = []
    for i, idx in enumerate(best_order):
        block = blocks[idx]
        word.extend(block)
        word.append(n)
        if i < N - 1:
             # We need to handle the overlap for the actual string construction?
             # No, the string is simply the concatenation of the segments?
             # Wait, the "length" we minimized was based on overlaps.
             # The actual string must be constructed by overlapping.
             # Segment i: B_i n B_i
             # Segment i+1: B_{i+1} n B_{i+1}
             # Overlap between end of Seg i (which is B_i) and start of Seg i+1 (which is B_{i+1}).
             # So we append B_{i+1} n B_{i+1} minus the overlapping prefix.
             
             # But wait, the first segment is B_0 n B_0.
             # Next is B_1 n B_1.
             # Overlap k = M[B_0][B_1].
             # So we add B_1[k:] + [n] + B_1?
             # No, the segment is B_1 n B_1.
             # The prefix of B_1 n B_1 is B_1.
             # So we add B_1[k:] + [n] + B_1.
             
             # Let's trace:
             # Current word ends with B_0.
             # Next part is B_1 n B_1.
             # Overlap k means B_0[-k:] == B_1[:k].
             # So we can start adding from B_1[k:].
             # Then add n.
             # Then add B_1.
             
             pass

    # Re-construct properly
    word = []
    for i, idx in enumerate(best_order):
        block = blocks[idx]
        if i == 0:
            word.extend(block)
            word.append(n)
            word.extend(block)
        else:
            prev_idx = best_order[i-1]
            prev_block = blocks[prev_idx]
            k = M[prev_idx][idx]
            # Add suffix of current block (excluding overlap)
            word.extend(block[k:])
            word.append(n)
            word.extend(block)
            
    return "".join(map(str, word))
