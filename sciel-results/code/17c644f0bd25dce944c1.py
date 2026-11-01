import time
import itertools
import numpy as np
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Base cases
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations of 1..n-1
    # For n=7, this is 720 permutations of length 6
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    N = len(perms)
    
    # Precompute overlap matrix
    # M[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # Max k is n-1 (length of perm is n-1, so max overlap is n-2? No, if perms are identical, overlap is n-1. But they are distinct, so max is n-2? 
    # Actually, if p1 = (1,2,3,4,5,6) and p2 = (2,3,4,5,6,1), overlap is 0? 
    # p1 suffix: 6, 56, 456, 3456, 23456
    # p2 prefix: 2, 23, 234, 2345, 23456
    # Overlap 23456 (length 5).
    # So max overlap is n-2 = 5 for n=7.
    M = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        pi = perms[i]
        for j in range(N):
            if i == j: continue
            pj = perms[j]
            # Check overlaps from max down to 1
            # Max possible overlap is n-2 because they are distinct permutations
            # If n-1=6, max overlap 5.
            max_k = n - 2 
            for k in range(max_k, 0, -1):
                if pi[-k:] == pj[:k]:
                    M[i][j] = k
                    break
    
    # We want to maximize sum of M[path[i]][path[i+1]]
    
    def calc_score(path):
        score = 0
        for i in range(len(path) - 1):
            score += M[path[i]][path[i+1]]
        return score

    def two_opt(path, current_score):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline:
                    return current_score
                u = path[i]
                v = path[i+1]
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        return current_score
                    x = path[j]
                    y = path[j+1] if j + 1 < N else None
                    if j == N - 1:
                        # Reversing path[i+1..N-1]
                        # Old: u-v ... x
                        # New: u-x ... v
                        # Delta: M[u][x] + M[x][path[j-1]] - M[u][v] - M[path[j-1]][x] ? No.
                        # Let's be precise.
                        # Path: ... u, v, ..., x, END
                        # Reverse v..x
                        # New: ... u, x, ..., v, END
                        # Edges changed: (u,v) and (prev_x, x) become (u,x) and (prev_x, v).
                        # Wait, if j=N-1, x is the last element.
                        # Old edges: (u,v) and (path[j-1], x).
                        # New edges: (u,x) and (path[j-1], v).
                        # Delta = M[u][x] + M[path[j-1]][v] - M[u][v] - M[path[j-1]][x]
                        if j == i + 1:
                            # Only 2 elements v, x
                            # Old: u-v, v-x (if x not last? No, x is last in this branch)
                            # If j=i+1, x=path[i+2]
                            # This logic is getting messy. Let's use a simpler delta calculation.
                            pass
                        # Generic 2-opt delta for reversing path[i+1..j]:
                        # Old edges: (path[i], path[i+1]) and (path[j], path[j+1]) [if j+1 < N]
                        # New edges: (path[i], path[j]) and (path[i+1], path[j+1]) [if j+1 < N]
                        # If j == N-1:
                        # Old: (path[i], path[i+1]) and (path[N-2], path[N-1])
                        # New: (path[i], path[N-1]) and (path[N-2], path[i+1])
                        # Delta = M[path[i]][path[N-1]] + M[path[N-2]][path[i+1]] - M[path[i]][path[i+1]] - M[path[N-2]][path[N-1]]
                        
                        # Let's implement a helper to get delta for reversal [i+1, j]
                        a = path[i]
                        b = path[i+1]
                        c = path[j]
                        d = path[j+1] if j + 1 < N else None
                        
                        if d is None:
                            # j is last index
                            # Reverse b..c
                            # Old: a-b, ... , prev_c-c
                            # New: a-c, ... , prev_c-b
                            # Delta = M[a][c] + M[path[j-1]][b] - M[a][b] - M[path[j-1]][c]
                            # Note: if j == i+1, path[j-1] is path[i] which is a.
                            # Delta = M[a][c] + M[a][b] - M[a][b] - M[a][c] = 0. Correct, reversing 2 elements doesn't change sum?
                            # Wait. If path is a, b, c.
                            # Reverse b..c -> a, c, b.
                            # Old score: M[a][b] + M[b][c]
                            # New score: M[a][c] + M[c][b]
                            # Delta = M[a][c] + M[c][b] - M[a][b] - M[b][c]
                            # My formula above: M[a][c] + M[path[j-1]][b] - M[a][b] - M[path[j-1]][c]
                            # If j=i+1, path[j-1]=a.
                            # Delta = M[a][c] + M[a][b] - M[a][b] - M[a][c] = 0.
                            # This is wrong. The edge (b,c) is also reversed to (c,b).
                            # The formula needs to account for the internal edge reversal?
                            # No, 2-opt only changes the boundary edges. The internal edges are reversed, but since M is not symmetric, M[x][y] != M[y][x].
                            # So we MUST sum the delta of all internal edges too?
                            # That makes 2-opt O(N^3) or worse if we sum internal edges.
                            # Standard 2-opt on asymmetric TSP:
                            # Delta = M[a][c] + M[b][d] - M[a][b] - M[c][d] (if d exists)
                            # Plus sum_{k=i+1}^{j-1} (M[path[k]][path[k-1]] - M[path[k-1]][path[k]])
                            # This is expensive.
                            
                            # Alternative: Use a simpler local search that is O(N^2) per pass.
                            # Or-opt: Move a segment.
                            # Or just use 2-opt assuming symmetric? No, M is not symmetric.
                            
                            # Let's use a different approach: Simulated Annealing or simple Random Swap.
                            # Or, since N=720 is small, we can afford O(N^2) local search.
                            
                            # Let's stick to 2-opt but calculate delta correctly.
                            # For asymmetric TSP, 2-opt delta calculation is:
                            # Delta = M[a][c] + M[b][d] - M[a][b] - M[c][d] + sum_{k=i+1}^{j-1} (M[path[k]][path[k-1]] - M[path[k-1]][path[k]])
                            # This sum is O(N) in the worst case, making 2-opt O(N^3).
                            # For N=720, 720^3 is ~370 million. Too slow in Python.
                            
                            # Better approach: Or-opt (move 1, 2, or 3 nodes).
                            # Or-opt is O(N^2) per pass if implemented carefully?
                            # Moving a node u from pos p to pos q:
                            # Delta = M[prev_u][next_u] + M[prev_q][next_q] - M[prev_u][next_q] - M[prev_q][u] - M[u][next_q] + M[u][next_u] ... complex.
                            
                            # Let's try a simpler heuristic: Greedy + Hill Climbing (swap adjacent).
                            # Swap adjacent elements i, i+1.
                            # Delta = M[path[i-1]][path[i+1]] + M[path[i+1]][path[i+2]] - M[path[i-1]][path[i]] - M[path[i]][path[i+1]] - M[path[i+1]][path[i+2]] + M[path[i]][path[i+2]] ?
                            # No, swap i, i+1:
                            # Old: ... A, X, Y, B ...
                            # New: ... A, Y, X, B ...
                            # Delta = M[A][Y] + M[Y][X] + M[X][B] - M[A][X] - M[X][Y] - M[Y][B]
                            # This is O(1) per swap.
                            # Hill climbing with adjacent swaps is slow to converge but safe.
                            
                            # Let's use 2-opt but approximate or use a faster method.
                            # Actually, for N=720, even O(N^2) is ~500k ops.
                            # If we do 1000 passes, 500M ops. Might be tight in 110s.
                            
                            # Let's use Or-opt for segments of length 1, 2, 3.
                            # This is standard and effective.
                            
                            pass # Placeholder for logic
                    else:
                        # j < N-1
                        # Old: a-b, c-d
                        # New: a-c, b-d
                        # Delta = M[a][c] + M[b][d] - M[a][b] - M[c][d]
                        # Plus internal reversal sum.
                        pass
            return current_score

    # Given the complexity of asymmetric 2-opt delta, let's use a simpler local search:
    # Random Swap Hill Climbing.
    # Pick two random indices i, j. Swap path[i] and path[j].
    # Calculate delta. If positive, accept.
    # This is O(N^2) to find best swap, or O(1) for random swap.
    # Random swap hill climbing is slow.
    
    # Better: 2-opt with precomputed prefix sums? No, M is not additive in a simple way.
    
    # Let's try a different strategy:
    # The problem is equivalent to finding a maximum weight Hamiltonian path.
    # We can use a simple greedy algorithm to build the path, then improve it.
    
    # Greedy Construction:
    # Start with a random permutation.
    # At each step, append the unvisited node with the max overlap to the current last node.
    
    # Then, apply local search:
    # 1. Try all adjacent swaps. If improvement, take it. Repeat until no improvement.
    # 2. Try all 2-opt reversals (calculating delta properly).
    
    # Let's implement 2-opt with proper delta calculation.
    # To make it fast, we can precompute the "cost" of the path.
    # But delta for reversal [i+1, j] requires summing internal edge changes.
    # We can maintain a prefix sum of "directed edge costs" vs "reverse edge costs"?
    # Let S[i] = M[path[i]][path[i+1]].
    # Let R[i] = M[path[i+1]][path[i]].
    # Sum of internal edges in reversed segment [i+1, j]:
    # Original internal edges: (i+1, i+2), ..., (j-1, j). Cost = sum_{k=i+1}^{j-1} M[path[k]][path[k+1]].
    # Reversed internal edges: (j, j-1), ..., (i+2, i+1). Cost = sum_{k=i+1}^{j-1} M[path[k+1]][path[k]].
    # So we need prefix sums of M[path[k]][path[k+1]] and M[path[k+1]][path[k]].
    
    # Let P1[i] = sum_{k=0}^{i-1} M[path[k]][path[k+1]]
    # Let P2[i] = sum_{k=0}^{i-1} M[path[k+1]][path[k]]
    
    # Then sum_{k=i+1}^{j-1} M[path[k]][path[k+1]] = P1[j] - P1[i+1]
    # And sum_{k=i+1}^{j-1} M[path[k+1]][path[k]] = P2[j] - P2[i+1]
    
    # This allows O(1) delta calculation for 2-opt!
    
    def two_opt_fast(path):
        N = len(path)
        # Calculate prefix sums
        P1 = np.zeros(N) # P1[i] is sum of edges 0..i-1
        P2 = np.zeros(N)
        for i in range(N - 1):
            P1[i+1] = P1[i] + M[path[i]][path[i+1]]
            P2[i+1] = P2[i] + M[path[i+1]][path[i]]
        
        total_score = P1[N-1]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline:
                    break
                a = path[i]
                b = path[i+1]
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        break
                    c = path[j]
                    d = path[j+1] if j + 1 < N else None
                    
                    # Calculate delta
                    # Old boundary edges: (a,b) and (c,d) [if d exists]
                    # New boundary edges: (a,c) and (b,d) [if d exists]
                    
                    # Internal edges change from forward to reverse
                    # Old internal sum: P1[j] - P1[i+1]
                    # New internal sum: P2[j] - P2[i+1]
                    
                    old_internal = P1[j] - P1[i+1]
                    new_internal = P2[j] - P2[i+1]
                    
                    if d is None:
                        # j is last index
                        # Old boundary: M[a][b] + M[path[j-1]][c]
                        # New boundary: M[a][c] + M[path[j-1]][b]
                        # Note: path[j-1] is the element before c.
                        # If j == i+1, path[j-1] is b.
                        # Old boundary: M[a][b] + M[b][c]
                        # New boundary: M[a][c] + M[b][b]?? No.
                        # If j == i+1, we are reversing b..c.
                        # Path: a, b, c.
                        # New: a, c, b.
                        # Old edges: (a,b), (b,c).
                        # New edges: (a,c), (c,b).
                        # Delta = M[a][c] + M[c][b] - M[a][b] - M[b][c].
                        
                        # Using formula:
                        # old_internal = 0 (empty range)
                        # new_internal = 0
                        # Old boundary: M[a][b] + M[path[j-1]][c]
                        # If j > i+1, path[j-1] is not b.
                        # Let's handle j == i+1 separately.
                        
                        if j == i + 1:
                            delta = M[a][c] + M[c][b] - M[a][b] - M[b][c]
                        else:
                            # path[j-1] is the element before c.
                            prev_c = path[j-1]
                            old_boundary = M[a][b] + M[prev_c][c]
                            new_boundary = M[a][c] + M[prev_c][b]
                            delta = new_boundary - old_boundary + (new_internal - old_internal)
                    else:
                        # j < N-1
                        # Old boundary: M[a][b] + M[c][d]
                        # New boundary: M[a][c] + M[b][d]
                        old_boundary = M[a][b] + M[c][d]
                        new_boundary = M[a][c] + M[b][d]
                        delta = new_boundary - old_boundary + (new_internal - old_internal)
                    
                    if delta > 0:
                        # Perform reversal
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        total_score += delta
                        improved = True
                        # Update prefix sums? 
                        # This is expensive. 
                        # Alternative: Don't use prefix sums, just recalculate score? No.
                        # Alternative: Use a list and update prefix sums locally?
                        # Or, just use the simple O(1) delta for adjacent swaps first, then 2-opt without prefix sums (slower but simpler)?
                        # Given N=720, even O(N^2) 2-opt with O(N) delta calc is O(N^3) = 370M.
                        # In Python, 370M ops is ~10-30s.
                        # So we can afford O(N) delta calc if we limit passes.
                        
                        # Let's restart the inner loop to be safe with prefix sums invalidation.
                        break
                if improved:
                    # Recalculate prefix sums
                    P1 = np.zeros(N)
                    P2 = np.zeros(N)
                    for i in range(N - 1):
                        P1[i+1] = P1[i] + M[path[i]][path[i+1]]
                        P2[i+1] = P2[i] + M[path[i+1]][path[i]]
                    break # Restart outer loop
        return total_score

    # Initial path: Identity
    best_path = list(range(N))
    best_score = two_opt_fast(best_path)
    
    # Random restarts
    for r in range(50):
        if time.time() > deadline:
            break
        path = list(range(N))
        random.shuffle(path)
        score = two_opt_fast(path)
        if score > best_score:
            best_score = score
            best_path = path[:]
            
    # Construct the final word
    first_p = perms[best_path[0]]
    word = list(first_p) + [n] + list(first_p)
    
    for idx in range(1, N):
        p = perms[best_path[idx]]
        block = list(p) + [n] + list(p)
        ov = M[best_path[idx-1]][best_path[idx]]
        word += block[ov:]
        
    return "".join(map(str, word))
