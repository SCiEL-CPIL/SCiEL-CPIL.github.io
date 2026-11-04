"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    if n <= 6:
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

    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
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

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        overlap_mat = [[0] * N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i][j] = 6
                    continue
                pj = perms[j]
                ov = 0
                # Max possible overlap is 5 (since they are distinct, 6 is impossible)
                for k in range(5, 0, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

        # Simulated Annealing
        # Initial solution: Nearest Neighbor
        random.seed(42)
        unvisited = list(range(N))
        random.shuffle(unvisited)
        path = [unvisited.pop()]
        
        while unvisited:
            last = path[-1]
            best_idx = -1
            best_ov = -1
            for idx in unvisited:
                ov = overlap_mat[last][idx]
                if ov > best_ov:
                    best_ov = ov
                    best_idx = idx
            path.append(best_idx)
            unvisited.remove(best_idx)

        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

        current_score = calc_total_overlap(path)
        best_score = current_score
        best_path = path[:]

        # SA Parameters
        T = 1.0
        cooling = 0.999
        min_T = 1e-6
        
        # Time management
        t_start = time.time()
        
        while time.time() < deadline - 10:
            if T < min_T:
                break
            
            # Random move: Swap two random positions
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            
            # Calculate delta
            # Edges involved: (prev, i), (i, i+1), (i+1, next) vs (prev, j), (j, j+1), (j+1, next)
            # This is complex for arbitrary swap. 
            # Simpler: 2-opt reversal is easier to calculate delta for.
            
            # Let's use 2-opt reversal
            if i > j:
                i, j = j, i
            if i == j:
                continue
                
            # Reverse path[i+1 ... j]
            # Edges changing:
            # 1. (path[i], path[i+1]) becomes (path[i], path[j])
            # 2. (path[j], path[j+1]) becomes (path[i+1], path[j+1])
            # (If j == N-1, edge 2 is not present, only edge 1 changes effectively? 
            #  No, if j==N-1, the segment is path[i+1...N-1].
            #  Old edges: (path[i], path[i+1]) and (path[N-2], path[N-1])? No.
            #  The path is a sequence.
            #  Old: ... A, B, C, D, E ... (A=path[i], B=path[i+1], ..., D=path[j-1], E=path[j])
            #  Reversed: ... A, E, D, C, B ...
            #  Edges removed: (A,B) and (D,E) [if j<N-1, then (E, next) is also involved? No, E is the end of segment]
            #  Let's be precise.
            #  Segment is path[i+1] to path[j].
            #  Left neighbor: L = path[i]
            #  Right neighbor: R = path[j+1] (if j < N-1)
            
            old_ov1 = overlap_mat[path[i]][path[i+1]]
            new_ov1 = overlap_mat[path[i]][path[j]]
            
            if j < N - 1:
                old_ov2 = overlap_mat[path[j]][path[j+1]]
                new_ov2 = overlap_mat[path[i+1]][path[j+1]]
                delta = (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
            else:
                # j is the last element
                # Only edge (path[i], path[i+1]) is removed.
                # New edge is (path[i], path[j]) which is (path[i], path[N-1]).
                # The edge (path[N-2], path[N-1]) is internal to the reversed segment?
                # No, the segment is path[i+1...N-1].
                # Old internal edges: (path[i+1], path[i+2]), ..., (path[N-2], path[N-1]).
                # New internal edges: (path[N-1], path[N-2]), ..., (path[i+2], path[i+1]).
                # Since the graph is directed, internal edge weights change!
                # My previous 2-opt delta calculation in the seed code was WRONG for directed graphs 
                # if it assumed symmetry or ignored internal edges.
                # Wait, the seed code:
                # "Two edges change"
                # It only considered the boundary edges.
                # This is only valid if the internal edges sum is invariant.
                # Sum of overlaps in a segment vs reversed segment.
                # $\sum_{k=i+1}^{j-1} w(p_k, p_{k+1})$ vs $\sum_{k=i+1}^{j-1} w(p_{k+1}, p_k)$.
                # These are NOT equal in general.
                # So the seed code's 2-opt was actually performing a "2-opt" that only changed boundary edges, 
                # effectively reversing the segment but keeping the internal connections? 
                # No, `path[i+1:j+1] = reversed(path[i+1:j+1])` changes the order.
                # So the internal edges change.
                # The seed code's delta calculation `new_ov1 + new_ov2 > old_ov1 + old_ov2` 
                # implicitly assumed internal edges don't change or are symmetric.
                # Overlap is NOT symmetric.
                # So the seed code's 2-opt was flawed! It accepted moves that might decrease internal overlap.
                # This explains why it might be stuck at a suboptimal solution.
                
                # We must calculate the FULL delta.
                # For N=720, calculating full delta for a long segment is O(N).
                # We can restrict segment length or use a different move.
                
                # Let's use Swap of two adjacent elements (1-opt) or Swap of two arbitrary elements.
                # Swap i, j:
                # Only 3 edges change: (i-1, i), (i, i+1), (i+1, i+2) ? No.
                # If we swap p_i and p_j:
                # Edges:
                # 1. (p_{i-1}, p_i) -> (p_{i-1}, p_j)
                # 2. (p_i, p_{i+1}) -> (p_j, p_{i+1})
                # 3. (p_{j-1}, p_j) -> (p_{j-1}, p_i)
                # 4. (p_j, p_{j+1}) -> (p_i, p_{j+1})
                # Plus if i+1 == j, edges merge.
                
                # Let's implement a robust Swap move.
                pass
            
            # To keep it simple and fast, let's use a "2-opt" that calculates the exact delta.
            # To avoid O(N) cost, we can precompute prefix sums? No, weights depend on indices.
            # We can just accept O(N) cost for 2-opt and limit the number of iterations.
            # Or, we can use "Or-Opt": move a segment of length 1, 2, or 3 to a new position.
            
            # Let's try Or-Opt 1 (move one node).
            # Remove p_k, insert between p_i and p_{i+1}.
            # Edges removed: (p_{k-1}, p_k), (p_k, p_{k+1}).
            # Edges added: (p_{k-1}, p_{k+1}), (p_i, p_k), (p_k, p_{i+1}).
            # Delta = w(p_{k-1}, p_{k+1}) + w(p_i, p_k) + w(p_k, p_{i+1}) - w(p_{k-1}, p_k) - w(p_k, p_{k+1}) - w(p_i, p_{i+1})
            # This is O(1)!
            
            k = random.randint(0, N - 1)
            if k == 0:
                # Move first element to position i
                # Remove p_0. Insert between p_i and p_{i+1}.
                # If i == 0, no change.
                # If i > 0:
                # Old edges: (p_0, p_1) and (p_i, p_{i+1}).
                # New edges: (p_1, ... p_i), (p_i, p_0), (p_0, p_{i+1}).
                # Wait, the segment p_1..p_i shifts left.
                # The edge (p_{i-1}, p_i) becomes (p_{i-1}, p_{i+1})? No.
                # Sequence: p_0, p_1, ..., p_i, p_{i+1}, ...
                # Remove p_0: p_1, ..., p_i, p_{i+1}, ...
                # Insert p_0 between p_i and p_{i+1}: p_1, ..., p_i, p_0, p_{i+1}, ...
                # Edges removed: (p_0, p_1) and (p_i, p_{i+1}).
                # Edges added: (p_i, p_0) and (p_0, p_{i+1}).
                # The edge (p_{i-1}, p_i) is unchanged? 
                # Original: ... p_{i-1}, p_i, p_{i+1} ...
                # New: ... p_{i-1}, p_i, p_0, p_{i+1} ...
                # Yes, (p_{i-1}, p_i) is unchanged.
                # So Delta = w(p_i, p_0) + w(p_0, p_{i+1}) - w(p_0, p_1) - w(p_i, p_{i+1})
                
                i = random.randint(1, N - 1) # Insert after i (so between i and i+1)
                # But if i == N-1, insert at end.
                # Let's define i as index of element BEFORE insertion point.
                # Insertion point is between path[i] and path[i+1].
                # If k == 0:
                #   Remove path[0].
                #   Insert between path[i] and path[i+1].
                #   i must be >= 1? If i=0, we are putting it back.
                #   If i=N-1, we put it at end.
                
                if i == 0:
                    delta = 0
                else:
                    old1 = overlap_mat[path[0]][path[1]]
                    old2 = overlap_mat[path[i]][path[i+1]]
                    new1 = overlap_mat[path[i]][path[0]]
                    new2 = overlap_mat[path[0]][path[i+1]]
                    delta = new1 + new2 - old1 - old2
                    
            elif k == N - 1:
                # Move last element to position i (between i and i+1)
                # Remove path[N-1].
                # Insert between path[i] and path[i+1].
                # i can be 0 to N-2.
                i = random.randint(0, N - 2)
                old1 = overlap_mat[path[N-2]][path[N-1]]
                old2 = overlap_mat[path[i]][path[i+1]]
                new1 = overlap_mat[path[i]][path[N-1]]
                new2 = overlap_mat[path[N-1]][path[i+1]]
                delta = new1 + new2 - old1 - old2

            else:
                # Move internal element k to position i (between i and i+1)
                # Remove path[k].
                # Edges removed: (path[k-1], path[k]) and (path[k], path[k+1]).
                # And the edge (path[i], path[i+1]) is split.
                # Edges added: (path[k-1], path[k+1]), (path[i], path[k]), (path[k], path[i+1]).
                # Careful if i is k-1 or k.
                
                i = random.randint(0, N - 2)
                
                # Case 1: i == k-1 (insert right before original position, i.e., no move or swap with next?)
                # If we remove k and insert between k-1 and k, it goes back to original spot.
                # Actually, if i == k-1, we insert between path[k-1] and path[k] (which is now path[k+1] after removal? No, indices shift).
                # Let's just handle general case and check for adjacency.
                
                if i == k - 1 or i == k:
                    # These are "no-op" or swap with neighbor.
                    # If i == k-1: Insert between path[k-1] and path[k].
                    # Original: ... A, K, B ...
                    # Remove K: ... A, B ...
                    # Insert K between A and B: ... A, K, B ...
                    # No change. Delta = 0.
                    delta = 0
                elif i == k + 1:
                    # Insert between path[k+1] and path[k+2]?
                    # Original: ... A, K, B, C ...
                    # Remove K: ... A, B, C ...
                    # Insert K between B and C: ... A, B, K, C ...
                    # This is a swap of K and B.
                    # Old edges: (A,K), (K,B), (B,C).
                    # New edges: (A,B), (B,K), (K,C).
                    # Delta = w(A,B) + w(B,K) + w(K,C) - w(A,K) - w(K,B) - w(B,C)
                    # Here i = k+1. But after removal, the element at index k is B (original k+1).
                    # So we insert between current path[k] (which is B) and path[k+1] (which is C).
                    # So i in the *original* indices is k+1.
                    # Let's just use the formula for general i, but be careful with indices.
                    
                    # Let's stick to the general formula:
                    # Removed: (path[k-1], path[k]), (path[k], path[k+1]), (path[i], path[i+1])
                    # Added: (path[k-1], path[k+1]), (path[i], path[k]), (path[k], path[i+1])
                    # This formula works if i is not k-1 or k.
                    # If i == k-1, we insert between path[k-1] and path[k+1] (since k is removed).
                    # This is the "no-op" case.
                    # If i == k, we insert between path[k] (which is path[k+1] in new list) and path[k+1] (which is path[k+2] in new list)?
                    # No, if we remove k, the element at index k becomes old k+1.
                    # So inserting at index k (in new list) means between old k+1 and old k+2.
                    # This is equivalent to moving k to position k+2?
                    
                    # To avoid index confusion, let's just calculate delta for the specific case of swapping adjacent elements.
                    # Swap k and k+1.
                    # Old: A, K, B, C
                    # New: A, B, K, C
                    # Delta = w(A,B) + w(B,K) + w(K,C) - w(A,K) - w(K,B) - w(B,C)
                    # This is a valid move.
                    # Let's just use this move type: Swap adjacent elements.
                    # It's simple and O(1).
                    pass # We'll use a different move type below

            # Let's restart the move selection with a simpler, robust strategy:
            # 1. Swap two random elements (O(1) delta if we handle boundaries correctly)
            # 2. Reverse a segment of length 2 (Swap adjacent)
            
            # Let's implement Swap Adjacent (2-opt of length 1)
            # This is equivalent to Or-Opt 1 where the destination is adjacent.
            
            # Actually, let's just use the Swap of two arbitrary elements i, j.
            # It is O(1) to calculate delta.
            # Edges:
            # If i < j:
            # 1. (i-1, i) -> (i-1, j)  [if i>0]
            # 2. (i, i+1) -> (j, i+1)
            # 3. (j-1, j) -> (j-1, i)
            # 4. (j, j+1) -> (i, j+1)  [if j<N-1]
            # Internal edges (i+1..j-1) are unchanged in value, just shifted?
            # No, the nodes are swapped. The edges between them are (i+1, i+2)...(j-1, j).
            # After swap, the nodes at these positions are the same, but the order is reversed?
            # No, we swap the VALUES at i and j. The nodes in between are unchanged.
            # So the edges (i+1, i+2) ... (j-1, j) are UNCHANGED.
            # Only the 4 boundary edges change.
            # This is perfect.
            
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            delta = 0
            
            # Edge 1: (i-1, i)
            if i > 0:
                delta -= overlap_mat[path[i-1]][path[i]]
                delta += overlap_mat[path[i-1]][path[j]]
            # Edge 2: (i, i+1)
            if i < N - 1:
                delta -= overlap_mat[path[i]][path[i+1]]
                # Note: path[i] is now path[j] (original), path[i+1] is unchanged
                delta += overlap_mat[path[j]][path[i+1]]
            # Edge 3: (j-1, j)
            if j > 0:
                delta -= overlap_mat[path[j-1]][path[j]]
                # Note: path[j] is now path[i] (original), path[j-1] is unchanged (unless j-1 == i)
                # If j-1 == i, then path[j-1] is path[i] (original).
                # We need to be careful.
                # Let's store the values.
                pass
            
            # This is getting messy. Let's just do the swap and recalculate the local score?
            # No, recalculation is O(N).
            
            # Let's use the Adjacent Swap (Or-Opt 1) which is simpler.
            # Move element k to position k+1 (swap k and k+1).
            # Old: A, K, B, C
            # New: A, B, K, C
            # Delta = w(A,B) + w(B,K) + w(K,C) - w(A,K) - w(K,B) - w(B,C)
            # Handles boundaries:
            # If k=0: A is dummy (start). w(start, K) is 0? No, there is no start node.
            # The path is p_0, p_1, ..., p_{N-1}.
            # Score = sum_{i=0}^{N-2} w(p_i, p_{i+1}).
            # Swap p_0 and p_1:
            # Old: w(p_0, p_1)
            # New: w(p_1, p_0)
            # Delta = w(p_1, p_0) - w(p_0, p_1)
            
            # Swap p_k and p_{k+1} for 0 < k < N-2:
            # Old: w(p_{k-1}, p_k) + w(p_k, p_{k+1}) + w(p_{k+1}, p_{k+2})
            # New: w(p_{k-1}, p_{k+1}) + w(p_{k+1}, p_k) + w(p_k, p_{k+2})
            # Delta = New - Old
            
            # Swap p_{N-2} and p_{N-1}:
            # Old: w(p_{N-3}, p_{N-2}) + w(p_{N-2}, p_{N-1})
            # New: w(p_{N-3}, p_{N-1}) + w(p_{N-1}, p_{N-2})
            # Delta = New - Old
            
            k = random.randint(0, N - 2)
            
            if k == 0:
                delta = overlap_mat[path[1]][path[0]] - overlap_mat[path[0]][path[1]]
            elif k == N - 2:
                delta = (overlap_mat[path[N-3]][path[N-1]] + overlap_mat[path[N-1]][path[N-2]]) - \
                        (overlap_mat[path[N-3]][path[N-2]] + overlap_mat[path[N-2]][path[N-1]])
            else:
                delta = (overlap_mat[path[k-1]][path[k+1]] + overlap_mat[path[k+1]][path[k]] + overlap_mat[path[k]][path[k+2]]) - \
                        (overlap_mat[path[k-1]][path[k]] + overlap_mat[path[k]][path[k+1]] + overlap_mat[path[k+1]][path[k+2]])
            
            # Accept move
            if delta > 0 or random.random() < math.exp(delta / T):
                path[k], path[k+1] = path[k+1], path[k]
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_path = path[:]
            
            T *= cooling

        # Use best path found
        path = best_path

        # Construct S_7 using the optimized path
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            # Find overlap with current new_word
            overlap = 0
            min_len = min(len(new_word), len(block))
            for o in range(min_len, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]

        return "".join(map(str, new_word))

    # Fallback for n > 7
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
# EVOLVE-BLOCK-END
