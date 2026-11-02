import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n >= 4, use the block TSP approach
    # Generate permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    num_blocks = len(prev_perms)
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    # Precompute overlaps
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Max overlap is len(block) - 1 = 2(n-1)
    # Since blocks are of form p + [n] + p, the overlap is limited.
    # Actually, we can just compute it.
    
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    # Optimization: Use string representation for faster slicing/comparison if needed, 
    # but lists are fine for n=7 (len 13).
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            # Max possible overlap is len(bi)-1
            # We check k from len(bi)-1 down to 1
            # Note: k=0 is always true
            found = 0
            # Optimization: The last char of bi is bi[-1] = p[-1]
            # The first char of bj is bj[0] = q[0]
            # So we only need to check if bi[-1] == bj[0]
            if bi[-1] != bj[0]:
                overlaps[i][j] = 0
            else:
                max_k = min(len(bi), len(bj))
                for k in range(max_k - 1, 0, -1):
                    if bi[-k:] == bj[:k]:
                        found = k
                        break
                overlaps[i][j] = found

    # Heuristic TSP: Nearest Neighbor
    # Start with block 0
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    
    while unvisited:
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    # 2-Opt Local Search
    # We want to maximize total overlap, which minimizes length.
    # Length = sum(len(blocks)) - sum(overlaps[path[k]][path[k+1]])
    # So we maximize sum(overlaps).
    
    def get_total_overlap(path):
        total = 0
        for k in range(len(path)-1):
            total += overlaps[path[k]][path[k+1]]
        return total
    
    current_overlap = get_total_overlap(path)
    
    improved = True
    # Time limit check: we have ~100s left.
    start_time = time.time()
    timeout = 100.0
    
    while improved:
        improved = False
        for i in range(len(path)-2):
            for j in range(i+1, len(path)-1):
                # Reverse path[i+1 : j+1]
                # Old segment: ... A, B, ..., C, D ...
                # A = path[i], B = path[i+1], C = path[j], D = path[j+1]
                # Old overlaps involved:
                # ov(A, B) + ov(B, path[i+2]) + ... + ov(path[j-1], C) + ov(C, D)
                # New segment: ... A, C, ..., B, D ...
                # New overlaps involved:
                # ov(A, C) + ov(C, path[j-1]) + ... + ov(path[i+2], B) + ov(B, D)
                
                # We need to calculate the change.
                # Let's just extract the subpath and calculate delta.
                # This is O(L) per swap. L=720.
                # 720^2 * 720 is too slow.
                
                # We need O(1) or O(log N) delta calculation.
                # Delta = (ov(A,C) + ov(C, path[j-1]) + ... + ov(path[i+2], B) + ov(B,D))
                #       - (ov(A,B) + ov(B, path[i+2]) + ... + ov(path[j-1], C) + ov(C,D))
                
                # The internal terms are reversed.
                # Sum of overlaps in reversed segment:
                # Old internal: sum_{k=i+1}^{j-1} ov(path[k], path[k+1])
                # New internal: sum_{k=i+1}^{j-1} ov(path[k], path[k-1])  (indices adjusted)
                # Actually, the set of pairs is the same, just direction reversed?
                # No, ov(X, Y) != ov(Y, X).
                
                # So we must sum the internal overlaps for the reversed segment.
                # This is O(j-i).
                # Total complexity O(N^3) worst case. 720^3 is ~3.7 * 10^8.
                # In Python, this might be too slow (seconds to minutes).
                
                # Let's try Or-opt (move one node) first? Or just a few 2-opt passes with early termination?
                # Or use a simpler heuristic: Random Restart + Nearest Neighbor?
                
                # Given the constraints, I'll implement a simplified 2-opt that breaks if time is running out.
                
                if time.time() - start_time > timeout:
                    break
                    
                # Calculate delta
                # A = path[i], B = path[i+1]
                # C = path[j], D = path[j+1]
                # Note: if j == len(path)-1, D doesn't exist.
                # If i == 0, A doesn't exist? No, path[0] is start.
                # The path is a sequence.
                
                old_cost = 0
                new_cost = 0
                
                # Boundary 1: A -> B vs A -> C
                if i > 0:
                    # Left neighbor of A is path[i-1]
                    # Old: ov(path[i-1], A) + ov(A, B)
                    # New: ov(path[i-1], A) + ov(A, C)
                    # Change: ov(A, C) - ov(A, B)
                    old_cost += overlaps[path[i]][path[i+1]]
                    new_cost += overlaps[path[i]][path[j]]
                else:
                    # No left neighbor for A (start of path)
                    # Old: ov(A, B)
                    # New: ov(A, C)
                    old_cost += overlaps[path[i]][path[i+1]]
                    new_cost += overlaps[path[i]][path[j]]
                
                # Boundary 2: C -> D vs B -> D
                if j < len(path) - 1:
                    # Right neighbor of D is path[j+1]
                    # Old: ov(C, D)
                    # New: ov(B, D)
                    old_cost += overlaps[path[j]][path[j+1]]
                    new_cost += overlaps[path[i+1]][path[j+1]]
                else:
                    # D is end of path
                    # Old: ov(C, D) -- wait, D is path[j+1]?
                    # If j is last index, D doesn't exist.
                    # My loop is j < len(path)-1.
                    # So j+1 is always valid index.
                    pass
                
                # Internal:
                # Old internal: sum_{k=i+1}^{j-1} ov(path[k], path[k+1])
                # New internal: sum_{k=i+1}^{j-1} ov(path[k], path[k-1])
                # Note: The segment is path[i+1] ... path[j].
                # Old order: p[i+1], p[i+2], ..., p[j]
                # New order: p[j], p[j-1], ..., p[i+1]
                
                # We can precompute prefix sums of overlaps?
                # No, overlaps are directional.
                
                # Let's just sum them.
                # If j-i is small, it's fast.
                # Average j-i is N/3 ~ 240.
                # 720^2 * 240 ~ 1.2 * 10^8. Still might be slow in Python.
                
                # Let's try a different local search: 1-swap (Or-opt 1).
                # Move one node to a new position.
                # Or just accept the greedy result if it's fast?
                # The greedy result is likely much better than random.
                
                # Let's try to speed up 2-opt by only checking a window of swaps?
                # Or just run 2-opt for a few seconds.
                
                # I will implement a time-bounded 2-opt.
                
                delta = new_cost - old_cost
                # Add internal difference
                # Old internal sum
                sum_old_int = 0
                for k in range(i+1, j):
                    sum_old_int += overlaps[path[k]][path[k+1]]
                
                sum_new_int = 0
                for k in range(i+1, j):
                    # New sequence at index i+1 is path[j]
                    # Index i+1: path[j]
                    # Index i+2: path[j-1]
                    # ...
                    # Index k: path[j - (k - (i+1))]
                    # Next index k+1: path[j - (k+1 - (i+1))] = path[j - (k - i)]
                    # So pair is (path[j - (k-i-1)], path[j - (k-i)])
                    # Let m = k - i - 1.
                    # Pair is (path[j-m-1], path[j-m])?
                    # Let's just iterate the reversed list.
                    pass
                
                # This is getting complicated.
                # I will use a simpler approach:
                # Just do a few iterations of "best improvement" 2-opt, but limit the number of checks.
                
                # Actually, for N=720, a single pass of 2-opt checking all pairs is ~250k checks.
                # Each check involves summing ~200 overlaps. 250k * 200 = 50M ops.
                # In Python, 50M ops might take 5-10 seconds.
                # We can do 10-20 passes.
                
                # Let's implement the delta calculation correctly and efficiently.
                
                # Precompute nothing, just sum.
                # To make it faster, I'll break out of the inner loop if time is up.
                
                # Re-calculate internal sums
                # Old internal: path[i+1] to path[j]
                # New internal: path[j] to path[i+1]
                
                # Let's just do it.
                
                if delta > 0:
                    # Check internal
                    # If delta is already > 0, internal might make it worse or better.
                    # We need the total delta.
                    
                    # Optimization: If the segment is small, calculate.
                    # If large, maybe skip?
                    
                    sum_old_int = 0
                    for k in range(i+1, j):
                        sum_old_int += overlaps[path[k]][path[k+1]]
                        
                    sum_new_int = 0
                    # Reversed path[i+1:j+1]
                    # The new sequence is path[j], path[j-1], ..., path[i+1]
                    # Overlaps: ov(path[j], path[j-1]), ov(path[j-1], path[j-2]), ..., ov(path[i+2], path[i+1])
                    for k in range(j, i, -1):
                        sum_new_int += overlaps[path[k]][path[k-1]]
                        
                    total_delta = (new_cost + sum_new_int) - (old_cost + sum_old_int)
                    
                    if total_delta > 0:
                        # Perform swap
                        path[i+1:j+1] = reversed(path[i+1:j+1])
                        current_overlap += total_delta
                        improved = True
                        # Restart i loop? Or continue?
                        # Continue to see if other swaps are possible.
                        
        if time.time() - start_time > timeout:
            break

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
