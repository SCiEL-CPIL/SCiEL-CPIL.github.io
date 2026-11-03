import time
import itertools
import math
import sys

def _rank_perm(p, n):
    """Compute lexicographic rank of permutation p (0-indexed) in 0..n!-1."""
    # p is a tuple of integers 1..n
    # For n=7, this is slow if called many times, but we only use it for validation or specific logic.
    # In the new algorithm, we might not need ranking if we just track sets.
    pass

def _overlap(a, b):
    """
    Calculate the maximum overlap between the end of string a and the start of string b.
    a and b are strings or lists of chars/ints.
    """
    max_ov = 0
    # The overlap cannot be larger than min(len(a), len(b))
    # Also, for superpermutations of distinct symbols, we can be smarter.
    # But general check is fine for small sizes (len ~ 13 for n=7 blocks)
    min_len = min(len(a), len(b))
    for o in range(min_len, -1, -1):
        if o == 0:
            return 0
        if a[len(a) - o:] == b[:o]:
            return o
    return 0

def _greedy_tsp(order, cost_matrix, n_nodes, time_limit):
    """
    Optimize the order using 2-Opt to maximize overlap (minimize cost = -overlap).
    cost_matrix is overlap matrix. We want to maximize sum of overlaps.
    """
    current_order = order[:]
    n = n_nodes
    current_cost = 0
    
    # Calculate initial cost
    for i in range(n - 1):
        current_cost += cost_matrix[current_order[i]][current_order[i+1]]
    
    improved = True
    start_time = time.time()
    
    while improved:
        improved = False
        if time.time() > time_limit:
            break
            
        for i in range(n - 1):
            if time.time() > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    break
                
                # Calculate change in cost if we reverse segment [i+1, j]
                # Old edges: (i-1, i), (i, i+1), (j, j+1), (j+1, j+2) ??
                # Let's be precise.
                # We are considering reversing the subpath from node at index i to node at index j.
                # Actually, standard 2-opt swaps two edges.
                # Let's use standard 2-opt: remove edges (u, v) and (x, y), add (u, x) and (v, y).
                # In a path 0..n-1:
                # Edges are (0,1), (1,2), ..., (n-2, n-1).
                # If we reverse segment [i, j], the edges affected are:
                # (i-1, i) becomes (i-1, j)
                # (i, i+1) ... (j-1, j) are reversed and internal to segment, but costs are symmetric?
                # Overlap is NOT symmetric. a->b != b->a.
                # So we must recalculate the whole cost or be very careful.
                
                # Given n=720, n^2 is 518k. Recalculating cost O(n) for each 2-opt move is O(n^3) ~ 300 billion. Too slow.
                # We need to calculate delta cost in O(1) or O(log n).
                
                # Delta cost for reversing segment [i, j] in path 0..n-1:
                # Old: ... A -- B ... C -- D ...
                # Where B is node i, C is node j.
                # Edges removed: (A, B) i.e., (i-1, i) and (C, D) i.e., (j, j+1).
                # Edges added: (A, C) i.e., (i-1, j) and (B, D) i.e., (i, j+1).
                # Wait, if we reverse the middle part, the internal edges are reversed.
                # Cost of internal edges changes from sum(overlap(v_k, v_{k+1})) to sum(overlap(v_{k+1}, v_k)).
                # This requires precomputed prefix sums of forward and reverse costs.
                
                # Let's precompute:
                # fwd_cost[i] = overlap(order[i], order[i+1])
                # rev_cost[i] = overlap(order[i+1], order[i])
                
                # This 2-opt implementation is getting complex and risky for correctness under time pressure.
                # Alternative: Just use a simple "Nearest Neighbor" + "First Improvement" 2-opt where we only check local moves or use a smaller time limit.
                # Or, since we just need to beat 5913, a good Greedy might be enough?
                # Greedy on 720 nodes:
                # Start with a node. Pick neighbor with max overlap.
                # This is O(n^2).
                
                # Let's try a simpler optimization: 
                # 1. Greedy construction.
                # 2. Random restarts or local search with a fixed number of iterations (not until convergence).
                
                # Let's implement a time-boxed 2-opt that recalculates delta by checking the boundary edges and assuming internal edges don't change cost? 
                # NO, internal edges DO change cost because overlap is directional.
                
                # However, for the specific structure of superpermutations, the blocks are $P + [k] + P$.
                # Is the overlap matrix symmetric?
                # $A = p_A + [k] + p_A$
                # $B = p_B + [k] + p_B$
                # Overlap $A \to B$: suffix of $A$ matches prefix of $B$.
                # Suffix of $A$ is $p_A$. Prefix of $B$ is $p_B$.
                # So we need suffix of $p_A$ to match prefix of $p_B$.
                # Overlap $B \to A$: suffix of $p_B$ matches prefix of $p_A$.
                # These are generally different.
                
                # Let's just do a simple "Steepest Descent" on the order, but limit the number of swaps.
                # Or, use the fact that $n=7$ is small enough that we can do a "Simulated Annealing" for a fixed number of steps.
                
                # Given the time constraint (120s), and that we do this for n=2..7:
                # n=6: 120 nodes. 2-opt is fast.
                # n=7: 720 nodes.
                
                # Let's implement a simple "Random Swap" local search with a time limit.
                # If we find a better solution, keep it.
                pass

    return current_order

def construct(n):
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Build superpermutations recursively
    # S[1] = "1"
    # S[k+1] from S[k]
    
    superperm = "1"
    
    for k in range(2, n + 1):
        # Get all permutations of 1..k-1
        symbols = list(range(1, k))
        perms = list(itertools.permutations(symbols))
        
        # Create blocks: P + [k] + P
        # Represent as strings for easy overlap calculation?
        # Or lists.
        # Let's use strings for simplicity in overlap check, but convert back to list at end.
        # For k=7, symbols are 1..6.
        # Blocks are length 2*(k-1)+1 = 13.
        
        blocks = []
        for p in perms:
            # p is tuple of ints
            # block = str(p[0]) + ... + str(k) + ...
            # Actually, just join
            block_str = "".join(map(str, p)) + str(k) + "".join(map(str, p))
            blocks.append(block_str)
        
        n_blocks = len(blocks)
        
        # Compute overlap matrix
        # overlap[i][j] = _overlap(blocks[i], blocks[j])
        # To save memory/time, we can compute on the fly in the search,
        # but 720x720 is small.
        
        # Precompute overlaps
        # This is O(n_blocks^2 * len(block))
        # 720^2 * 13 ~ 6.7 million char comparisons. Fast.
        
        overlap_matrix = [[0] * n_blocks for _ in range(n_blocks)]
        for i in range(n_blocks):
            b_i = blocks[i]
            for j in range(n_blocks):
                if i == j:
                    continue
                b_j = blocks[j]
                # Calculate overlap
                # Max possible overlap is len(b_i)-1? No, can't be full overlap unless identical, which they aren't.
                # Suffix of b_i matches prefix of b_j.
                # b_i ends with p_i. b_j starts with p_j.
                # So we only need to check suffix of p_i vs prefix of p_j?
                # No, the 'k' in the middle might be part of the overlap if the overlap extends across it.
                # But 'k' is unique in the block? No, k is the new symbol.
                # In block $P+k+P$, the symbol k appears once.
                # If the overlap includes k from block i, it must match a k in block j.
                # Block j has k at position len(P).
                # Block i has k at position len(P).
                # If we overlap suffix of i with prefix of j:
                # If the overlap length $L >$ len(P), then it includes the k from i.
                # Then the k in i must match the k in j.
                # This implies the suffix of i after k matches prefix of j up to k.
                # i: ... p_i_suffix + k
                # j: p_j_prefix + k ...
                # So p_i_suffix (part of p_i) must match p_j_prefix.
                # This is consistent with just matching the full strings.
                
                # Simple loop is fine.
                max_ov = 0
                min_len = min(len(b_i), len(b_j))
                # We can optimize: The overlap must be < len(b_i) and < len(b_j).
                # Also, since symbols are distinct within a block? No, P has distinct, k is new.
                # So all chars in block are distinct?
                # P is a permutation of 1..k-1. k is distinct.
                # So yes, all characters in a block are distinct.
                # This means the overlap length is at most k (if k matches k)?
                # If k in i matches k in j, the overlap is exactly len(P) + 1?
                # No.
                # Let's just do the standard string overlap.
                for o in range(min_len - 1, -1, -1):
                    if b_i[len(b_i) - o:] == b_j[:o]:
                        max_ov = o
                        break
                overlap_matrix[i][j] = max_ov

        # Solve TSP to maximize total overlap
        # Use Greedy + Local Search
        
        # 1. Greedy
        unvisited = set(range(n_blocks))
        # Start with node 0? Or random?
        # Let's start with 0 for determinism, or random for variety?
        # Deterministic is safer for debugging, but Random might find better.
        # Let's try starting with 0.
        
        current_path = [0]
        unvisited.remove(0)
        
        while unvisited:
            last = current_path[-1]
            best_next = None
            best_ov = -1
            for u in unvisited:
                ov = overlap_matrix[last][u]
                if ov > best_ov:
                    best_ov = ov
                    best_next = u
            
            current_path.append(best_next)
            unvisited.remove(best_next)
        
        # Calculate initial score
        total_ov = sum(overlap_matrix[current_path[i]][current_path[i+1]] for i in range(n_blocks - 1))
        
        # 2. Local Search (2-Opt)
        # We will perform random swaps and accept if better.
        # Time budget for this level: e.g., 5 seconds for n=7, less for smaller.
        time_budget = 10.0 if k == 7 else 2.0
        level_deadline = time.time() + time_budget
        
        import random
        random.seed(42) # Deterministic
        
        improved = True
        while improved and time.time() < level_deadline:
            improved = False
            # Try random pair of swaps
            # Or 2-opt: reverse segment
            # Let's do a simple "swap two positions" local search first, it's O(1) delta if we can compute it.
            # Delta for swapping i and j:
            # Old edges: (i-1, i), (i, i+1), (j-1, j), (j, j+1)
            # New edges: (i-1, j), (j, i+1), (j-1, i), (i, j+1)
            # Need to handle boundaries.
            
            for _ in range(1000): # Limit iterations per while loop to avoid long stalls
                if time.time() > level_deadline:
                    break
                
                i, j = random.sample(range(n_blocks), 2)
                if abs(i - j) < 2:
                    continue
                
                # Calculate delta
                # Get indices
                # Note: if i and j are adjacent, it's just a 2-swap, but we skip abs < 2.
                
                # Helper to get edge cost
                def get_cost(a, b):
                    if a is None or b is None:
                        return 0
                    return overlap_matrix[current_path[a]][current_path[b]]
                
                # Nodes involved:
                # i-1, i, i+1
                # j-1, j, j+1
                
                # If i and j are "far" apart:
                # Old: C(i-1, i) + C(i, i+1) + C(j-1, j) + C(j, j+1)
                # New: C(i-1, j) + C(j, i+1) + C(j-1, i) + C(i, j+1)
                # Wait, swapping i and j in the list:
                # ... A, X, B ... Y, Z, W ...
                # Swap X and Y:
                # ... A, Y, B ... X, Z, W ...
                # Edges changed:
                # (A, X) -> (A, Y)
                # (X, B) -> (Y, B)
                # (Y, Z) -> (X, Z)  <-- Wait, Y moves to i, Z stays at j+1.
                # (W, Y) ?? No, W is after Z.
                
                # Let's be precise.
                # Path: ... u_{i-1}, u_i, u_{i+1} ... u_{j-1}, u_j, u_{j+1} ...
                # Swap u_i and u_j.
                # New Path: ... u_{i-1}, u_j, u_{i+1} ... u_{j-1}, u_i, u_{j+1} ...
                
                # Removed Edges:
                # e1: (u_{i-1}, u_i)
                # e2: (u_i, u_{i+1})
                # e3: (u_{j-1}, u_j)
                # e4: (u_j, u_{j+1})
                
                # Added Edges:
                # f1: (u_{i-1}, u_j)
                # f2: (u_j, u_{i+1})
                # f3: (u_{j-1}, u_i)
                # f4: (u_i, u_{j+1})
                
                # Handle boundaries (i=0, j=n-1)
                cost_removed = 0
                cost_added = 0
                
                # Left of i
                if i > 0:
                    cost_removed += overlap_matrix[current_path[i-1]][current_path[i]]
                    cost_added += overlap_matrix[current_path[i-1]][current_path[j]]
                else:
                    # No left edge for i
                    pass
                
                # Right of i (now right of j)
                if i < n_blocks - 1:
                    cost_removed += overlap_matrix[current_path[i]][current_path[i+1]]
                    # The node at i+1 is now connected to j (which is at i)
                    cost_added += overlap_matrix[current_path[j]][current_path[i+1]]
                else:
                    pass
                
                # Left of j (now left of i)
                if j > 0:
                    cost_removed += overlap_matrix[current_path[j-1]][current_path[j]]
                    # The node at j-1 is now connected to i (which is at j)
                    cost_added += overlap_matrix[current_path[j-1]][current_path[i]]
                else:
                    pass
                
                # Right of j (now right of i)
                if j < n_blocks - 1:
                    cost_removed += overlap_matrix[current_path[j]][current_path[j+1]]
                    # The node at j+1 is now connected to i (which is at j)
                    cost_added += overlap_matrix[current_path[i]][current_path[j+1]]
                else:
                    pass
                
                delta = cost_added - cost_removed
                if delta > 0:
                    # Perform swap
                    current_path[i], current_path[j] = current_path[j], current_path[i]
                    total_ov += delta
                    improved = True
        
        # Assemble the final word
        final_word = blocks[current_path[0]]
        for idx in range(1, n_blocks):
            prev = current_path[idx-1]
            curr = current_path[idx]
            ov = overlap_matrix[prev][curr]
            final_word += blocks[curr][ov:]
        
        superperm = final_word
        
        if time.time() > deadline:
            break
            
    return superperm

if __name__ == "__main__":
    # Test with n=7
    result = construct(7)
    print(len(result))
