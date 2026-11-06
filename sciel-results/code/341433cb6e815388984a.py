import math
import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin from 90s limit
    
    N = 128
    elements = list(range(1, N + 1))
    
    # 1. Precompute GCD matrix
    gcd_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            gcd_mat[i][j] = math.gcd(elements[i], elements[j])
            
    # 2. Build Conflict Graph
    # A conflict exists between i and j if there exists k such that {i, j, k} is a bad triple.
    # Actually, the constraint is on triples. 
    # Let's define the graph G where vertices are elements.
    # An edge (i, j) exists if adding both i and j to a set creates a "potential" conflict 
    # that restricts the third element? No, that's not right.
    # The condition is: No three distinct elements a, b, c in S such that gcd(a,b)=gcd(a,c)=gcd(b,c).
    # This is a 3-uniform hypergraph independence problem.
    # However, we can model it as a Maximum Independent Set on a graph if we define conflicts pairwise?
    # Not directly. If {a,b,c} is bad, we cannot have all three.
    # This is a 3-SAT like problem or Hypergraph Independent Set.
    
    # But wait, look at the structure.
    # If gcd(a,b) = gcd(a,c) = gcd(b,c) = g, then a, b, c are all multiples of g.
    # Let a = g*a', b = g*b', c = g*c'.
    # gcd(a', b') = gcd(a', c') = gcd(b', c') = 1.
    # So a', b', c' are pairwise coprime.
    # This means that for any g, the set of multiples of g in S must not contain three numbers 
    # whose reduced forms are pairwise coprime.
    
    # Alternative View:
    # The "bad" triples are defined by the GCDs.
    # Let's stick to the bitmask approach for efficiency in checking, but use a different search strategy.
    # The current program uses Simulated Annealing. 
    # A different strategy: **Branch and Bound with Strong Pruning**.
    # We want to find the largest subset S of [1, 128] with no bad triples.
    
    # Precompute bad triples?
    # Total triples ~ 128^3 / 6 = 350,000.
    # We can precompute for each pair (i, j) the set of k's that form a bad triple with i and j.
    # bad_k[i][j] = bitmask of k such that {i, j, k} is bad.
    
    bad_k = [[0] * N for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            g = gcd_mat[i][j]
            mask = 0
            for k in range(N):
                if k == i or k == j:
                    continue
                if gcd_mat[i][k] == g and gcd_mat[j][k] == g:
                    mask |= (1 << k)
            bad_k[i][j] = mask
            
    # We want to maximize |S| such that for all distinct a,b,c in S, it is NOT the case that
    # gcd(a,b)=gcd(a,c)=gcd(b,c).
    
    # This is equivalent to: For all distinct a,b in S, S does not contain any k in bad_k[a][b].
    # i.e., (S & bad_k[a][b]) must be empty for all a,b in S, a!=b.
    # Note: bad_k[a][b] does not contain a or b.
    
    # This is a "Conflict Graph" if we consider edges?
    # No, it's a hypergraph. But we can use a recursive backtracking search with pruning.
    
    # Strategy:
    # 1. Greedy initialization to get a good lower bound (current best size).
    # 2. Branch and Bound:
    #    - Order elements by "conflict degree" (number of elements they conflict with).
    #    - Recursively decide to include or exclude each element.
    #    - Prune if current_size + remaining_elements <= best_size.
    #    - Prune if adding an element creates a conflict with existing elements?
    #      Actually, we maintain the set of "available" elements.
    #      If we include x, we must remove from 'available' any element y such that
    #      there exists z in current_set such that {x, y, z} is bad.
    #      Wait, if {x, y, z} is bad, we can't have x, y, z all in S.
    #      If x and z are in S, then y CANNOT be in S.
    #      So if we include x, and we already have z in S, then any y in bad_k[x][z] is forbidden.
    #      Also, if we include x, and we later include y, then any z in bad_k[x][y] is forbidden.
    #      
    #      Let's maintain:
    #      - current_set: list of elements included so far.
    #      - candidates: bitmask of elements that can still be added.
    #      
    #      When we include x:
    #      - Remove x from candidates.
    #      - For each z in current_set:
    #          - Any y in bad_k[x][z] cannot be added anymore (because x and z are in S).
    #          - So candidates &= ~bad_k[x][z]
    #      - Also, any y such that {x, y, ?} is bad? No, the third element isn't chosen yet.
    #      - But wait, if we include x, and later include y, we must ensure no z in S is in bad_k[x][y].
    #      - This is handled when we pick y.
    #      - Is there a conflict between x and y immediately?
    #      - No, a pair (x,y) is only "bad" if there is a third element z in S.
    #      - So including x doesn't immediately forbid y, unless there is already a z in S such that {x,y,z} is bad.
    #      - That is exactly: y is in bad_k[x][z] for some z in S.
    #      - So the update `candidates &= ~bad_k[x][z]` for all z in S is correct.
      
    # Greedy Initialization
    def greedy_init():
        # Sort by number of conflicts?
        # Let's just try a few random orders or heuristic orders.
        # Heuristic: Elements with fewer "bad partners" are better to include early?
        # Or elements that are "central" in the conflict graph?
        
        # Let's compute a score for each element: number of pairs (j,k) such that {i,j,k} is bad.
        # This is sum_{j != i} popcount(bad_k[i][j]).
        # Actually, just sum of bits in bad_k[i][j] for all j.
        # This is O(N^2) to compute.
        
        conflict_scores = [0] * N
        for i in range(N):
            for j in range(N):
                if i != j:
                    conflict_scores[i] += bad_k[i][j].bit_count()
        
        # Sort indices by conflict score (ascending? descending?)
        # If an element has high conflict score, it "kills" many others.
        # Maybe include low conflict elements first?
        # Or high conflict elements? If we include a high conflict element, we prune many.
        # Usually, in MIS, we pick high degree first.
        
        indices = list(range(N))
        indices.sort(key=lambda x: conflict_scores[x], reverse=True)
        
        S = 0
        current_set = []
        
        for i in indices:
            # Check if i can be added to current_set
            # i can be added if for all z in current_set, i is not in bad_k[z][?] 
            # Wait, the condition is: for all z in current_set, i must NOT be in bad_k[z][w] for any w in current_set?
            # No. The condition is: for all distinct z, w in current_set, i must not be in bad_k[z][w].
            # And for all z in current_set, the pair (i, z) must not form a bad triple with any existing w?
            # Yes.
            # So, we need to check:
            # For all z in current_set:
            #    Is i in bad_k[z][w] for any w in current_set (w != z)?
            #    This is equivalent to: (S & bad_k[z][i]) == 0 ?
            #    Let's check. bad_k[z][i] contains w such that {z, i, w} is bad.
            #    If S & bad_k[z][i] != 0, then there is a w in S such that {z, i, w} is bad.
            #    So we cannot add i.
            
            valid = True
            temp = S
            while temp:
                lsb = temp & -temp
                z = lsb.bit_length() - 1
                if S & bad_k[z][i]:
                    valid = False
                    break
                temp ^= lsb
                
            if valid:
                S |= (1 << i)
                current_set.append(i)
                
        return S, len(current_set)
        
    best_mask, best_size = greedy_init()
    
    # Refine with Branch and Bound
    # We will search for a larger set.
    # To make it fast, we use the greedy result as a lower bound.
    
    # Order of processing in B&B:
    # We should process elements in an order that allows strong pruning.
    # Let's use the conflict_scores order again.
    
    # Optimization: Precompute "conflict masks" for pairs?
    # We already have bad_k.
    
    # Recursive function
    # args: 
    #   idx: index in the ordered list of elements to consider
    #   current_mask: bitmask of selected elements
    #   current_size: number of selected elements
    #   candidates: bitmask of elements that are still allowed to be added
    #   (Note: candidates should be a subset of the remaining elements in the order)
    
    # To make it efficient, we pass the remaining elements as a bitmask `remaining`.
    # `candidates` is the intersection of `remaining` and elements that don't conflict with `current_mask`.
    
    # Actually, maintaining `candidates` dynamically is tricky with the ordered list.
    # Let's stick to:
    # `remaining`: bitmask of elements not yet decided (indices > idx in the order).
    # `allowed`: bitmask of elements in `remaining` that are compatible with `current_mask`.
    
    # If we include element `i` (from `allowed`):
    #   new_current_mask = current_mask | (1 << i)
    #   new_allowed = 0
    #   For each j in remaining \ {i}:
    #       if j is compatible with new_current_mask:
    #           new_allowed |= (1 << j)
    #   Compatibility check for j:
    #       For all z in current_mask, (new_current_mask \ (1<<i)) & bad_k[z][j] == 0?
    #       No, (current_mask & bad_k[z][j]) == 0 is not enough, because we added i.
    #       We need: (new_current_mask & bad_k[z][j]) == 0 for all z in new_current_mask \ {j}?
    #       Actually, simpler:
    #       j is compatible with new_current_mask if:
    #       For all z in new_current_mask, z != j:
    #           j is NOT in bad_k[z][w] for any w in new_current_mask?
    #           This is equivalent to: (new_current_mask & bad_k[z][j]) == 0 ?
    #           Wait. bad_k[z][j] contains w such that {z, j, w} is bad.
    #           If new_current_mask contains such a w, then conflict.
    #           So yes, condition is: for all z in new_current_mask, (new_current_mask & bad_k[z][j]) == 0.
    #           Note: z=j is not in new_current_mask \ {j}, but bad_k[j][j]=0.
    #           Also, if z is in new_current_mask, and w is in new_current_mask, and w in bad_k[z][j], then conflict.
    
    # This check is O(|new_current_mask|) per candidate j.
    # Total O(|new_current_mask| * |remaining|) per step.
    # |new_current_mask| is up to 60-80. |remaining| is up to 128.
    # 80 * 128 = 10k ops per node.
    # Depth 128.
    # Branching factor?
    # If we branch on "Include i" and "Exclude i", it's binary.
    # But we can prune "Exclude i" if i is not in `allowed`?
    # Actually, standard B&B for MIS:
    #   If `allowed` is empty, update best.
    #   If current_size + popcount(allowed) <= best_size, prune.
    #   Pick a vertex v from `allowed` (e.g., highest degree in induced subgraph).
    #   Branch 1: Include v.
    #       new_allowed = allowed \ {v} \ neighbors(v)
    #       Recurse.
    #   Branch 2: Exclude v.
    #       new_allowed = allowed \ {v}
    #       Recurse.
    
    # BUT, this is for Graph MIS.
    # Our problem is Hypergraph Independent Set.
    # The "neighbor" of v is not just a set of vertices, but depends on the current set.
    # So we can't precompute neighbors.
    
    # However, we can adapt the logic:
    # Pick v from `allowed`.
    # Branch 1: Include v.
    #   Calculate new_allowed:
    #   For each u in allowed \ {v}:
    #       Check if u is compatible with (current_mask U {v}).
    #       If yes, add to new_allowed.
    #   Recurse(new_allowed, current_mask U {v}, current_size + 1).
    # Branch 2: Exclude v.
    #   Recurse(allowed \ {v}, current_mask, current_size).
    
    # This is correct.
    # To make it fast:
    # 1. Use bitmasks.
    # 2. Prune aggressively.
    # 3. Choose v wisely.
    
    # Let's implement this.
    
    # Precompute bit_count for speed? Python 3.8+ has int.bit_count().
    
    # We need a global best_size and best_mask.
    
    # To avoid recursion limit issues, we can use an iterative stack or increase limit.
    # Depth is 128, so recursion limit 1000 is fine.
    sys.setrecursionlimit(1000)
    
    # Order of elements for B&B
    # Let's use the conflict_scores order.
    order = list(range(N))
    order.sort(key=lambda x: conflict_scores[x], reverse=True)
    
    # Map original index to position in order?
    # Or just iterate through the order list.
    # `allowed` is a bitmask over original indices.
    
    def is_compatible(u, mask):
        # Check if u can be added to set represented by mask.
        # For all z in mask:
        #   (mask & bad_k[z][u]) == 0
        # Note: bad_k[z][u] does not contain z or u.
        # So if mask contains z, and mask contains w in bad_k[z][u], then conflict.
        # The check (mask & bad_k[z][u]) == 0 ensures no w in mask is in bad_k[z][u].
        
        temp = mask
        while temp:
            lsb = temp & -temp
            z = lsb.bit_length() - 1
            if mask & bad_k[z][u]:
                return False
            temp ^= lsb
        return True
        
    # This is O(|mask|) per call.
    # We call it for each u in allowed.
    # So O(|mask| * |allowed|) per node.
    
    # We can optimize is_compatible by precomputing "conflict_mask[u][mask]"? No, mask is dynamic.
    # But we can precompute for each u, the list of z's?
    # Or, we can maintain a "forbidden" mask.
    # When we add v to mask, the new forbidden elements are:
    # union over z in mask of bad_k[z][v]
    # plus union over z in {v} of bad_k[v][z] for z in mask?
    # Actually, if we add v, the new conflicts for a candidate u are:
    # 1. u is in bad_k[z][v] for some z in old_mask. (Because z, v are in new_mask, so u cannot be).
    # 2. u is in bad_k[v][z] for some z in old_mask. (Symmetric).
    # 3. u is in bad_k[v][w] for some w in old_mask? No, w is in old_mask.
    #    Wait, bad_k[v][w] contains u such that {v, w, u} is bad.
    #    So if v and w are in new_mask, u cannot be.
    #    So yes, union over w in old_mask of bad_k[v][w].
    #    And union over z in old_mask of bad_k[z][v].
    #    Note that bad_k[z][v] and bad_k[v][z] are the same set?
    #    Let's check.
    #    bad_k[i][j] contains k such that gcd(i,j)=gcd(i,k)=gcd(j,k).
    #    This condition is symmetric in i and j.
    #    So bad_k[i][j] == bad_k[j][i].
    #    So we just need: new_forbidden = union_{z in old_mask} bad_k[z][v]
    
    # So, if we maintain a `forbidden` mask for the current `mask`:
    # `forbidden` = set of u such that u is NOT compatible with `mask`.
    # Initially, for empty mask, forbidden = 0.
    # When we add v:
    #   new_forbidden = forbidden | (union_{z in mask} bad_k[z][v])
    #   Note: v itself is not in forbidden? 
    #   bad_k[z][v] does not contain v.
    #   So v is not in new_forbidden.
    #   But we remove v from candidates anyway.
    
    # So we can maintain `forbidden` incrementally!
    # This reduces the cost of checking compatibility to O(1) per candidate u.
    # u is compatible iff (forbidden & (1<<u)) == 0.
    
    # Let's restructure the B&B with this optimization.
    
    # State:
    # - mask: current set
    # - forbidden: bitmask of elements that conflict with mask
    # - candidates: bitmask of elements not in mask, not in forbidden, and not yet decided?
    #   Actually, we can just use `candidates` as the set of available elements.
    #   candidates = ~mask & ~forbidden & all_ones
    
    # When we add v:
    #   new_mask = mask | (1<<v)
    #   new_forbidden = forbidden | (union_{z in mask} bad_k[z][v])
    #   We need to compute union_{z in mask} bad_k[z][v].
    #   This is O(|mask|) per step.
    #   But we do it once per branch.
    
    # Let's implement this.
    
    # Global variables for B&B
    global_best_size = best_size
    global_best_mask = best_mask
    
    # Precompute union_bad_k[v][mask]? No, mask is dynamic.
    # We compute it on the fly.
    
    def compute_new_forbidden(mask, v, old_forbidden):
        # old_forbidden is the forbidden mask for `mask`.
        # We add v.
        # New forbidden = old_forbidden | union_{z in mask} bad_k[z][v]
        
        new_forb = old_forbidden
        temp = mask
        while temp:
            lsb = temp & -temp
            z = lsb.bit_length() - 1
            new_forb |= bad_k[z][v]
            temp ^= lsb
        return new_forb
        
    # B&B Function
    def branch_and_bound(mask, forbidden, candidates, current_size):
        nonlocal global_best_size, global_best_mask
        
        if time.time() > DEADLINE:
            return
            
        if current_size + candidates.bit_count() <= global_best_size:
            return
            
        if candidates == 0:
            if current_size > global_best_size:
                global_best_size = current_size
                global_best_mask = mask
            return
            
        # Pick a vertex v from candidates
        # Heuristic: Pick v with highest "degree" in the induced subgraph?
        # Or just pick the first one?
        # Let's pick the one with highest conflict score (precomputed) to prune more?
        # Or the one that appears in the most bad_k masks?
        
        # For simplicity and speed, let's pick the lowest index in candidates?
        # Or the one with highest conflict_score.
        
        # Let's pick the one with highest conflict_score among candidates.
        # To do this fast, we can precompute a list of candidates and sort?
        # That's O(N log N) per node. Too slow.
        
        # Let's just pick the first set bit.
        # Or, we can precompute an array `conflict_score` and iterate?
        
        # Let's try picking the element with the highest conflict_score.
        # We can iterate through candidates bits and find the max score.
        # O(|candidates|) per node.
        
        v = -1
        max_score = -1
        temp_cand = candidates
        while temp_cand:
            lsb = temp_cand & -temp_cand
            u = lsb.bit_length() - 1
            if conflict_scores[u] > max_score:
                max_score = conflict_scores[u]
                v = u
            temp_cand ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        new_mask = mask | (1 << v)
        new_forbidden = compute_new_forbidden(mask, v, forbidden)
        # Remove v and new forbidden from candidates
        # Note: new_forbidden might contain elements already in mask?
        # bad_k[z][v] does not contain z or v.
        # So new_forbidden is disjoint from mask?
        # Not necessarily. If w is in mask, and w is in bad_k[z][v], then w is in new_forbidden.
        # But w is already in mask, so it's not in candidates.
        # So candidates &= ~new_forbidden is safe.
        
        new_candidates = candidates & ~(1 << v) & ~new_forbidden
        
        branch_and_bound(new_mask, new_forbidden, new_candidates, current_size + 1)
        
        # Branch 2: Exclude v
        # candidates without v
        # forbidden remains the same?
        # Yes, excluding v doesn't change the conflicts of the current mask.
        # But wait, if we exclude v, we just remove it from consideration.
        # The current mask is unchanged.
        # So forbidden is unchanged.
        
        branch_and_bound(mask, forbidden, candidates & ~(1 << v), current_size)
        
    # Initial call
    # mask = 0
    # forbidden = 0
    # candidates = (1 << N) - 1
    # current_size = 0
    
    # To improve initial bound, we can use the greedy result as a starting point?
    # No, B&B starts from scratch but prunes using the global best.
    # So we set global_best_size = best_size from greedy.
    
    # One optimization: Start B&B with the greedy set?
    # No, B&B explores the space.
    # But we can order the branches: Include first (to find large sets quickly).
    
    # Let's run it.
    # To avoid deep recursion stack overflow or slowness, we can add a depth limit or time check.
    
    # The number of nodes can explode.
    # We need to be careful.
    # Maybe limit the depth or use a heuristic to stop early?
    # The deadline guard is the main safety.
    
    # Let's run the B&B.
    # We pass the initial state.
    
    # Note: `conflict_scores` is computed outside.
    
    # Run B&B
    branch_and_bound(0, 0, (1 << N) - 1, 0)
    
    # If B&B didn't find anything better (or didn't run), use greedy result.
    # Actually, global_best_mask is updated inside.
    
    # Convert result to list
    result = []
    for i in range(N):
        if global_best_mask & (1 << i):
            result.append(elements[i])
            
    if not result:
        # Fallback
        result = [1, 2, 4, 8, 16, 32, 64, 128]
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Final size: {len(res)}")
    print(res)

