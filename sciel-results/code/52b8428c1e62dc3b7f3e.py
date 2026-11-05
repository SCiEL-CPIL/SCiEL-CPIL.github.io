"""
Ternary covering code K_3(7,1) via maximum independent set in conflict graph.

Strategy:
1. Build the conflict graph: vertices = all 2187 words, edge between two words
   if their Hamming distance <= 2 (i.e., they cannot both be in a radius-1 code).
2. A covering code of radius 1 is exactly an independent set in this graph
   that also covers all vertices (each vertex is within distance 1 of some codeword).
   Actually, an independent set in the distance<=2 graph is a code with minimum
   distance >= 3. But we need a covering code, not just a code.
   
   Wait - let me reconsider. A radius-1 covering code means every word is within
   distance 1 of some codeword. The codewords themselves must be distinct but don't
   need to be far apart. However, if two codewords are at distance 1, we could
   potentially remove one. If they're at distance 2, we might still need both.
   
   Actually, the right formulation: we want a minimum-size set C such that
   for every word w, dist(w, C) <= 1. This is the minimum dominating set problem
   on the graph where edges connect words at distance 1 (the Hamming graph H(7,3)).
   
   But the directive says to model it as MIS in the conflict graph (distance <= 2).
   Let me think about this more carefully.
   
   In a radius-1 covering code, if two codewords c1, c2 have dist(c1,c2) <= 1,
   then c2 is in the neighborhood of c1 (or vice versa). We might be able to
   remove one. But it's not always safe.
   
   Actually, the standard approach for covering codes is:
   - A set C is a radius-1 covering code iff C is a dominating set in the Hamming graph.
   - Minimum dominating set is NP-hard, but for this specific instance we can use
     branch-and-bound.
   
   However, the directive specifically says: "Model the covering code as a maximum 
   independent set in the conflict graph (edges between codewords at distance ≤ 2)".
   
   Let me reconsider: If we have a set S that is an independent set in the 
   distance<=2 graph (i.e., all pairs in S have distance >= 3), then S is a code
   with minimum distance 3. Such a code is NOT necessarily a covering code.
   
   But wait - maybe the idea is different. Let me re-read: "solve it using a custom 
   branch-and-bound algorithm with bitset-based clique lower bounds and dynamic 
   variable ordering, leveraging the graph's sparsity to directly find the optimal 
   186-word solution".
   
   I think the correct interpretation is:
   - We want a minimum dominating set in the Hamming graph H(7,3).
   - Equivalently, we want to find the minimum number of vertices such that every
     vertex is either in the set or adjacent to a vertex in the set.
   
   Let me use a different approach: branch-and-bound for minimum dominating set,
   using bitset representations for efficiency.
   
   Actually, let me try yet another angle. The problem can be formulated as:
   - Select minimum number of positions from {0,...,2186} such that every position
     is within distance 1 of at least one selected position.
   
   This is exactly the minimum dominating set on the 7-dimensional ternary Hamming graph.
   
   For the branch-and-bound:
   - Use bitsets (Python integers as bitsets) for the 2187 vertices.
   - Precompute the closed neighborhood (self + distance-1 neighbors) for each vertex.
   - Branch: pick an uncovered vertex, branch on which of its neighbors (or itself)
     to add to the dominating set.
   - Bound: use a greedy lower bound on the remaining uncovered vertices.
   
   Let me implement this with careful optimization.
"""
import time
import random
import itertools
import numpy as np
from functools import lru_cache

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s

def construct(q, n, radius):
    """Construct a covering code of radius `radius` for {0..q-1}^n."""
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 145.0

    if not (q == 3 and n == 7 and radius == 1):
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] 
                         for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode 
                    for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    # Total space: 3^7 = 2187 words
    # Each codeword covers 15 words (itself + 14 neighbors at distance 1)
    
    N = 3 ** 7  # 2187
    
    # Map between index and tuple
    def idx_to_tuple(idx):
        t = []
        for _ in range(7):
            t.append(idx % 3)
            idx //= 3
        return t
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(7):
            idx = idx * 3 + t[i]
        return idx
    
    # Precompute all words and their neighbors
    all_words = []
    for idx in range(N):
        all_words.append(idx_to_tuple(idx))
    
    # Precompute closed neighborhoods (self + distance-1 neighbors) as bitsets
    # Using Python integers as bitsets (2187 bits)
    neighborhoods = [0] * N
    for idx in range(N):
        word = all_words[idx]
        mask = 1 << idx  # self
        # Distance-1 neighbors
        for pos in range(7):
            for val in range(3):
                if val == word[pos]:
                    continue
                new_word = word.copy()
                new_word[pos] = val
                nidx = tuple_to_idx(new_word)
                mask |= (1 << nidx)
        neighborhoods[idx] = mask
    
    # Greedy initial solution
    def greedy_cover():
        uncovered = (1 << N) - 1  # all bits set
        selected = []
        while uncovered:
            # Find the vertex that covers the most uncovered vertices
            best_idx = -1
            best_count = 0
            # Sample to save time
            candidates = list(range(N))
            random.shuffle(candidates)
            for idx in candidates[:200]:
                covered = neighborhoods[idx] & uncovered
                count = bin(covered).count('1')
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            uncovered &= ~neighborhoods[best_idx]
        return selected
    
    # Get initial greedy solution
    random.seed(42)
    initial_sol = greedy_cover()
    best_sol = list(initial_sol)
    best_size = len(best_sol)
    
    # Branch and bound for minimum dominating set
    # State: (uncovered_mask, selected_set, depth)
    # We use a priority queue / best-first search with bounds
    
    import heapq
    
    # Lower bound: greedy packing bound
    def lower_bound(uncovered_mask):
        """Lower bound on number of additional codewords needed."""
        if uncovered_mask == 0:
            return 0
        # Each new codeword covers at most 15 vertices
        # But we can do better with a greedy packing
        remaining = uncovered_mask
        count = 0
        while remaining:
            # Pick a vertex with minimum coverage in remaining
            # (this gives a valid lower bound via packing argument)
            # For speed, just use the simple bound
            count += 1
            # Remove a neighborhood
            # Find any vertex in remaining
            v = (remaining & -remaining).bit_length() - 1
            remaining &= ~neighborhoods[v]
        return count
    
    # Better lower bound using LP-style or greedy
    def better_lower_bound(uncovered_mask):
        if uncovered_mask == 0:
            return 0
        # Simple: ceil(popcount / 15)
        return (bin(uncovered_mask).count('1') + 14) // 15
    
    # Branch and bound
    def branch_and_bound(uncovered_mask, selected, current_size):
        nonlocal best_sol, best_size, _START_TIME, _DEADLINE
        
        if time.time() > _DEADLINE - 2:
            return
        
        if current_size >= best_size:
            return
        
        if uncovered_mask == 0:
            if current_size < best_size:
                best_size = current_size
                best_sol = list(selected)
            return
        
        # Lower bound
        lb = better_lower_bound(uncovered_mask)
        if current_size + lb >= best_size:
            return
        
        # Pick a vertex to branch on: choose one with fewest options
        # among uncovered vertices
        # Find an uncovered vertex
        # Use heuristic: pick uncovered vertex with minimum degree in the 
        # "remaining" sense, or just pick the first one
        
        # Find first uncovered vertex
        first_bit = uncovered_mask & -uncovered_mask
        v = first_bit.bit_length() - 1
        
        # Options: add v itself, or add one of its neighbors
        # To reduce branching, consider only neighbors that are also uncovered
        # or that cover many uncovered vertices
        
        # Get candidates: v and its neighbors that help cover uncovered
        candidates = []
        # v itself
        candidates.append(v)
        # Neighbors of v
        nb_mask = neighborhoods[v]
        # Iterate through neighbors
        temp = nb_mask
        while temp:
            bit = temp & -temp
            u = bit.bit_length() - 1
            if u != v:
                candidates.append(u)
            temp ^= bit
        
        # Sort candidates by how many uncovered vertices they cover (descending)
        # This helps find good solutions faster
        def coverage_score(idx):
            return bin(neighborhoods[idx] & uncovered_mask).count('1')
        
        candidates.sort(key=coverage_score, reverse=True)
        
        # Branch on top candidates (limit branching factor for speed)
        max_branch = 8
        for cand in candidates[:max_branch]:
            new_uncovered = uncovered_mask & ~neighborhoods[cand]
            new_selected = selected + [cand]
            branch_and_bound(new_uncovered, new_selected, current_size + 1)
            if time.time() > _DEADLINE - 2:
                return
    
    # Run branch and bound from the initial solution
    # First, compute uncovered from initial solution
    covered_mask = 0
    for idx in initial_sol:
        covered_mask |= neighborhoods[idx]
    uncovered_mask = ((1 << N) - 1) & ~covered_mask
    
    # Start B&B
    branch_and_bound(uncovered_mask, [], 0)
    
    # Convert best solution to list of codewords
    if best_sol:
        result = [all_words[idx] for idx in best_sol]
    else:
        result = [all_words[idx] for idx in initial_sol]
    
    # Verify coverage
    result_set = set(tuple(cw) for cw in result)
    for word in all_words:
        covered = False
        for cw in result:
            if sum(a != b for a, b in zip(word, cw)) <= 1:
                covered = True
                break
        if not covered:
            # Fallback to initial greedy
            result = [all_words[idx] for idx in initial_sol]
            break
    
    return result
