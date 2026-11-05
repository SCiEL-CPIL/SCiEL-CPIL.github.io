# EVOLVE-BLOCK-START

import time
import random
import math
import sys
from itertools import combinations

def _diffs_of_row(row):
    """Return frozenset of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return frozenset(diffs)

def _total_scope(rows):
    """Return the maximum entry across all rows."""
    if not rows:
        return 0
    return max(max(r) for r in rows)

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    all_d = set()
    for r in rows:
        d = _diffs_of_row(r)
        if all_d & d:
            return False
        all_d |= d
    return True

def _greedy_construct(n_rows, k):
    """Pure greedy construction as fallback."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _generate_rows_for_scope(M, k):
    """
    Generate all valid rows of length k+1 starting at 0 with max element <= M.
    A row is valid if its internal differences are distinct.
    """
    rows = []
    # We need to choose k elements from 1..M
    # But we also need to ensure internal differences are unique within the row.
    # For k=5, we choose 5 elements from 1..M.
    # There are C(M, k) such combinations. For M=112, k=5, C(112, 5) ~ 75 million.
    # This is too many to generate all at once and check.
    # However, we can prune: 
    # 1. The first element must be 0.
    # 2. We pick k elements from 1..M.
    # 3. Check if differences are unique.
    
    # Optimization: Instead of generating all, we can just generate candidates 
    # that are "dense" or "sparse"? No, we need to find a clique.
    # Actually, for k=5, the number of valid rows is much smaller than C(M,5)
    # because many combinations have duplicate differences.
    
    # Let's estimate: 
    # Total combos C(112, 5) = 75,342,264.
    # Checking each takes O(k^2) = 25 ops. 75M * 25 = 1.8B ops. Too slow in Python.
    
    # Alternative: Generate rows via backtracking with pruning on differences.
    valid_rows = []
    
    # We can use a recursive generator that prunes early if a difference repeats.
    def backtrack(start, current_row, current_diffs):
        if len(current_row) == k + 1:
            valid_rows.append(current_row[:])
            return
        
        # Pruning: If we can't fill the rest, return
        # Remaining slots: k + 1 - len(current_row)
        # Max value allowed: M
        # Min value allowed: current_row[-1] + 1
        
        for next_val in range(current_row[-1] + 1, M + 1):
            # Check new differences
            new_diffs = set()
            valid = True
            for prev_val in current_row:
                d = next_val - prev_val
                if d in current_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                current_row.append(next_val)
                current_diffs = current_diffs | new_diffs
                backtrack(next_val, current_row, current_diffs)
                current_row.pop()
    
    backtrack(0, [0], set())
    return valid_rows

def _max_clique_size(adj_matrix, n, target_size, deadline):
    """
    Check if there is a clique of size target_size in the graph.
    Returns the clique if found, else None.
    Uses a simple backtracking with greedy coloring bound.
    """
    # adj_matrix[i][j] is True if i and j are compatible (disjoint diffs)
    
    # We want to find a clique of size target_size.
    # We can use a recursive search.
    
    # To speed up, we can use a heuristic: order nodes by degree.
    # But for simplicity and reliability, let's use a standard backtracking
    # with a "coloring" bound if possible, or just simple pruning.
    
    # Given the small target size (7) and potentially large N (thousands),
    # a simple backtracking might be slow if N is large.
    # However, if we iterate M from 105 upwards, N will be small initially.
    
    # Let's use a simple recursive function:
    # expand(C, P) where C is current clique, P is candidates.
    # If len(C) == target_size, return C.
    # If P is empty, return None.
    # Pick a node v from P.
    # If v is connected to all in C:
    #    recurse(C + [v], P ∩ neighbors(v))
    #    remove v from P
    
    # To improve, we can sort P by degree in the subgraph induced by P ∩ N(C).
    
    import sys
    sys.setrecursionlimit(10000)
    
    found_clique = [None]
    
    def expand(clique, candidates):
        if len(clique) == target_size:
            found_clique[0] = clique[:]
            return True
        
        if not candidates:
            return False
            
        # Heuristic: pick the node in candidates with the highest degree within candidates
        # This is O(N^2) per step, which might be expensive if N is large.
        # But for finding a small clique, it helps prune.
        
        # Actually, for N up to a few thousand, and depth 7, 
        # a simple "pick first" or "pick random" might be faster if we rely on the 
        # structure of the graph.
        
        # Let's try a simple backtracking with a time check.
        
        # Sort candidates by degree in the remaining graph?
        # For now, let's just iterate.
        
        i = 0
        while i < len(candidates):
            if time.time() > deadline:
                return False
                
            v = candidates[i]
            
            # Check if v is connected to all in clique
            valid = True
            for u in clique:
                if not adj_matrix[u][v]:
                    valid = False
                    break
            
            if valid:
                # New candidates are those in candidates[i+1:] that are connected to v
                new_candidates = []
                for j in range(i + 1, len(candidates)):
                    w = candidates[j]
                    if adj_matrix[v][w]:
                        new_candidates.append(w)
                
                if expand(clique + [v], new_candidates):
                    return True
            
            i += 1
            
        return False
    
    # Initial candidates are all nodes
    initial_candidates = list(range(n))
    
    # We can prune initial candidates if their degree is too low
    # If a node has degree < target_size - 1, it can't be in a clique of size target_size
    filtered_candidates = []
    for v in initial_candidates:
        deg = sum(1 for u in range(n) if u != v and adj_matrix[v][u])
        if deg >= target_size - 1:
            filtered_candidates.append(v)
            
    if not expand([], filtered_candidates):
        return None
    
    return found_clique[0]

def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    start_time = time.time()
    deadline = start_time + 145
    
    # Start from the theoretical lower bound or a reasonable estimate
    # Counting bound: (n_rows * k(k+1)/2) / (k+1 choose 2) ? No.
    # Total diffs needed: n_rows * k(k+1)/2
    # Max diffs available in scope M: M(M-1)/2
    # M(M-1)/2 >= n_rows * k(k+1)/2
    # M^2 - M >= n_rows * k(k+1)
    # For n=7, k=5: 7 * 5 * 6 = 210.
    # M^2 - M - 210 >= 0. Root of M^2 - M - 210 = 0 is approx (1 + sqrt(1+840))/2 = (1+29)/2 = 15? No.
    # sqrt(841) = 29. (1+29)/2 = 15.
    # Wait, the counting bound in the prompt is 105. Why?
    # Ah, the differences are from 1 to M.
    # Total distinct differences available is M.
    # Each row has k(k+1)/2 differences.
    # Total differences needed: n_rows * k(k+1)/2.
    # So M >= n_rows * k(k+1)/2.
    # For n=7, k=5: 7 * 15 = 105.
    # So the counting bound is indeed 105.
    
    lower_bound = n_rows * k * (k + 1) // 2
    
    # We know best known is 112. Let's try from lower_bound up to 120.
    # If we find a solution at M, we return it.
    
    # To avoid spending too much time on infeasible small M,
    # we can do a quick check or just start a bit higher if we have time.
    # But 105 is the bound, so we must try it.
    
    # Optimization: Generate rows for M, build graph, check clique.
    
    # We will iterate M from lower_bound to 115.
    # If we don't find a solution, we fall back to greedy.
    
    for M in range(lower_bound, 116):
        if time.time() > deadline - 10:
            break
            
        # Generate all valid rows with max element <= M
        rows = _generate_rows_for_scope(M, k)
        
        if len(rows) < n_rows:
            continue
            
        # Build compatibility graph
        n_rows_gen = len(rows)
        
        # Pre-compute differences for each row
        row_diffs = [_diffs_of_row(r) for r in rows]
        
        # Build adjacency matrix
        # adj[i][j] = True if row_diffs[i] and row_diffs[j] are disjoint
        adj = [[False] * n_rows_gen for _ in range(n_rows_gen)]
        
        for i in range(n_rows_gen):
            for j in range(i + 1, n_rows_gen):
                if not (row_diffs[i] & row_diffs[j]):
                    adj[i][j] = True
                    adj[j][i] = True
                    
        # Check for clique of size n_rows
        # Use the max clique finder
        clique = _max_clique_size(adj, n_rows_gen, n_rows, deadline - 5)
        
        if clique is not None:
            # Extract the rows
            result_rows = [rows[i] for i in clique]
            # Validate just in case
            if _is_valid(result_rows):
                return result_rows
                
    # Fallback to greedy if exact search fails or times out
    return _greedy_construct(n_rows, k)

# EVOLVE-BLOCK-END
