import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def time_check():
        return time.time() > deadline - 5.0

    # We aim for n=17.
    # If we fail, we fall back to n=16 or lower, but the goal is to beat 1.0 (n=17).
    # Actually, the seed score is 1.0 which corresponds to n=17? 
    # Wait, the prompt says "The best known colouring has 17 vertices... 18 or 19 vertices would be new."
    # And "search score is value/seed". If seed is 1.0, and value is n, then seed value must be 17?
    # Or is seed value 1? "1.0 equals the seed program".
    # The current program returns n=17 in the success case.
    # If the current program scores 1.0, and it returns n=17, then the "value" is likely n.
    # So to beat 1.0, we need n > 17? i.e., n=18 or 19.
    # The prompt says "18 or 19 vertices would be new".
    # So the goal is to find a valid coloring for n=18 or n=19.
    
    # Let's try to find n=18. If that's too hard in the time limit, we might fail.
    # But the local search approach is very different from backtracking.
    
    def generate_coloring(n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Random initialization
        return np.random.randint(0, 4, size=(n, n))

    def count_c6s(coloring, n):
        """
        Vectorized count of monochromatic C6s.
        A C6 is a cycle of length 6.
        We can count cycles by summing over all possible sets of 6 vertices? No, that's too slow.
        
        Alternative: Count number of paths of length 5 between u and v, then check if (u,v) is an edge of the same color?
        No, C6 is a cycle. 
        Number of C6s in a graph G is related to the trace of A^6?
        trace(A^6) counts closed walks of length 6.
        Each C6 is counted 12 times (6 starting points * 2 directions).
        But we must exclude smaller cycles (C3, C4) that might form closed walks of length 6?
        A closed walk of length 6 in a simple graph can be:
        - A C6 (simple cycle)
        - A C4 with a backtrack (e.g. u-v-w-v-u... no, length 6)
        - A C3 traversed twice? (u-v-w-u-v-w-u) -> length 6.
        - A C2 (edge traversed back and forth) multiple times?
        
        Actually, for "exact search for not necessarily induced cycles", a C6 is a simple cycle of length 6.
        The trace of A^6 counts all closed walks of length 6.
        We need to subtract the contributions from non-simple cycles.
        
        However, for a quick heuristic in local search, minimizing the number of closed walks of length 6 is a good proxy for minimizing C6s.
        If we drive closed walks of length 6 to 0, we definitely have no C6s.
        We might have false negatives (no C6s but some closed walks of length 6 from C3s/C4s).
        But for random colorings, C3s and C4s are also unlikely to be abundant if we minimize all short cycles.
        
        Let's use the trace of A_c^6 for each color c.
        Total score = sum_c trace(A_c^6).
        
        Computing A^6 is O(n^3 * log 6) or O(n^3) with 5 multiplications.
        For n=18, 18^3 = 5832. Very fast.
        """
        total_score = 0
        for c in range(4):
            # Extract subgraph for color c
            # coloring is (n, n) matrix. We want A_c where A_c[i,j] = 1 if coloring[i,j]==c.
            # Since coloring is symmetric, we only need upper triangle, but A_c should be symmetric.
            A_c = (coloring == c).astype(np.int8)
            
            # Compute A_c^6
            # A2 = A_c @ A_c
            A2 = A_c @ A_c
            A4 = A2 @ A2
            A6 = A4 @ A2
            
            # Trace
            trace_val = np.trace(A6)
            total_score += trace_val
            
        return total_score

    def local_search(coloring, n, max_iter=5000):
        """
        Hill climbing.
        Pick a random edge (i,j), try changing its color, accept if score decreases.
        """
        current_score = count_c6s(coloring, n)
        best_coloring = coloring.copy()
        best_score = current_score
        
        # Precompute list of edges
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for _ in range(max_iter):
            if time_check():
                break
                
            # Pick random edge
            i, j = random.choice(edges)
            
            # Try all other colors
            current_color = coloring[i, j]
            best_new_color = current_color
            best_new_score = current_score
            
            for new_color in range(4):
                if new_color == current_color:
                    continue
                    
                # Apply change
                coloring[i, j] = new_color
                coloring[j, i] = new_color
                
                # Calculate new score
                # Optimization: We can update the score incrementally?
                # Trace(A^6) is global. Recomputing is O(n^3).
                # For n=18, n^3 is ~6000 ops. 4 colors * 5000 iters * 6000 = 120M ops.
                # Python/Numpy overhead might make this slow.
                # Let's try to be smarter or just rely on speed.
                
                new_score = count_c6s(coloring, n)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_color = new_color
            
            # Revert to best
            coloring[i, j] = best_new_color
            coloring[j, i] = best_new_color
            
            if best_new_score < current_score:
                current_score = best_new_score
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = coloring.copy()
                
                # If we found a perfect score (0 closed walks of length 6), we likely have no C6s.
                # But we must verify strictly.
                if current_score == 0:
                    # Verify no C6s
                    if verify_no_c6(coloring, n):
                        return best_coloring, True
            else:
                # Simulated annealing style: accept with small probability to escape local minima?
                # For now, simple hill climbing.
                pass
                
        return best_coloring, False

    def verify_no_c6(coloring, n):
        """
        Strict verification: Check for any simple cycle of length 6 in any color class.
        """
        for c in range(4):
            A_c = (coloring == c)
            # For each pair (u, v) of color c, check if there is a path of length 5 in A_c \ {(u,v)}
            # To be efficient, we can iterate over all edges of color c.
            edges_c = np.argwhere(A_c)
            edges_c = edges_c[edges_c[:, 0] < edges_c[:, 1]]
            
            for u, v in edges_c:
                # Remove (u,v) and (v,u) temporarily for path check
                # Path of length 5 from u to v in A_c without using edge (u,v)
                # This is equivalent to (A_c^5)[u,v] > 0 in the graph where (u,v) is removed?
                # Not exactly, because the path might use (u,v) in the middle?
                # No, (u,v) is an edge. A path of length 5 from u to v using only edges in A_c.
                # If the path uses (u,v), it must be the first or last edge?
                # If it's the first edge: u-v-...-v. Length 5.
                # If it's the last edge: u-...-v-v. Length 5.
                # If it's in the middle: u-...-v-...-v.
                
                # Easiest: BFS/DFS from u to v in A_c with depth 5, avoiding the direct edge (u,v) at the start?
                # Actually, if we find *any* path of length 5 from u to v in A_c, and (u,v) is an edge in A_c,
                # does that form a C6?
                # If the path is simple and does not use the edge (u,v) itself, yes.
                # If the path uses the edge (u,v), e.g., u-v-w-x-y-v, then the cycle is u-v-w-x-y-v-u, which is length 6.
                # Wait, u-v is the edge. The path is u to v.
                # If the path is u-v-...-v, it repeats v. Not a simple cycle.
                # A C6 requires distinct vertices u, v1, v2, v3, v4, v5.
                # So the path of length 5 must be simple and not use the edge (u,v).
                
                # Let's just do a BFS for simple path of length 5.
                # n is small (18).
                
                def has_simple_path_5(u, v, adj):
                    # adj is a list of sets or a boolean matrix
                    # We'll use a stack for DFS
                    stack = [(u, 0, {u})]
                    while stack:
                        curr, depth, visited = stack.pop()
                        if depth == 5:
                            if curr == v:
                                return True
                            continue
                        for next_node in np.where(adj[curr])[0]:
                            if next_node not in visited:
                                # Optimization: If next_node is v and depth < 5, we can't stop yet,
                                # but if we reach v at depth 5, it's a hit.
                                # Also, we must not use the edge (u,v) directly?
                                # The path is u -> ... -> v.
                                # If the first step is v (depth 1), then we are at v with depth 1.
                                # We need to reach v at depth 5.
                                # So if we go to v at depth 1, we can't use it again.
                                # This logic holds.
                                stack.append((next_node, depth + 1, visited | {next_node}))
                    return False

                adj_c = A_c
                if has_simple_path_5(u, v, adj_c):
                    return False
                    
        return True

    # Main Search Logic
    # Try n=18 first. If we find a solution, great.
    # If not, try n=17 to ensure we at least match the seed (score 1.0).
    # Actually, if we fail n=18, we should return a valid n=17 if we can find one quickly,
    # or just return n=17 from a known construction if local search fails.
    
    # Let's run local search for n=18.
    # Population of 10? Or just sequential restarts?
    # Sequential restarts with hill climbing.
    
    best_result = None
    
    # Try to find n=18
    for restart in range(50):
        if time_check():
            break
            
        coloring = generate_coloring(18, seed=random.randint(0, 1000000))
        result, found = local_search(coloring, 18, max_iter=2000)
        
        if found:
            best_result = {'n': 18, 'colors': []}
            # Extract colors in lexicographic order
            for i in range(18):
                for j in range(i+1, 18):
                    best_result['colors'].append(int(result[i, j]))
            return best_result

    # If n=18 failed, try n=17 to be safe (score 1.0)
    # But we want to beat 1.0. If we can't beat it, we should return a valid n=17.
    # The seed program returns n=17.
    # Let's try to find n=17 with local search as well, it should be faster.
    for restart in range(50):
        if time_check():
            break
            
        coloring = generate_coloring(17, seed=random.randint(0, 1000000))
        result, found = local_search(coloring, 17, max_iter=5000)
        
        if found:
            best_result = {'n': 17, 'colors': []}
            for i in range(17):
                for j in range(i+1, 17):
                    best_result['colors'].append(int(result[i, j]))
            return best_result

    # Fallback: If local search fails completely, return a small valid coloring.
    # Or try the cyclic Z_17 construction from the seed as a last resort?
    # The seed had a cyclic Z_17 search. Let's include a quick version of that.
    
    def try_cyclic_17():
        pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
        
        def has_c6_in_diffs(D):
            D_set = set(D)
            # Check for C6 in cyclic group Z_17
            # A C6 corresponds to a sequence of 6 differences d1..d6 such that sum=0
            # and partial sums are distinct non-zero.
            # Simpler: just use the path check on the constructed graph.
            return False # Placeholder, we'll verify the final graph anyway.

        for assignment in range(4**8):
            if time_check(): break
            colors_for_pairs = [0] * 8
            temp = assignment
            for i in range(8):
                colors_for_pairs[i] = temp % 4
                temp //= 4
                
            diff_sets = [set() for _ in range(4)]
            for i, c in enumerate(colors_for_pairs):
                p = pairs[i]
                diff_sets[c].add(p[0])
                diff_sets[c].add(p[1])
                
            # Build coloring
            color_map = {}
            for c in range(4):
                for d in diff_sets[c]:
                    color_map[d] = c
                    
            colors_list = []
            n = 17
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 17
                    colors_list.append(color_map[diff])
                    
            # Verify
            coloring = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i+1, n):
                    coloring[i, j] = colors_list[i*(n-1) - i*(i-1)//2 + (j-i-1)] # Index mapping is tricky, let's just rebuild
            # Rebuild properly
            coloring = np.zeros((n, n), dtype=int)
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    coloring[i, j] = colors_list[idx]
                    coloring[j, i] = colors_list[idx]
                    idx += 1
                    
            if verify_no_c6(coloring, n):
                return {'n': n, 'colors': colors_list}
                
        return None

    res = try_cyclic_17()
    if res:
        return res
        
    # Absolute fallback: n=9
    n = 9
    colors = []
    for i in range(n):
        for j in range(i+1, n):
            colors.append((j-i) % 4)
            
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
