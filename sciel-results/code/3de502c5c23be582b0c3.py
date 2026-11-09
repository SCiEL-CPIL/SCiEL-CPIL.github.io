<<<<<<< SEARCH
import time
import sys
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def _count_c6(adj_matrix, n):
        """Count number of 6-cycles in the graph represented by adj_matrix.
        A 6-cycle is a cycle of length 6 (not necessarily induced).
        We count each cycle 12 times (6 starting positions * 2 directions),
        so divide by 12 at the end.
        """
        count = 0
        # Enumerate all 6-tuples of distinct vertices
        # For efficiency, use nested loops with pruning
        for v0 in range(n):
            for v1 in range(n):
                if v1 == v0 or not adj_matrix[v0, v1]:
                    continue
                for v2 in range(n):
                    if v2 == v0 or v2 == v1 or not adj_matrix[v1, v2]:
                        continue
                    for v3 in range(n):
                        if v3 == v0 or v3 == v1 or v3 == v2 or not adj_matrix[v2, v3]:
                            continue
                        for v4 in range(n):
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3 or not adj_matrix[v3, v4]:
                                continue
                            for v5 in range(n):
                                if v5 == v0 or v5 == v1 or v5 == v2 or v5 == v3 or v5 == v4:
                                    continue
                                if adj_matrix[v4, v5] and adj_matrix[v5, v0]:
                                    # Check it's a simple cycle (all distinct)
                                    # Already ensured distinctness above
                                    count += 1
        # Each cycle counted 12 times (6 rotations * 2 directions)
        return count // 12

    def _count_c6_fast(adj_matrix, n):
        """Faster C6 counting using matrix operations where possible.
        For a 6-cycle v0-v1-v2-v3-v4-v5-v0, we can count by:
        For each pair (v0, v3) opposite in the cycle, count paths of length 3.
        But this is complex. Let's use a different approach:
        Count all 6-cycles by iterating over triples of consecutive edges.
        
        Alternative: For each 6-tuple (a,b,c,d,e,f) with all distinct and
        edges (a,b),(b,c),(c,d),(d,e),(e,f),(f,a) present.
        To reduce counting, fix a < b and consider all orderings.
        
        Let's use the standard approach: iterate over all 6-combinations and
        check if they form a cycle in some order.
        """
        count = 0
        # For each set of 6 vertices, check if they contain a 6-cycle
        # This is expensive but let's optimize with bitmasks
        
        # Better approach: enumerate paths of length 5 and check if endpoints are adjacent
        # Path: a-b-c-d-e-f, check if a-f edge exists and all distinct
        for a in range(n):
            for b in range(n):
                if b == a or not adj_matrix[a, b]:
                    continue
                for c in range(n):
                    if c == a or c == b or not adj_matrix[b, c]:
                        continue
                    for d in range(n):
                        if d == a or d == b or d == c or not adj_matrix[c, d]:
                            continue
                        for e in range(n):
                            if e == a or e == b or e == c or e == d or not adj_matrix[d, e]:
                                continue
                            for f in range(n):
                                if f == a or f == b or f == c or f == d or f == e:
                                    continue
                                if adj_matrix[e, f] and adj_matrix[f, a]:
                                    count += 1
        # Each cycle counted 12 times
        return count // 12

    def _has_c6(adj_matrix, n):
        """Check if there's any 6-cycle in the graph."""
        # Early exit version
        for a in range(n):
            for b in range(n):
                if b == a or not adj_matrix[a, b]:
                    continue
                for c in range(n):
                    if c == a or c == b or not adj_matrix[b, c]:
                        continue
                    for d in range(n):
                        if d == a or d == b or d == c or not adj_matrix[c, d]:
                            continue
                        for e in range(n):
                            if e == a or e == b or e == c or e == d or not adj_matrix[d, e]:
                                continue
                            for f in range(n):
                                if f == a or f == b or f == c or f == d or f == e:
                                    continue
                                if adj_matrix[e, f] and adj_matrix[f, a]:
                                    return True
        return False

    def _local_search(n, max_iters=500000):
        """Randomized local search to find a valid 4-coloring of K_n with no C6 in any color class."""
        np.random.seed(42)
        
        # Initialize with random coloring
        colors = np.random.randint(0, 4, size=(n, n))
        # Make symmetric
        colors = (colors + colors.T) // 2
        np.fill_diagonal(colors, -1)
        
        # Build adjacency matrices for each color
        adj = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        
        def _rebuild_adj():
            for c in range(4):
                adj[c][:] = False
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[i, j]
                    adj[c][i, j] = True
                    adj[c][j, i] = True
        
        _rebuild_adj()
        
        def _total_violations():
            total = 0
            for c in range(4):
                total += _has_c6(adj[c], n)
            return total
        
        def _count_violations_per_color():
            violations = [0] * 4
            for c in range(4):
                violations[c] = _has_c6(adj[c], n)
            return violations
        
        best_n = n
        best_colors = colors.copy()
        best_violations = _total_violations()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_color = colors[i, j]
            
            # Try changing to a different color
            new_color = np.random.choice([c for c in range(4) if c != old_color])
            
            # Update adjacency
            adj[old_color][i, j] = False
            adj[old_color][j, i] = False
            adj[new_color][i, j] = True
            adj[new_color][j, i] = True
            colors[i, j] = new_color
            colors[j, i] = new_color
            
            new_violations = _total_violations()
            
            if new_violations <= best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
                if new_violations == 0:
                    break
            else:
                # Revert
                adj[new_color][i, j] = False
                adj[new_color][j, i] = False
                adj[old_color][i, j] = True
                adj[old_color][j, i] = True
                colors[i, j] = old_color
                colors[j, i] = old_color
        
        return best_colors, best_violations

    def _flatten_colors(colors_matrix):
        """Convert n x n matrix to list of colors in lexicographic order."""
        n = colors_matrix.shape[0]
        result = []
        for i in range(n):
            for j in range(i+1, n):
                result.append(int(colors_matrix[i, j]))
        return result

    best_n = 1
    best_colors_flat = []
    
    # Try from larger n down
    for n_target in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        # Try local search
        colors_matrix, violations = _local_search(n_target, max_iters=200000)
        
        if violations == 0:
            best_n = n_target
            best_colors_flat = _flatten_colors(colors_matrix)
            break
        else:
            # Keep as fallback if we haven't found anything better
            if n_target > best_n:
                # Only accept if violations are low (heuristic)
                if violations <= 2:
                    best_n = n_target
                    best_colors_flat = _flatten_colors(colors_matrix)
    
    if best_n == 1:
        return {'n': 1, 'colors': []}
    
    return {'n': best_n, 'colors': best_colors_flat}
=======
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def _has_c6_bitmask(adj_masks, n):
        """Check if any 6-cycle exists using bitmasks.
        adj_masks[v] is a bitmask of neighbors of v.
        A 6-cycle: v0-v1-v2-v3-v4-v5-v0, all distinct.
        """
        for v0 in range(n):
            m0 = adj_masks[v0]
            if m0 == 0:
                continue
            for v1 in range(n):
                if v1 == v0 or not (m0 >> v1) & 1:
                    continue
                m1 = adj_masks[v1] & ~(1 << v0)
                if m1 == 0:
                    continue
                for v2 in range(n):
                    if v2 == v0 or v2 == v1 or not (m1 >> v2) & 1:
                        continue
                    m2 = adj_masks[v2] & ~(1 << v0) & ~(1 << v1)
                    if m2 == 0:
                        continue
                    for v3 in range(n):
                        if v3 == v0 or v3 == v1 or v3 == v2 or not (m2 >> v3) & 1:
                            continue
                        m3 = adj_masks[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        if m3 == 0:
                            continue
                        for v4 in range(n):
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3 or not (m3 >> v4) & 1:
                                continue
                            m4 = adj_masks[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            if m4 == 0:
                                continue
                            # v5 must be neighbor of v4 and v0, distinct from v0-v4
                            candidates = m4 & adj_masks[v0]
                            candidates &= ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            if candidates != 0:
                                return True
        return False

    def _count_c6_bitmask(adj_masks, n):
        """Count 6-cycles using bitmasks."""
        count = 0
        for v0 in range(n):
            m0 = adj_masks[v0]
            if m0 == 0:
                continue
            for v1 in range(n):
                if v1 == v0 or not (m0 >> v1) & 1:
                    continue
                m1 = adj_masks[v1] & ~(1 << v0)
                if m1 == 0:
                    continue
                for v2 in range(n):
                    if v2 == v0 or v2 == v1 or not (m1 >> v2) & 1:
                        continue
                    m2 = adj_masks[v2] & ~(1 << v0) & ~(1 << v1)
                    if m2 == 0:
                        continue
                    for v3 in range(n):
                        if v3 == v0 or v3 == v1 or v3 == v2 or not (m2 >> v3) & 1:
                            continue
                        m3 = adj_masks[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        if m3 == 0:
                            continue
                        for v4 in range(n):
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v3 or not (m3 >> v4) & 1:
                                continue
                            m4 = adj_masks[v4] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            if m4 == 0:
                                continue
                            candidates = m4 & adj_masks[v0]
                            candidates &= ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            count += bin(candidates).count('1')
        # Each cycle counted 12 times
        return count // 12

    def _local_search(n, max_iters=100000):
        """Randomized local search to find a valid 4-coloring of K_n with no C6 in any color class."""
        rng = np.random.RandomState(42)
        
        # Initialize with random coloring
        colors = rng.randint(0, 4, size=(n, n))
        # Make symmetric
        for i in range(n):
            for j in range(i+1, n):
                colors[j, i] = colors[i, j]
        np.fill_diagonal(colors, -1)
        
        # Build adjacency bitmasks for each color
        adj_masks = [[0] * n for _ in range(4)]
        
        def _rebuild_adj():
            for c in range(4):
                for v in range(n):
                    adj_masks[c][v] = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[i, j]
                    adj_masks[c][i] |= (1 << j)
                    adj_masks[c][j] |= (1 << i)
        
        _rebuild_adj()
        
        def _total_violations():
            total = 0
            for c in range(4):
                if _has_c6_bitmask(adj_masks[c], n):
                    total += 1
            return total
        
        best_colors = colors.copy()
        best_violations = _total_violations()
        
        if best_violations == 0:
            return best_colors, 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Pick a random edge
            i = rng.randint(0, n)
            j = rng.randint(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_color = colors[i, j]
            
            # Try changing to a different color
            new_color = rng.choice([c for c in range(4) if c != old_color])
            
            # Update adjacency
            adj_masks[old_color][i] &= ~(1 << j)
            adj_masks[old_color][j] &= ~(1 << i)
            adj_masks[new_color][i] |= (1 << j)
            adj_masks[new_color][j] |= (1 << i)
            colors[i, j] = new_color
            colors[j, i] = new_color
            
            new_violations = _total_violations()
            
            if new_violations < best_violations:
                best_violations = new_violations
                best_colors = colors.copy()
                if new_violations == 0:
                    break
            elif new_violations == best_violations:
                # Accept with some probability for diversity
                if rng.random() < 0.3:
                    best_colors = colors.copy()
            else:
                # Revert
                adj_masks[new_color][i] &= ~(1 << j)
                adj_masks[new_color][j] &= ~(1 << i)
                adj_masks[old_color][i] |= (1 << j)
                adj_masks[old_color][j] |= (1 << i)
                colors[i, j] = old_color
                colors[j, i] = old_color
        
        return best_colors, best_violations

    def _flatten_colors(colors_matrix):
        """Convert n x n matrix to list of colors in lexicographic order."""
        n = colors_matrix.shape[0]
        result = []
        for i in range(n):
            for j in range(i+1, n):
                result.append(int(colors_matrix[i, j]))
        return result

    best_n = 1
    best_colors_flat = []
    
    # Try from larger n down
    for n_target in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        # Try local search with multiple restarts
        found = False
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            colors_matrix, violations = _local_search(n_target, max_iters=50000)
            
            if violations == 0:
                best_n = n_target
                best_colors_flat = _flatten_colors(colors_matrix)
                found = True
                break
        
        if found:
            break
    
    if best_n == 1:
        return {'n': 1, 'colors': []}
    
    return {'n': best_n, 'colors': best_colors_flat}
>>>>>>> REPLACE
