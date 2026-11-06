import time
import random
import itertools
import numpy as np

def _build_bitsets(colors, n):
    """Build adjacency bitsets from colors list. Returns list of Python ints."""
    bitsets = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx]:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
            idx += 1
    return bitsets

def _count_k222_bitsets(bitsets, n):
    """Count K2,2,2 subgraphs using bitset operations. Fast O(n^4/word) per call."""
    count = 0
    for c in range(n - 1):
        bc = bitsets[c]
        for d in range(c + 1, n):
            if not (bc >> d) & 1:
                continue
            # Common neighbors of c and d (excluding c,d themselves via bitmasks)
            common = bc & bitsets[d]
            # Count K2,2 in subgraph induced by 'common'
            # For each pair (a,b) in common, count common neighbors of a and b within common
            common_bits = common
            verts = []
            while common_bits:
                lsb = common_bits & (-common_bits)
                v = lsb.bit_length() - 1
                verts.append(v)
                common_bits ^= lsb
            m = len(verts)
            if m < 4:
                continue
            for ii in range(m - 1):
                a = verts[ii]
                ba = bitsets[a]
                for jj in range(ii + 1, m):
                    b = verts[jj]
                    # Common neighbors of a and b that are also in common
                    nb = ba & bitsets[b] & common
                    # Count bits in nb
                    cnt = nb.bit_count()
                    if cnt >= 2:
                        count += 1
    return count

def _delta_k222_flip(bitsets, n, i, j):
    """Compute change in K2,2,2 count when flipping edge (i,j).
    Returns delta = new_count - old_count.
    
    Key insight: A K2,2,2 with parts {a,b},{c,d},{e,f} contains edge (i,j) iff
    two of {a,b,c,d,e,f} are i and j. We count contributions involving edge (i,j).
    """
    # Edge (i,j) can appear in K2,2,2 in these ways:
    # Case 1: i,j are in the same part (say {i,j})
    #   Need 2 more parts of 2: pick (a,b) and (c,d) such that all 4 are 
    #   common neighbors of both i and j, AND form K2,2 (i.e., a-b, a-c, a-d, b-c, b-d all same color as i-j)
    #   Wait, K2,2,2 means the 3 pairs are the parts. Edges BETWEEN parts must be monochromatic.
    #   If i,j are in part 1, then all edges from {i,j} to {a,b} must be color c, and all edges from {i,j} to {c,d} must be color c, and all edges from {a,b} to {c,d} must be color c.
    #   So: i-a, i-b, j-a, j-b, i-c, i-d, j-c, j-d all color c, AND a-b, a-c, a-d, b-c, b-d all color c.
    #   But a-b is within part {a,b}, so a-b is NOT an edge of K2,2,2! 
    #   Actually in K2,2,2 (complete tripartite), there are NO edges within parts.
    #   So the 6 vertices form 3 parts of size 2, and all 12 edges between parts exist.
    #   The color class contains K2,2,2 if all 12 cross-part edges have that color.
    
    # So for edge (i,j) with color col:
    # Case 1: i,j in same part. Need two more disjoint pairs {a,b},{c,d} all common neighbors of i and j,
    #          with all edges between {i,j}x{a,b}, {i,j}x{c,d}, {a,b}x{c,d} having color col.
    #          But "all edges between" means: for every x in {i,j}, y in {a,b}: edge(x,y) has color col.
    #          Since i,j are both in part 1, and a,b in part 2, ALL of i-a,i-b,j-a,j-b must be color col.
    #          Similarly for c,d. And a-c,a-d,b-c,b-d must be color col.
    #          Note: a-b is NOT required (it's within a part).
    # Case 2: i in part 1, j in part 2. Need part 3 = {a,b} (2 vertices), and parts 1,2 each have one more vertex.
    #          Part 1 = {i, x}, Part 2 = {j, y}, Part 3 = {a, b}.
    #          All cross edges: i-j (given), i-y, x-j, x-y (between part1 and part2),
    #          i-a, i-b, x-a, x-b (between part1 and part3),
    #          j-a, j-b, y-a, y-b (between part2 and part3).
    #          All must be color col.
    #          Note: x is the other vertex in part1 (so x != i), y is other in part2 (y != j), a,b in part3.
    #          Also x != j (different parts), x != a,b, y != i, y != a,b, a != b.
    
    # This is complex. Let me use a different approach: 
    # Count K2,2,2s containing edge (i,j) by enumerating the 6-vertex set.
    # Actually, let's just do the full recount but optimized.
    # For now, return 0 to indicate "need full recount" - we'll use a hybrid approach.
    return None  # Signal that we should do full recount

def _is_valid_bitsets(bitsets, n):
    return _count_k222_bitsets(bitsets, n) == 0

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def _delta_k222_flip_fast(bitsets, n, i, j):
    """Compute delta in K2,2,2 count when flipping edge (i,j).
    
    A monochromatic K2,2,2 with edge (i,j) of color col must have all 12 cross-part edges of color col.
    We count how many K2,2,2 subgraphs contain edge (i,j) and are monochromatic in the current coloring.
    Flipping (i,j) removes those (if they were of the old color) and adds new ones (of the new color).
    
    delta = (new K2,2,2 count) - (old K2,2,2 count)
         = (K222s of new color containing (i,j)) - (K222s of old color containing (i,j))
    """
    old_col = (bitsets[i] >> j) & 1
    new_col = 1 - old_col
    
    def count_k222_with_edge(i, j, col):
        """Count monochromatic K2,2,2 subgraphs containing edge (i,j) where all edges have color col."""
        # First, i and j must both be connected to each other with color col (they are, by definition of the edge)
        # Common neighbors of i and j with color col:
        # Vertices v such that edge(i,v) has color col AND edge(j,v) has color col
        if old_col == col:
            # i-j already has color col
            pass
        else:
            # i-j does NOT have color col, so no K2,2,2 of color col can contain edge (i,j)
            # Wait, but we're counting K2,2,2s that CONTAIN edge (i,j) as one of the 12 cross-part edges.
            # For a K2,2,2 of color col to contain edge (i,j), edge (i,j) must have color col.
            # If edge (i,j) currently has color != col, then there are 0 such K2,2,2s.
            return 0
        
        # So we need edge(i,j) to have color col. In the "new" scenario, it will.
        # Let's count assuming edge(i,j) has color col.
        # Common neighbors v with edge(i,v)=col and edge(j,v)=col:
        common = bitsets[i] & bitsets[j]
        # Remove i and j themselves
        common &= ~(1 << i)
        common &= ~(1 << j)
        
        count = 0
        # Case 1: i and j are in the same part {i,j}
        # Need two more disjoint pairs {a,b} and {c,d} from common, such that:
        #   All edges from {i,j} to {a,b} are col: i-a, i-b, j-a, j-b are col. (Already true since a,b in common)
        #   All edges from {i,j} to {c,d} are col: i-c, i-d, j-c, j-d are col. (Already true since c,d in common)
        #   All edges from {a,b} to {c,d} are col: a-c, a-d, b-c, b-d are col.
        # So we need: for disjoint pairs {a,b},{c,d} from common, all 4 cross edges between them are col.
        # This is equivalent to: in the subgraph induced by common (with edges of color col), 
        #   the number of ways to pick 4 vertices forming a K2,2 (i.e., a 4-cycle with all 4 cross edges present).
        # Actually, K2,2 on {a,b,c,d} means a-c, a-d, b-c, b-d all present. That's exactly K2,2.
        
        # Enumerate 4-subsets of common that form K2,2 in the color-col subgraph
        verts = []
        bits = common
        while bits:
            lsb = bits & (-bits)
            v = lsb.bit_length() - 1
            verts.append(v)
            bits ^= lsb
        
        m = len(verts)
        if m < 4:
            return 0
        
        # Count K2,2 subgraphs in the color-col induced subgraph on 'verts'
        for ii in range(m - 3):
            a = verts[ii]
            ba = bitsets[a]
            for jj in range(ii + 1, m - 2):
                b = verts[jj]
                # a and b must be in different parts of the K2,2, so edge a-b should NOT be in the K2,2
                # But wait, in K2,2,2, a and b are in the SAME part, so edge a-b is NOT part of K2,2,2.
                # The K2,2 here is on the bipartite graph between {a,b} and {c,d}.
                # So we need: a-c, a-d, b-c, b-d all color col.
                # a and b don't need to be connected.
                ab_common = ba & bitsets[b] & common
                # ab_common contains c,d such that a-c, a-d, b-c, b-d are all color col
                # But we need to pick 2 from ab_common that are NOT a or b (already excluded by common)
                cnt = ab_common.bit_count()
                if cnt >= 2:
                    # Choose 2 from cnt: C(cnt, 2)
                    count += cnt * (cnt - 1) // 2
        
        # Case 2: i in part1, j in part2
        # Part1 = {i, x}, Part2 = {j, y}, Part3 = {a, b}
        # All cross edges must be color col:
        #   i-j: col (given)
        #   i-y: col, x-j: col, x-y: col (between part1 and part2)
        #   i-a, i-b, x-a, x-b: col (between part1 and part3)
        #   j-a, j-b, y-a, y-b: col (between part2 and part3)
        # So: x must be a common neighbor of i and j with color col (x in common)
        #     y must be a common neighbor of i and j with color col (y in common)
        #     a,b must be common neighbors of i, j, x, y with color col
        #     Also x != y, x != a,b, y != a,b, a != b
        #     And edge x-y must be color col
        
        # x and y are both in common (neighbors of i and j with color col)
        # Additionally, x-y must be color col
        # a,b must be neighbors of i, j, x, y all with color col
        
        for ii in range(m):
            x = verts[ii]
            bx = bitsets[x]
            # y must be in common, y != x, and x-y is color col
            y_candidates = common & bx & ~(1 << x)
            ybits = y_candidates
            while ybits:
                lsb_y = ybits & (-ybits)
                y = lsb_y.bit_length() - 1
                # a,b must be common neighbors of i, j, x, y with color col
                # i,j already: a,b in common
                # x: a,b in bx
                # y: a,b in bitsets[y]
                abc = common & bx & bitsets[y]
                abc &= ~(1 << i) & ~(1 << j) & ~(1 << x) & ~(1 << y)
                cnt = abc.bit_count()
                if cnt >= 2:
                    count += cnt * (cnt - 1) // 2
                ybits ^= lsb_y
        
        return count
    
    # Count K2,2,2s of old color containing (i,j) in current coloring
    # These will be REMOVED when we flip (i,j) to new color
    removed = count_k222_with_edge(i, j, old_col)
    
    # Count K2,2,2s of new color containing (i,j) after flip
    # We need to temporarily consider (i,j) as new color
    # For count_k222_with_edge with col=new_col, we need edge(i,j) to be new_col
    # Let's compute it by temporarily flipping
    bitsets[i] ^= (1 << j)
    bitsets[j] ^= (1 << i)
    added = count_k222_with_edge(i, j, new_col)
    bitsets[i] ^= (1 << j)
    bitsets[j] ^= (1 << i)
    
    return added - removed

def _idx_to_ij(idx, n):
    """Convert linear edge index to (i, j) pair in O(1) using quadratic formula."""
    # idx = i*(2n-2-i)/2 + (j-i-1)
    # Solve: i*(2n-2-i) = 2*(idx - (j-i-1)) <= 2*idx
    # i^2 - (2n-2)*i + 2*idx >= 0 (approximately)
    # i = (2n-2 - sqrt((2n-2)^2 - 8*idx)) / 2
    a = 2 * n - 2
    disc = a * a - 8 * idx
    if disc < 0:
        disc = 0
    i = (a - int(disc ** 0.5)) // 2
    # Clamp i to valid range
    if i < 0:
        i = 0
    if i >= n - 1:
        i = n - 2
    # Compute base for this i
    base = i * (2 * n - 2 - i) // 2
    offset = idx - base
    j = i + 1 + offset
    # Verify and adjust if needed
    if j >= n:
        j = n - 1
    if j <= i:
        j = i + 1
    return i, j

def _local_search_bitsets(colors, n, max_iters=10000, time_limit=30):
    """Steepest-descent local search using incremental K2,2,2 delta."""
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_bitsets(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try all single-edge flips, pick best improvement using delta
        best_idx = -1
        best_delta = 0
        
        # Sample a subset of edges for speed (if n is large)
        if num_edges <= 400:
            candidates = range(num_edges)
        else:
            candidates = random.sample(range(num_edges), min(300, num_edges))
        
        for idx in candidates:
            i, j = _idx_to_ij(idx, n)
            
            delta = _delta_k222_flip_fast(bitsets, n, i, j)
            
            if delta < best_delta:
                best_delta = delta
                best_idx = idx
        
        if best_idx >= 0:
            # Apply best flip
            i, j = _idx_to_ij(best_idx, n)
            
            old_val = (bitsets[i] >> j) & 1
            if old_val:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
                colors[best_idx] = 0
            else:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
                colors[best_idx] = 1
            
            current_cost += best_delta
            if current_cost <= 0:
                # Verify and return
                if _count_k222_bitsets(bitsets, n) == 0:
                    return colors, True
                else:
                    current_cost = _count_k222_bitsets(bitsets, n)
        else:
            # No improving move found
            break
    
    return colors, current_cost == 0

def _simulated_annealing_bitsets(colors, n, max_iters=50000, time_limit=80):
    """Simulated annealing for K2,2,2-free coloring."""
    start_time = time.time()
    bitsets = _build_bitsets(colors, n)
    current_cost = _count_k222_bitsets(bitsets, n)
    
    if current_cost == 0:
        return colors, True
    
    num_edges = n * (n - 1) // 2
    T_start = max(1.0, current_cost * 0.5)
    T_end = 0.01
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    T = T_start
    
    best_cost = current_cost
    best_colors = colors[:]
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        i, j = _idx_to_ij(idx, n)
        
        delta = _delta_k222_flip_fast(bitsets, n, i, j)
        new_cost = current_cost + delta
        
        # Accept if better or with probability exp(-delta/T)
        if delta <= 0 or (T > T_end and random.random() < np.exp(-delta / T)):
            # Apply flip
            old_val = (bitsets[i] >> j) & 1
            if old_val:
                bitsets[i] ^= (1 << j)
                bitsets[j] ^= (1 << i)
                colors[idx] = 0
            else:
                bitsets[i] |= (1 << j)
                bitsets[j] |= (1 << i)
                colors[idx] = 1
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return colors, True
        else:
            # Revert (don't change)
            pass
        
        T *= cooling
    
    if best_cost == 0:
        colors[:] = best_colors
        return colors, True
    return colors, False

def construct():
    deadline = time.time() + 140
    target_n = 29
    p = 29  # 29 is prime
    
    # 1. Try Paley(29) directly
    colors = _paley(target_n, p)
    if colors:
        bitsets = _build_bitsets(colors, target_n)
        if _is_valid_bitsets(bitsets, target_n):
            return {'n': target_n, 'colors': colors}
    
    # 2. Local search from Paley (with incremental delta - much faster)
    colors_copy = colors[:]
    _, success = _local_search_bitsets(colors_copy, target_n, max_iters=20000, time_limit=45)
    if success:
        return {'n': target_n, 'colors': colors_copy}
    
    # 3. Simulated annealing from Paley
    if time.time() < deadline - 35:
        colors_sa = colors[:]
        _, success = _simulated_annealing_bitsets(colors_sa, target_n, max_iters=50000, time_limit=70)
        if success:
            return {'n': target_n, 'colors': colors_sa}
    
    # 4. Random restarts with local search
    for attempt in range(30):
        if time.time() > deadline - 12:
            break
        colors_rand = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _local_search_bitsets(colors_rand, target_n, max_iters=3000, time_limit=8)
        if success:
            return {'n': target_n, 'colors': colors_rand}
    
    # 5. Fallback: try smaller n with Paley
    for n in range(28, 10, -1):
        if time.time() > deadline - 5:
            break
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                colors = _paley(n, pp)
                if colors:
                    bitsets = _build_bitsets(colors, n)
                    if _is_valid_bitsets(bitsets, n):
                        return {'n': n, 'colors': colors}
    
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
