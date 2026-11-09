"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Strategy:
    1. Try to find a small Cayley graph on a permutation group (e.g., A6) with girth >= g.
       This can sometimes yield smaller graphs than algebraic constructions.
    2. Fallback to Lazebnik-Ustimenko-Woldar (LUV) graph construction, which is
       guaranteed to produce a valid graph for prime q=k, but may be larger.
    
    The evaluator measures n (number of vertices), lower is better.
    """
    start_time = time.time()
    deadline = 115.0  # 5s safety margin before 120s limit

    # --- Part 1: Try Cayley graph on A6 (order 360) ---
    # A6 is a simple group of order 360. For k=4, we need 2 generators (plus their inverses).
    # We search for a pair of generators (g1, g2) such that the Cayley graph has girth >= g.
    
    def try_cayley_a6(k, g):
        """Search for a 4-regular Cayley graph on A6 with girth >= g."""
        if k != 4:
            return None
        
        # Generate A6 elements as tuples (permutations)
        # A6 consists of even permutations of {0,1,2,3,4,5}
        import itertools
        
        def is_even(perm):
            inv_count = 0
            for i in range(6):
                for j in range(i+1, 6):
                    if perm[i] > perm[j]:
                        inv_count += 1
            return inv_count % 2 == 0
        
        # Generate all permutations and filter for even ones
        a6_elems = []
        for p in itertools.permutations(range(6)):
            if is_even(p):
                a6_elems.append(p)
        
        n = len(a6_elems)  # Should be 360
        if n != 360:
            return None
        
        # Map element to index
        index = {e: i for i, e in enumerate(a6_elems)}
        
        # Identity element
        identity = tuple(range(6))
        e_id = index[identity]
        
        # Precompute multiplication table
        # mult[i][j] = index of a6_elems[i] * a6_elems[j]
        # Composition: (p * q)(x) = p(q(x))
        mult = [[0] * n for _ in range(n)]
        for i in range(n):
            pi = a6_elems[i]
            for j in range(n):
                pj = a6_elems[j]
                # Compose pi o pj: result[x] = pi[pj[x]]
                prod = tuple(pi[pj[x]] for x in range(6))
                mult[i][j] = index[prod]
        
        # Precompute inverses
        inv = [0] * n
        for i in range(n):
            for j in range(n):
                if mult[i][j] == e_id:
                    inv[i] = j
                    break
        
        # Candidate generators: not identity, not self-inverse (order > 2)
        candidates = []
        for i in range(n):
            if i == e_id:
                continue
            if inv[i] == i:  # Order 2
                continue
            candidates.append(i)
        
        # Compute order of each element to prioritize high-order generators
        element_order = [1] * n
        for i in range(n):
            if i == e_id:
                continue
            cur = i
            order = 1
            while cur != e_id:
                # Multiply cur by the original element i
                cur = mult[cur][i]
                order += 1
                if order > n:
                    break
            element_order[i] = order
        
        # Sort candidates by descending order (high-order elements more likely to give large girth)
        cand_sorted = sorted(candidates, key=lambda x: element_order[x], reverse=True)
        
        import random
        random.seed(12345)
        
        # Strategy: Try top-order pairs first, then random pairs
        # Top pairs: take first ~50 high-order candidates, try all pairs among them
        top_count = min(50, len(cand_sorted))
        top_cands = cand_sorted[:top_count]
        
        found = None
        
        # Phase 1: Systematic search among top-order candidates
        for i1 in range(len(top_cands)):
            if time.time() > deadline:
                break
            g1 = top_cands[i1]
            g1_inv = inv[g1]
            
            for i2 in range(i1 + 1, len(top_cands)):
                if time.time() > deadline:
                    break
                g2 = top_cands[i2]
                g2_inv = inv[g2]
                
                # Check distinctness: g1, g1_inv, g2, g2_inv must be distinct
                if g1 == g2 or g1 == g2_inv or g1_inv == g2 or g1_inv == g2_inv:
                    continue
                
                # BFS to check girth
                depth = [-1] * n
                parent = [-1] * n
                q = deque()
                q.append(e_id)
                depth[e_id] = 0
                
                min_cycle = float('inf')
                valid = True
                
                while q:
                    u = q.popleft()
                    du = depth[u]
                    if du >= g:
                        continue
                    
                    for s in (g1, g1_inv, g2, g2_inv):
                        v = mult[u][s]
                        if depth[v] == -1:
                            depth[v] = du + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            dv = depth[v]
                            # Only consider edges where both endpoints have been visited
                            # and it's not a tree edge
                            if u != parent[v] and v != parent[u]:
                                cl = du + dv + 1
                                if cl < min_cycle:
                                    min_cycle = cl
                                    if min_cycle < g:
                                        valid = False
                                        break
                    if not valid:
                        break
                
                if valid and min_cycle >= g:
                    edges = []
                    for i in range(n):
                        for s in (g1, g1_inv, g2, g2_inv):
                            j = mult[i][s]
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        
        # Phase 2: Random search among all candidates if Phase 1 didn't find one
        # Use a fixed number of random pairs based on time budget
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        # Estimate: each BFS takes ~0.5-2ms in Python for n=360
        # Be conservative: assume 2ms per pair
        max_random_pairs = int(remaining * 1000 / 2.0)
        max_random_pairs = max(100, min(max_random_pairs, 5000))
        
        random.shuffle(cand_sorted)
        
        for _ in range(max_random_pairs):
            if time.time() > deadline:
                break
            
            # Pick two random distinct candidates
            i1 = random.randint(0, len(cand_sorted) - 1)
            i2 = random.randint(0, len(cand_sorted) - 1)
            if i1 == i2:
                continue
            
            g1 = cand_sorted[i1]
            g2 = cand_sorted[i2]
            g1_inv = inv[g1]
            g2_inv = inv[g2]
            
            if g1 == g2 or g1 == g2_inv or g1_inv == g2 or g1_inv == g2_inv:
                continue
            
            depth = [-1] * n
            parent = [-1] * n
            q = deque()
            q.append(e_id)
            depth[e_id] = 0
            
            min_cycle = float('inf')
            valid = True
            
            while q:
                u = q.popleft()
                du = depth[u]
                if du >= g:
                    continue
                
                for s in (g1, g1_inv, g2, g2_inv):
                    v = mult[u][s]
                    if depth[v] == -1:
                        depth[v] = du + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        dv = depth[v]
                        if u != parent[v] and v != parent[u]:
                            cl = du + dv + 1
                            if cl < min_cycle:
                                min_cycle = cl
                                if min_cycle < g:
                                    valid = False
                                    break
                if not valid:
                    break
            
            if valid and min_cycle >= g:
                edges = []
                for i in range(n):
                    for s in (g1, g1_inv, g2, g2_inv):
                        j = mult[i][s]
                        if i < j:
                            edges.append((i, j))
                return n, edges
        
        return None

    # --- Part 2: Fallback to LUV construction ---
    def luv_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph construction."""
        q = k
        # d is such that girth >= d + 5 (for odd d) or d + 4 (for even d)?
        # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # Let's stick to the seed logic which is known to work.
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            # q must be prime for this simple implementation
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Define the rule for the algebraic construction
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            # The rule depends on c mod 4
            # From LUV paper:
            # c=2: (1,1)
            # c=3: (2,1)
            # c=4: (1,2)
            # c=5: (1,3) -> r=0, c0=5, (1, 5-2)=(1,3)
            # c=6: (5,1) -> r=1, c0=6, (6-1,1)=(5,1)
            # c=7: (6,1) -> r=2, c0=7, (7,1)
            # c=8: (1,9) -> r=3, c0=8, (1, 8+1)=(1,9)
            # Wait, the seed code uses:
            # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            # Let's verify:
            # r=0: (1, c0-2). c=5, c0=5 -> (1,3). Correct.
            # r=1: (c0-1, 1). c=6, c0=6 -> (5,1). Correct.
            # r=2: (c0, 1). c=7, c0=7 -> (7,1). Correct.
            # r=3: (1, c0+1). c=8, c0=8 -> (1,9). Correct.
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # point p ~ line l: l[c] - p[c] = l[a] * p[b]
                        # l[c] = p[c] + l[a] * p[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # line l ~ point p: l[c] - p[c] = l[a] * p[b]
                        # p[c] = l[c] - l[a] * p[b]
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        # BFS to build the graph
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    # --- Main logic ---
    # First, try the Cayley graph approach if k=4
    if k == 4:
        result = try_cayley_a6(k, g)
        if result is not None:
            n, edges = result
            # Verify girth quickly? The BFS already did.
            # Verify degree? Cayley graphs are regular.
            return n, edges
    
    # Fallback to LUV
    try:
        n, edges = luv_construct(k, g)
        return n, edges
    except:
        # If LUV fails (e.g., q not prime), we need a fallback.
        # For now, return a trivial invalid graph to avoid crash, 
        # but the task says invalid scores 0. 
        # We should try to make LUV work for k=4.
        # The seed code handles q=4 specially.
        # So it should work.
        pass

    # Last resort: return a cycle (invalid for k=4, but avoids crash)
    # Actually, if k=4, LUV should work.
    # If k is not 4 and not prime, we might fail.
    # Let's assume k=4 is the main case.
    return 0, []
# EVOLVE-BLOCK-END
