import time
import itertools
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # 1. Try Cayley graph on SL(2, 16)
    result = try_cayley_sl2(16, k, g, deadline)
    if result is not None:
        return result
    
    # 2. Try Cayley graph on SL(2, 32)
    result = try_cayley_sl2(32, k, g, deadline)
    if result is not None:
        return result
    
    # 3. Fallback to Lazebnik-Ustimenko-Woldar
    return luw_construct(k, g)

def luw_construct(k, g):
    # Original seed logic, kept as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def try_cayley_sl2(q, k, g, deadline):
    if time.time() > deadline:
        return None
        
    # GF(q) arithmetic for q = 2^m
    m = int(math.log2(q))
    if (1 << m) != q:
        return None
        
    # Primitive polynomial for GF(16) and GF(32)
    # GF(16): x^4 + x + 1 is primitive
    # GF(32): x^5 + x^2 + 1 is primitive
    if q == 16:
        prim_poly = 0b10011 # x^4 + x + 1
    elif q == 32:
        prim_poly = 0b100101 # x^5 + x^2 + 1
    else:
        return None

    # Precompute addition and multiplication tables
    add_table = [[0] * q for _ in range(q)]
    mul_table = [[0] * q for _ in range(q)]
    
    for i in range(q):
        for j in range(q):
            add_table[i][j] = i ^ j
            # Multiplication in GF(2^m)
            res = 0
            a = i
            b = j
            for _ in range(m):
                if b & 1:
                    res ^= a
                b >>= 1
                # shift a left
                a <<= 1
                if a & (1 << m):
                    a ^= prim_poly
            mul_table[i][j] = res

    # Inverse table
    inv_table = [0] * q
    for i in range(1, q):
        for j in range(1, q):
            if mul_table[i][j] == 1:
                inv_table[i] = j
                break

    # Generate all matrices in SL(2, q)
    # Matrix is [[a, b], [c, d]] with ad - bc = 1
    # In char 2, ad + bc = 1
    matrices = []
    mat_to_idx = {}
    
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad + bc = 1 => ad = 1 + bc
                # If a == 0, then bc = 1 => c != 0, b != 0, d = 0? No, if a=0, bc=1 is impossible if we need d?
                # Wait, if a=0, then bc=1. Then d can be anything? No, det = 0*d - b*c = bc = 1.
                # So if a=0, we need bc=1. Then d is free?
                # No, the matrix is [[0, b], [c, d]]. Det = 0*d - bc = bc. So bc must be 1.
                # If a != 0, then d = (1 + bc) * a^{-1}.
                
                if a == 0:
                    if mul_table[b][c] == 1:
                        for d in range(q):
                            mat = (a, b, c, d)
                            if mat not in mat_to_idx:
                                mat_to_idx[mat] = len(matrices)
                                matrices.append(mat)
                else:
                    bc = mul_table[b][c]
                    # 1 + bc
                    rhs = add_table[1][bc]
                    d = mul_table[inv_table[a]][rhs]
                    mat = (a, b, c, d)
                    if mat not in mat_to_idx:
                        mat_to_idx[mat] = len(matrices)
                        matrices.append(mat)

    n = len(matrices)
    # n should be q(q^2 - 1)
    expected_n = q * (q*q - 1)
    if n != expected_n:
        return None

    # Define generators
    # A = [[0, 1], [1, 0]]
    # B = [[1, 1], [0, 1]]
    # C = [[1, 0], [1, 1]]
    A = (0, 1, 1, 0)
    B = (1, 1, 0, 1)
    C = (1, 0, 1, 1)
    
    # Check if generators are in the group (they should be, det=1)
    # det(A) = 0*0 - 1*1 = 1 (in char 2, -1=1)
    # det(B) = 1*1 - 1*0 = 1
    # det(C) = 1*1 - 0*1 = 1
    
    # Map generators to indices
    gen_indices = []
    for gen in [A, B, C]:
        if gen in mat_to_idx:
            gen_indices.append(mat_to_idx[gen])
        else:
            return None
            
    if len(gen_indices) != 3:
        return None

    # Build adjacency list
    adj = [[] for _ in range(n)]
    edges = set()
    
    for i in range(n):
        mat_i = matrices[i]
        for gen_idx in gen_indices:
            gen = matrices[gen_idx]
            # Multiply mat_i * gen
            a, b, c, d = mat_i
            ga, gb, gc, gd = gen
            # [[a, b], [c, d]] * [[ga, gb], [gc, gd]]
            # = [[a*ga + b*gc, a*gb + b*gd], [c*ga + d*gc, c*gb + d*gd]]
            p00 = add_table[mul_table[a][ga]][mul_table[b][gc]]
            p01 = add_table[mul_table[a][gb]][mul_table[b][gd]]
            p10 = add_table[mul_table[c][ga]][mul_table[d][gc]]
            p11 = add_table[mul_table[c][gb]][mul_table[d][gd]]
            prod = (p00, p01, p10, p11)
            j = mat_to_idx.get(prod, -1)
            if j != -1:
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
                adj[i].append(j)
                
    # Verify girth
    # Girth is the length of the shortest cycle.
    # We can run BFS from each node, but that's O(n^2).
    # Better: BFS from each node up to depth g/2?
    # Or just check if any word of length < g is identity.
    # Since it's a Cayley graph, girth is min length of non-trivial word in S equal to I.
    # We can do a BFS in the group starting from Identity, tracking the path.
    # If we reach Identity again with path length L > 0, we have a cycle of length L.
    # We need to ensure no such L < g.
    
    identity = (1, 0, 0, 1)
    if identity not in mat_to_idx:
        return None
    id_idx = mat_to_idx[identity]
    
    # BFS from identity
    # State: (current_node, prev_node, path)
    # We want to find the shortest path from identity to identity that is not just the start.
    # Actually, we just need to ensure that the first time we return to identity is at depth >= g.
    
    visited = {} # node -> (prev, depth)
    queue = [(id_idx, -1, 0)]
    visited[id_idx] = (-1, 0)
    
    min_girth = float('inf')
    
    while queue:
        node, prev, depth = queue.pop(0)
        if depth > g: # We only care about cycles shorter than g
            break
            
        for neighbor in adj[node]:
            if neighbor == id_idx and depth > 0:
                # Found a cycle of length depth + 1
                # But wait, BFS ensures shortest path.
                # The first time we hit identity from a neighbor, the path length is depth+1.
                # However, we must not count the "back" edge to prev if prev is identity?
                # No, standard BFS for girth in Cayley graph:
                # We start at Id. We look for shortest non-trivial word.
                # If we reach Id at depth d, and d>0, then girth is d.
                # But we must ensure the path doesn't just go Id -> A -> Id (length 2).
                # A is an involution, so Id -> A -> Id is a cycle of length 2?
                # No, in a simple graph, an edge (u, v) is a cycle of length 1? No.
                # A cycle must have length >= 3 in a simple graph?
                # No, a 2-cycle is u-v-u. This corresponds to word s s^{-1}.
                # Since our generators are involutions, s = s^{-1}.
                # So s s is a word of length 2 equal to Id.
                # This would imply girth 2.
                # But the graph is simple, so there is no edge (u, u).
                # The "cycle" u-v-u is just traversing the edge back and forth.
                # In graph theory, girth is the length of the shortest *simple* cycle.
                # A simple cycle cannot visit the same vertex twice.
                # So u-v-u is not a simple cycle.
                # We need to find the shortest simple cycle.
                
                # To find girth correctly in a Cayley graph with involutions:
                # We perform a BFS from Identity, but we don't allow immediate backtracking.
                # i.e., we don't go from u to v if v is the node we came from.
                
                pass # Handled below
            
            if neighbor not in visited:
                visited[neighbor] = (node, depth + 1)
                queue.append((neighbor, node, depth + 1))
            elif neighbor != prev:
                # We found a cycle
                # The cycle length is depth + 1 + (depth of neighbor) ?
                # No, standard girth BFS:
                # If we see a visited node that is not our parent, we have a cycle.
                # Length = current_depth + 1 + visited_depth[neighbor]
                # But this might not be the *shortest* cycle in the whole graph, just one.
                # We want the minimum such length.
                
                cycle_len = depth + 1 + visited[neighbor][1]
                if cycle_len < min_girth:
                    min_girth = cycle_len
                
    # The BFS above is slightly flawed for finding the *global* minimum girth efficiently without storing full paths or using a different algorithm.
    # However, for small n=4080, we can just run BFS from every node? No, too slow.
    # Better: The girth of a Cayley graph is the length of the shortest non-trivial word.
    # We can enumerate words up to length g.
    # Number of words of length L is 3 * 2^(L-1).
    # For L=15, this is 3 * 2^14 = 49152. This is small!
    
    # Let's re-evaluate girth by checking words.
    # Word is a sequence of generators.
    # We check if product is Identity.
    # We must exclude trivial words like s s (length 2).
    # A word is "non-trivial" if it's not of the form s s ... s where all s are the same?
    # No, we just need the shortest word equal to Id that is not "backtracking" in the free group sense?
    # Actually, if a word w evaluates to Id, it represents a cycle.
    # The length of the cycle is the length of w, provided w is reduced (no s s^{-1} pairs).
    # Since generators are involutions, s s^{-1} = s s = Id.
    # So we must avoid words with adjacent identical generators.
    
    def check_girth_by_words(generators, target_girth):
        # generators: list of indices
        # We generate all words of length L < target_girth
        # that do not have adjacent identical generators.
        # If any word evaluates to Identity, girth is L.
        
        def eval_word(word):
            # word is list of generator indices
            a, b, c, d = 1, 0, 0, 1 # Identity
            for gen_idx in word:
                ga, gb, gc, gd = matrices[gen_idx]
                # Multiply current (a,b,c,d) by gen
                p00 = add_table[mul_table[a][ga]][mul_table[b][gc]]
                p01 = add_table[mul_table[a][gb]][mul_table[b][gd]]
                p10 = add_table[mul_table[c][ga]][mul_table[d][gc]]
                p11 = add_table[mul_table[c][gb]][mul_table[d][gd]]
                a, b, c, d = p00, p01, p10, p11
            return (a, b, c, d) == identity

        for L in range(3, target_girth): # Cycles of length < 3 are impossible in simple graph
            # Generate all words of length L with no adjacent identical generators
            # Use recursion or iterative
            words = []
            def gen_words(prefix, last_gen):
                if len(prefix) == L:
                    if eval_word(prefix):
                        return True
                    return False
                for g_idx in gen_indices:
                    if g_idx != last_gen:
                        if gen_words(prefix + [g_idx], g_idx):
                            return True
                return False
                
            # Start with any generator
            for g_idx in gen_indices:
                if gen_words([g_idx], g_idx):
                    return L
        return target_girth # If no shorter cycle found

    actual_girth = check_girth_by_words(gen_indices, g)
    
    if actual_girth >= g:
        return n, list(edges)
    else:
        return None
